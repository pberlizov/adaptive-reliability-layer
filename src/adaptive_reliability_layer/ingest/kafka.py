from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

import numpy as np

from ..runtime.types import RuntimeBatch
from ..serving.state import ServingState
from .contract import IngestEvent


@dataclass(frozen=True)
class KafkaIngestConfig:
    bootstrap_servers: str = "localhost:9092"
    events_topic: str = "arl.events"
    labels_topic: str | None = "arl.labels"
    decisions_topic: str | None = None
    group_id: str = "arl-sidecar"
    auto_offset_reset: str = "earliest"
    poll_timeout_seconds: float = 1.0
    security_protocol: str | None = None
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_ca_location: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> KafkaIngestConfig:
        if not data:
            return cls()
        import os

        fields = cls.__dataclass_fields__
        filtered = {key: data[key] for key in data if key in fields}
        if not filtered.get("sasl_password"):
            env_password = os.environ.get("ARL_KAFKA_SASL_PASSWORD")
            if env_password:
                filtered["sasl_password"] = env_password
        return cls(**filtered)


def build_kafka_client_config(config: KafkaIngestConfig) -> dict[str, str]:
    client_config: dict[str, str] = {
        "bootstrap.servers": config.bootstrap_servers,
    }
    if config.security_protocol:
        client_config["security.protocol"] = config.security_protocol
    if config.sasl_mechanism:
        client_config["sasl.mechanism"] = config.sasl_mechanism
    if config.sasl_username:
        client_config["sasl.username"] = config.sasl_username
    if config.sasl_password:
        client_config["sasl.password"] = config.sasl_password
    if config.ssl_ca_location:
        client_config["ssl.ca.location"] = config.ssl_ca_location
    return client_config


def parse_kafka_payload(payload: dict[str, Any]) -> IngestEvent:
    """Parse a Kafka JSON value into a canonical ingest event."""

    if "event_id" not in payload and "batch_id" in payload:
        payload = dict(payload)
        payload["event_id"] = payload["batch_id"]

    if "event_id" not in payload:
        raise ValueError("event_id or batch_id is required")

    features = np.asarray(payload["features"], dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    label = payload.get("label")
    labels = payload.get("labels")
    resolved_label: int | None
    if label is not None:
        resolved_label = int(label)
    elif labels is not None and len(labels) == 1:
        resolved_label = int(labels[0])
    else:
        resolved_label = None

    if features.shape[0] > 1:
        raise ValueError(
            "single-event Kafka messages must use 1-D features; "
            "use ingest_batch_payload for multi-row batches"
        )

    return IngestEvent(
        event_id=str(payload["event_id"]),
        timestamp=str(payload.get("timestamp", datetime.now(UTC).isoformat())),
        features=features.reshape(-1),
        label=resolved_label,
        regime_id=str(payload.get("regime_id", payload.get("regime", "live"))),
        metadata=payload.get("metadata"),
    )


def ingest_batch_payload(payload: dict[str, Any]) -> RuntimeBatch:
    """Parse a multi-row batch message (batch_id + 2-D features)."""

    batch_id = payload.get("batch_id") or payload.get("event_id")
    if batch_id is None:
        raise ValueError("batch_id is required for batch payloads")

    features = np.asarray(payload["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("batch features must be a 2-D array")

    labels = payload.get("labels")
    label_array = None if labels is None else np.asarray(labels, dtype=np.int64)
    metadata = dict(payload.get("metadata") or {})
    metadata["batch_id"] = str(batch_id)
    metadata["ingest_source"] = "kafka"

    return RuntimeBatch(
        features=features,
        labels=label_array,
        regime=str(payload.get("regime_id", payload.get("regime", "live"))),
        timestamp=payload.get("timestamp"),
        metadata=metadata,
    )


def ingest_event_to_batch(event: IngestEvent) -> RuntimeBatch:
    metadata = dict(event.metadata or {})
    metadata["batch_id"] = event.event_id
    metadata["ingest_source"] = "kafka"
    labels = None if event.label is None else np.asarray([event.label], dtype=np.int64)
    return RuntimeBatch(
        features=event.features.reshape(1, -1),
        labels=labels,
        regime=event.regime_id,
        timestamp=event.timestamp,
        metadata=metadata,
    )


def parse_labels_payload(payload: dict[str, Any]) -> tuple[str, np.ndarray]:
    batch_id = payload.get("batch_id") or payload.get("event_id")
    if batch_id is None:
        raise ValueError("batch_id is required for label messages")
    labels = payload.get("labels")
    if labels is None:
        raise ValueError("labels field is required")
    return str(batch_id), np.asarray(labels, dtype=np.int64)


@dataclass
class KafkaIngestProcessor:
    """Bridge Kafka messages to ServingState (score + delayed label reveal)."""

    state: ServingState
    config: KafkaIngestConfig
    on_decision: Callable[[dict[str, Any]], None] | None = None

    def handle_payload(self, payload: dict[str, Any], *, topic: str) -> dict[str, Any]:
        labels_topic = self.config.labels_topic
        if labels_topic and topic == labels_topic:
            batch_id, labels = parse_labels_payload(payload)
            result = self.state.reveal_labels(batch_id=batch_id, step=None, labels=labels)
            return {"kind": "labels", "batch_id": batch_id, "metrics": result}

        if _is_batch_payload(payload):
            batch = ingest_batch_payload(payload)
        else:
            batch = ingest_event_to_batch(parse_kafka_payload(payload))

        record = self.state.process_batch(batch)
        if self.on_decision is not None:
            self.on_decision(record)
        return {"kind": "batch", "record": record}

    def handle_message_value(self, raw: bytes | str, *, topic: str) -> dict[str, Any]:
        if isinstance(raw, bytes):
            payload = json.loads(raw.decode("utf-8"))
        else:
            payload = json.loads(raw)
        return self.handle_payload(payload, topic=topic)


def _is_batch_payload(payload: dict[str, Any]) -> bool:
    features = payload.get("features")
    if features is None:
        return False
    array = np.asarray(features)
    return array.ndim == 2 and array.shape[0] > 1


def run_kafka_consumer_loop(
    processor: KafkaIngestProcessor,
    *,
    max_messages: int | None = None,
) -> int:
    """Block and consume Kafka topics until max_messages or KeyboardInterrupt."""

    try:
        from confluent_kafka import Consumer, Producer
    except ImportError as exc:
        raise ImportError(
            "confluent-kafka required for live Kafka ingest. "
            "Install with: pip install -e '.[kafka]'"
        ) from exc

    topics = [processor.config.events_topic]
    if processor.config.labels_topic:
        topics.append(processor.config.labels_topic)

    consumer = Consumer(
        {
            **build_kafka_client_config(processor.config),
            "group.id": processor.config.group_id,
            "auto.offset.reset": processor.config.auto_offset_reset,
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe(topics)

    producer = None
    decisions_topic = processor.config.decisions_topic
    if decisions_topic:

        def publish(record: dict[str, Any]) -> None:
            nonlocal producer
            if producer is None:
                producer = Producer(build_kafka_client_config(processor.config))
            producer.produce(decisions_topic, json.dumps(record).encode("utf-8"))
            producer.poll(0)

        processor.on_decision = publish

    processed = 0
    try:
        while max_messages is None or processed < max_messages:
            message = consumer.poll(processor.config.poll_timeout_seconds)
            if message is None:
                continue
            if message.error():
                raise RuntimeError(str(message.error()))
            processor.handle_message_value(message.value(), topic=message.topic())
            processed += 1
    finally:
        consumer.close()
        if producer is not None:
            producer.flush()

    return processed
