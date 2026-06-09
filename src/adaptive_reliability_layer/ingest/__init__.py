"""Stream ingest contracts for production-style event delivery."""

from .contract import IngestEvent, events_to_replay_stream, load_events_csv, load_events_jsonl
from .kafka import (
    KafkaIngestConfig,
    KafkaIngestProcessor,
    ingest_event_to_batch,
    parse_kafka_payload,
    parse_labels_payload,
    run_kafka_consumer_loop,
)

__all__ = [
    "IngestEvent",
    "KafkaIngestConfig",
    "KafkaIngestProcessor",
    "events_to_replay_stream",
    "ingest_event_to_batch",
    "load_events_csv",
    "load_events_jsonl",
    "parse_kafka_payload",
    "parse_labels_payload",
    "run_kafka_consumer_loop",
]
