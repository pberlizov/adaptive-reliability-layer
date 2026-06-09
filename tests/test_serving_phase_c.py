from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adaptive_reliability_layer.ingest.kafka import (
    KafkaIngestConfig,
    KafkaIngestProcessor,
    ingest_event_to_batch,
    parse_kafka_payload,
    parse_labels_payload,
)
from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay
from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.policy_state import export_policy_state
from adaptive_reliability_layer.runtime.policy_state_store import FilePolicyStateStore, build_policy_state_store
from adaptive_reliability_layer.runtime.types import OperatingMode
from adaptive_reliability_layer.serving.config import ServingConfig
from adaptive_reliability_layer.serving.state import ServingState


@pytest.fixture
def trained_layer(tmp_path):
    base = load_runtime_config("configs/default.yaml")
    config = RuntimeConfig(
        operating_mode=OperatingMode.SHADOW,
        bounded_auto_actions=base.bounded_auto_actions,
        model_version="test-v1",
        monitor=base.monitor,
        policy=base.policy,
        governance=base.governance.__class__(
            audit_db_path=str(tmp_path / "audit.db"),
            snapshot_dir=str(tmp_path / "snapshots"),
            max_snapshots=5,
            policy_version="test",
            environment="test",
        ),
        metrics=base.metrics.__class__(enabled=False, prometheus_port=9091, namespace="arl_test"),
        replay=replace(base.replay, label_delay_steps=2),
        log_json=False,
    )
    return build_layer_for_tabular_replay(config=config)


def test_parse_kafka_payload_single_event():
    event = parse_kafka_payload(
        {
            "event_id": "evt-1",
            "features": [0.1, 0.2, 0.3],
            "regime_id": "live",
        }
    )
    assert event.event_id == "evt-1"
    assert event.features.shape == (3,)


def test_parse_kafka_payload_batch_id_alias():
    event = parse_kafka_payload(
        {
            "batch_id": "evt-2",
            "features": [1.0, 2.0],
        }
    )
    assert event.event_id == "evt-2"


def test_parse_labels_payload():
    batch_id, labels = parse_labels_payload({"batch_id": "b1", "labels": [0, 1, 0]})
    assert batch_id == "b1"
    assert labels.tolist() == [0, 1, 0]


def test_kafka_processor_batch_and_labels(trained_layer):
    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    state = ServingState(layer=layer, serving=ServingConfig(allow_duplicate_batch_id=False))
    processor = KafkaIngestProcessor(
        state=state,
        config=KafkaIngestConfig(events_topic="arl.events", labels_topic="arl.labels"),
    )

    batch_id = str(uuid.uuid4())
    scored = processor.handle_payload(
        {
            "batch_id": batch_id,
            "features": np.random.randn(4, 30).astype(float).tolist(),
            "regime": "live",
        },
        topic="arl.events",
    )
    assert scored["kind"] == "batch"
    assert scored["record"]["batch_id"] == batch_id

    revealed = processor.handle_payload(
        {"batch_id": batch_id, "labels": [0, 1, 0, 1]},
        topic="arl.labels",
    )
    assert revealed["kind"] == "labels"
    assert "batch_accuracy" in revealed["metrics"]


def test_policy_state_atomic_save(tmp_path: Path):
    from adaptive_reliability_layer.runtime.policy_state import save_policy_state_atomic
    from adaptive_reliability_layer.tabular_benchmark import RegimeAwareDelayedBanditTabularPolicy

    reference = type("Ref", (), {"mean_confidence": 0.7, "mean_probability": 0.5})()
    policy = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    path = tmp_path / "policy.json"
    save_policy_state_atomic(policy, path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_file_policy_state_store_round_trip(tmp_path: Path):
    from adaptive_reliability_layer.tabular_benchmark import RegimeAwareDelayedBanditTabularPolicy

    reference = type("Ref", (), {"mean_confidence": 0.7, "mean_probability": 0.5})()
    policy = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    path = tmp_path / "policy.json"
    store = FilePolicyStateStore(load_path=str(path), save_path=str(path))
    store.save(policy)
    policy2 = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    assert store.load(policy2)
    assert export_policy_state(policy2)["kind"] == "regime_aware_delayed_bandit"


def test_build_policy_state_store_file_backend(tmp_path: Path):
    config = RuntimeConfig(
        policy_state_save_path=str(tmp_path / "policy.json"),
        log_json=False,
    )
    store = build_policy_state_store(config)
    assert store is not None


def test_ingest_event_to_batch_uses_event_id_as_batch_id():
    from adaptive_reliability_layer.ingest.contract import IngestEvent

    event = IngestEvent(
        event_id="evt-99",
        timestamp="2025-01-01T00:00:00Z",
        features=np.zeros(5, dtype=np.float32),
    )
    batch = ingest_event_to_batch(event)
    assert batch.metadata["batch_id"] == "evt-99"
