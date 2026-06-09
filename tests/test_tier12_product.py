import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay, run_replay_on_stream
from adaptive_reliability_layer.replay.loader import ReplayRecord, ReplayStream
from adaptive_reliability_layer.runtime.config import ReplayConfig, RuntimeConfig
from adaptive_reliability_layer.runtime.policy_state import export_policy_state, load_policy_state
from adaptive_reliability_layer.runtime.types import OperatingMode, RuntimeBatch
from adaptive_reliability_layer.tabular_benchmark import RegimeAwareDelayedBanditTabularPolicy


def _tiny_stream(rows: int = 8, feature_dim: int = 30) -> ReplayStream:
    rng = np.random.default_rng(0)
    records = []
    for index in range(rows):
        records.append(
            ReplayRecord(
                timestamp=f"t{index}",
                features=rng.normal(size=feature_dim).astype(np.float32),
                label=int(rng.random() > 0.5),
                metadata={"regime": "live"},
            )
        )
    return ReplayStream(
        records=tuple(records),
        feature_columns=tuple(f"feature_{index}" for index in range(feature_dim)),
    )


def test_reveal_labels_updates_delayed_bandit_policy(tmp_path: Path):
    config = RuntimeConfig(
        operating_mode=OperatingMode.BOUNDED_AUTO,
        replay=ReplayConfig(batch_size=4, label_delay_steps=2, max_steps=2),
        policy=replace(RuntimeConfig().policy, name="delayed_bandit"),
        log_json=False,
    )
    layer = build_layer_for_tabular_replay(config=config)
    stream = _tiny_stream(rows=8)

    from adaptive_reliability_layer.replay.loader import iter_replay_batches

    pending = []
    for step, batch, _ in iter_replay_batches(stream, batch_size=4, max_steps=2, label_delay_steps=2):
        labels = np.asarray(batch.labels, dtype=np.int64)
        surface = layer.process_batch(
            RuntimeBatch(features=batch.features, labels=None, regime=batch.regime)
        )
        pending.append((surface.step, labels))

    before = export_policy_state(layer._policy)
    for reveal_step, labels in pending:
        layer.reveal_labels(reveal_step, labels)
    after = export_policy_state(layer._policy)
    assert before != after


def test_policy_state_round_trip():
    reference = type("Ref", (), {})()
    reference.mean_confidence = 0.7
    reference.mean_probability = 0.5
    policy = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    state = export_policy_state(policy)
    policy2 = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    load_policy_state(policy2, state)
    assert policy2._encoder_step == policy._encoder_step  # type: ignore[attr-defined]


def test_policy_state_save_file(tmp_path: Path):
    reference = type("Ref", (), {})()
    reference.mean_confidence = 0.7
    reference.mean_probability = 0.5
    policy = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    path = tmp_path / "policy.json"
    from adaptive_reliability_layer.runtime.policy_state import save_policy_state, load_policy_state_from_file

    save_policy_state(policy, path)
    policy2 = RegimeAwareDelayedBanditTabularPolicy(reference)  # type: ignore[arg-type]
    load_policy_state_from_file(policy2, path)
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "regime_aware_delayed_bandit"


def test_ingest_contract_csv_round_trip(tmp_path: Path):
    import pandas as pd

    from adaptive_reliability_layer.ingest.contract import events_to_replay_stream, load_events_csv

    frame = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z"],
            "label": [0, 1],
            "feature_0": [0.1, 0.2],
            "feature_1": [0.3, 0.4],
        }
    )
    path = tmp_path / "events.csv"
    frame.to_csv(path, index=False)
    events = load_events_csv(path)
    stream = events_to_replay_stream(events)
    assert len(stream.records) == 2


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("fastapi") is None,
    reason="fastapi not installed",
)
def test_fastapi_health_endpoint():
    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.serving.app import create_app

    client = TestClient(create_app(config_path="configs/default.yaml"))
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
