from __future__ import annotations

import uuid

import numpy as np
import pytest

from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay
from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.types import OperatingMode, RuntimeBatch
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
        replay=base.replay.__class__(batch_size=16, max_steps=6, label_delay_steps=0),
        log_json=False,
    )
    return build_layer_for_tabular_replay(config=config)


@pytest.fixture
def serving_state(trained_layer):
    from adaptive_reliability_layer.runtime.config import RuntimeConfig

    config = RuntimeConfig(operating_mode=OperatingMode.SHADOW, log_json=False)
    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    return ServingState(
        layer=layer,
        serving=ServingConfig(max_batch_rows=32, allow_duplicate_batch_id=False),
    )


def test_reveal_labels_by_batch_id(trained_layer):
    from dataclasses import replace

    batch_id = "batch-001"
    features = np.random.randn(8, 30).astype(np.float32)
    trained_layer._config = replace(  # type: ignore[attr-defined]
        trained_layer._config,
        replay=replace(trained_layer._config.replay, label_delay_steps=2),
    )
    surface = trained_layer.process_batch(
        RuntimeBatch(features=features, labels=None, metadata={"batch_id": batch_id})
    )
    metrics = trained_layer.reveal_labels_by_batch_id(batch_id, np.zeros(8, dtype=np.int64))
    assert "batch_accuracy" in metrics
    assert metrics["batch_id"] == batch_id
    assert surface.batch_id == batch_id


def test_serving_state_idempotency(serving_state):
    batch_id = str(uuid.uuid4())
    features = np.random.randn(4, 30).astype(np.float32)
    batch = RuntimeBatch(features=features, metadata={"batch_id": batch_id})
    first = serving_state.process_batch(batch)
    second = serving_state.process_batch(batch)
    assert second["idempotent_replay"] is True
    assert first["step"] == second["step"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("fastapi") is None,
    reason="fastapi not installed",
)
def test_ready_and_batch_id_endpoints(trained_layer, tmp_path):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.runtime.config import RuntimeConfig
    from adaptive_reliability_layer.serving.app import create_app
    from adaptive_reliability_layer.serving.config import ServingConfig

    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    config = RuntimeConfig(
        operating_mode=OperatingMode.SHADOW,
        bounded_auto_actions=trained_layer._config.bounded_auto_actions,
        model_version="test-v1",
        monitor=trained_layer._config.monitor,
        policy=trained_layer._config.policy,
        governance=trained_layer._config.governance.__class__(
            audit_db_path=str(tmp_path / "audit2.db"),
            snapshot_dir=str(tmp_path / "snapshots2"),
            max_snapshots=5,
            policy_version="test",
            environment="test",
        ),
        metrics=trained_layer._config.metrics.__class__(enabled=False, prometheus_port=9091, namespace="arl_test"),
        replay=replace(trained_layer._config.replay, label_delay_steps=2),
        log_json=False,
    )
    layer._config = config  # type: ignore[attr-defined]
    serving = ServingConfig(allow_duplicate_batch_id=False)
    app = create_app(config_path="configs/default.yaml", layer=layer, serving=serving)
    client = TestClient(app)

    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/ready").status_code == 200

    batch_id = "evt-123"
    response = client.post(
        "/v1/batch",
        json={
            "batch_id": batch_id,
            "features": np.random.randn(4, 30).astype(np.float32).tolist(),
            "regime": "live",
        },
    )
    assert response.status_code == 200
    assert response.json()["batch_id"] == batch_id

    labels = client.post(f"/v1/batches/{batch_id}/labels", json={"labels": [0, 1, 0, 1]})
    assert labels.status_code == 200

    dup = client.post(
        "/v1/batch",
        json={
            "batch_id": batch_id,
            "features": np.random.randn(4, 30).astype(np.float32).tolist(),
        },
    )
    assert dup.json()["idempotent_replay"] is True
