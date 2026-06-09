from __future__ import annotations

import uuid

import numpy as np
import pytest

from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay
from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.types import OperatingMode
from adaptive_reliability_layer.serving.config import ServingConfig
from adaptive_reliability_layer.serving.security import RateLimiter


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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("fastapi") is None,
    reason="fastapi not installed",
)
class TestServingPhaseB:
    def test_api_key_blocks_and_allows_health(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(api_key="secret-key", allow_duplicate_batch_id=True)
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)

        assert client.get("/v1/health").status_code == 200
        denied = client.post("/v1/batch", json={"features": [[0.0] * 30]})
        assert denied.status_code == 401
        allowed = client.post(
            "/v1/batch",
            json={"features": [[0.0] * 30]},
            headers={"X-API-Key": "secret-key"},
        )
        assert allowed.status_code == 200

    def test_rate_limiter(self):
        from fastapi import HTTPException

        limiter = RateLimiter(requests_per_minute=2)
        limiter.check()
        limiter.check()
        with pytest.raises(HTTPException) as exc_info:
            limiter.check()
        assert exc_info.value.status_code == 429

    def test_operating_mode_switch(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        app = create_app(layer=layer, serving=ServingConfig(), runtime_config=layer._config)
        client = TestClient(app)

        assert client.get("/v1/operating-mode").json()["operating_mode"] == "shadow"
        switched = client.post("/v1/operating-mode", json={"mode": "bounded_auto"})
        assert switched.status_code == 200
        assert switched.json()["operating_mode"] == "bounded_auto"

    def test_audit_recent_and_export(self, trained_layer, tmp_path):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(audit_export_dir=str(tmp_path / "exports"))
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)

        client.post("/v1/batch", json={"features": np.random.randn(4, 30).astype(float).tolist()})
        recent = client.get("/v1/audit/recent?limit=5")
        assert recent.status_code == 200
        assert recent.json()["count"] >= 1

        exported = client.post("/v1/audit/export", json={"filename": "test.jsonl"})
        assert exported.status_code == 200
        assert (tmp_path / "exports" / "test.jsonl").exists()

    def test_recommend_pending_and_approve(self, trained_layer):
        from dataclasses import replace

        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        config = replace(
            layer._config,
            operating_mode=OperatingMode.RECOMMEND,
            policy=replace(layer._config.policy, name="controller"),
        )
        layer._config = config  # type: ignore[attr-defined]
        app = create_app(layer=layer, serving=ServingConfig(), runtime_config=config)
        client = TestClient(app)

        features = np.random.randn(16, 30).astype(np.float32)
        batch = {
            "batch_id": str(uuid.uuid4()),
            "features": features.tolist(),
            "regime": "approval_test",
        }
        recommended = client.post("/v1/batch", json=batch)
        assert recommended.status_code == 200
        pending = client.get("/v1/pending")
        assert pending.status_code == 200
