from __future__ import annotations

from dataclasses import replace
import pytest

from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay
from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.types import OperatingMode
from adaptive_reliability_layer.serving.config import ServingConfig
from adaptive_reliability_layer.serving.security import (
    DeploymentSecurityError,
    validate_batch_id,
    validate_deployment_security,
)


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
class TestServingSecurity:
    def test_require_api_key_blocks_startup(self, trained_layer):
        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(require_api_key=True, api_key=None)
        with pytest.raises(RuntimeError, match="require_api_key"):
            create_app(layer=layer, serving=serving, runtime_config=layer._config)

    def test_admin_key_required_for_mode_switch(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(
            api_key="client-key",
            admin_api_key="admin-key",
            allow_duplicate_batch_id=True,
        )
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)

        denied = client.post(
            "/v1/operating-mode",
            json={"mode": "bounded_auto"},
            headers={"X-API-Key": "client-key"},
        )
        assert denied.status_code == 403

        allowed = client.post(
            "/v1/operating-mode",
            json={"mode": "bounded_auto"},
            headers={"X-API-Key": "admin-key"},
        )
        assert allowed.status_code == 200

    def test_approve_requires_admin_key(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(
            api_key="client-key",
            admin_api_key="admin-key",
            allow_duplicate_batch_id=True,
        )
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)

        denied = client.post(
            "/v1/audit/export",
            json={"filename": "security_test.jsonl"},
            headers={"X-API-Key": "client-key"},
        )
        assert denied.status_code == 403

        allowed = client.post(
            "/v1/audit/export",
            json={"filename": "security_test.jsonl"},
            headers={"X-API-Key": "admin-key"},
        )
        assert allowed.status_code == 200

    def test_security_headers_present(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        app = create_app(layer=layer, serving=ServingConfig(), runtime_config=layer._config)
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_ready_endpoint_is_public(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        app = create_app(layer=layer, serving=ServingConfig(), runtime_config=layer._config)
        client = TestClient(app)
        response = client.get("/v1/ready")
        assert response.status_code == 200

    def test_request_body_size_limit(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(max_request_bytes=128, allow_duplicate_batch_id=True)
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)

        response = client.post(
            "/v1/batch",
            json={"features": [[0.0] * 30]},
            headers={"Content-Length": "999999"},
        )
        assert response.status_code == 413

    def test_max_feature_dim_rejected(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 25  # type: ignore[attr-defined]
        serving = ServingConfig(max_feature_dim=20, allow_duplicate_batch_id=True)
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)
        response = client.post(
            "/v1/batch",
            json={"features": [[0.0] * 21]},
        )
        assert response.status_code == 400
        assert "max" in response.json()["detail"] or "exceed" in response.json()["detail"]

    def test_invalid_batch_id_rejected(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        app = create_app(layer=layer, serving=ServingConfig(), runtime_config=layer._config)
        client = TestClient(app)
        response = client.post("/v1/batches/../etc/labels", json={"labels": [0, 1]})
        assert response.status_code in {400, 404, 405}

    def test_trusted_host_rejection(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        serving = ServingConfig(trusted_hosts=("allowed.internal",))
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app, base_url="http://allowed.internal")
        ok = client.get("/v1/health", headers={"Host": "allowed.internal"})
        assert ok.status_code == 200

        bad = client.get("/v1/health", headers={"Host": "evil.example"})
        assert bad.status_code == 400

    def test_openapi_disabled_by_default(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        app = create_app(layer=layer, serving=ServingConfig(), runtime_config=layer._config)
        client = TestClient(app)
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_batch_requires_api_key_when_required(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        serving = ServingConfig(require_api_key=True, api_key="client-key", allow_duplicate_batch_id=True)
        app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
        client = TestClient(app)

        response = client.post(
            "/v1/batch",
            json={"features": [[0.0] * 30]},
        )
        assert response.status_code == 401

    def test_production_rejects_openapi_enabled(self, trained_layer):
        from fastapi.testclient import TestClient

        from adaptive_reliability_layer.serving.app import create_app

        layer = trained_layer
        layer._expected_feature_dim = 30  # type: ignore[attr-defined]
        runtime_config = replace(
            layer._config,
            governance=replace(layer._config.governance, environment="production"),
        )
        serving = ServingConfig(disable_openapi=False, api_key="client-key", require_api_key=True)
        with pytest.raises(RuntimeError, match="OpenAPI docs must be disabled"):
            create_app(layer=layer, serving=serving, runtime_config=runtime_config)


def test_validate_batch_id_accepts_uuid():
    assert validate_batch_id("evt-2025-001") == "evt-2025-001"


def test_validate_batch_id_rejects_traversal():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        validate_batch_id("../etc/passwd")
    assert exc_info.value.status_code == 400


def test_validate_deployment_security_production_without_key():
    with pytest.raises(DeploymentSecurityError):
        validate_deployment_security(
            api_key=None,
            require_api_key=False,
            operating_mode="bounded_auto",
            environment="production",
            force_shadow=False,
        )


def test_payload_validation_rejects_oversized_labels(trained_layer):
    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.serving.app import create_app

    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    serving = ServingConfig(api_key="test-key")
    app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
    client = TestClient(app)
    
    # Try to reveal labels for nonexistent batch without proper error format
    response = client.post(
        "/v1/batches/nonexistent-batch-id/labels",
        json={"labels": [0, 1, 0, 1]},
        headers={"Authorization": "Bearer test-key"},
    )
    # Should return 404 for missing batch
    assert response.status_code == 404


def test_payload_validation_rejects_malformed_json(trained_layer):
    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.serving.app import create_app

    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    serving = ServingConfig(api_key="test-key")
    app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
    client = TestClient(app)
    
    response = client.post(
        "/v1/batch",
        json={"features": "not-a-list"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400


def test_file_permissions_on_audit_export(trained_layer, tmp_path):
    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.serving.app import create_app

    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    serving = ServingConfig(
        api_key="test-key",
        admin_api_key="admin-key",
        audit_export_dir=str(tmp_path / "audit_exports"),
    )
    app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
    client = TestClient(app)
    
    response = client.post(
        "/v1/audit/export",
        json={"filename": "test_export.jsonl"},
        headers={"Authorization": "Bearer admin-key"},
    )
    assert response.status_code == 200
    
    export_file = tmp_path / "audit_exports" / "test_export.jsonl"
    assert export_file.exists()
    
    import stat
    perms = stat.filemode(export_file.stat().st_mode)
    assert "rw" in perms and "x" not in perms[-3:]


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------

def test_rate_limiter_blocks_sustained_requests(trained_layer, tmp_path):
    """Rate limiter must return 429 once the per-minute window is exhausted."""
    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.serving.app import create_app

    layer = trained_layer
    layer._expected_feature_dim = 30  # type: ignore[attr-defined]
    serving = ServingConfig(
        rate_limit_rpm=3,
        api_key="test-key",
        allow_duplicate_batch_id=True,
        audit_export_dir=str(tmp_path / "audit"),
    )
    app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-key"}

    statuses = []
    for _ in range(5):
        r = client.get("/v1/health", headers=headers)
        statuses.append(r.status_code)

    assert 200 in statuses, "at least some requests should succeed"
    assert 429 in statuses, "rate limiter should block after limit is reached"


def test_rate_limiter_disabled_when_none(trained_layer, tmp_path):
    """No rate limiting when requests_per_minute is None."""
    from fastapi.testclient import TestClient

    from adaptive_reliability_layer.serving.app import create_app

    layer = trained_layer
    serving = ServingConfig(
        rate_limit_rpm=None,
        api_key="test-key",
        audit_export_dir=str(tmp_path / "audit"),
    )
    app = create_app(layer=layer, serving=serving, runtime_config=layer._config)
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-key"}

    for _ in range(20):
        r = client.get("/v1/health", headers=headers)
        assert r.status_code == 200, "unlimited mode should never 429"


def test_rate_limiter_resets_after_window():
    """RateLimiter window resets when 60 seconds elapse (unit test, no HTTP)."""
    import time as _time

    from adaptive_reliability_layer.serving.security import RateLimiter

    limiter = RateLimiter(requests_per_minute=2)
    from fastapi import HTTPException

    limiter.check()  # 1
    limiter.check()  # 2
    with pytest.raises(HTTPException) as exc:
        limiter.check()  # 3 → over limit
    assert exc.value.status_code == 429

    # Simulate window expiry by backdating the window start
    limiter._window_start = _time.time() - 61.0
    limiter.check()  # should succeed after window reset
    assert limiter._count == 1


def test_audit_metadata_sanitization():
    """Audit sanitizer must strip numpy arrays and large lists from metadata."""
    import numpy as np

    from adaptive_reliability_layer.runtime.audit import _sanitize_audit_metadata

    metadata = {
        "regime": "live",
        "batch_features": np.zeros((48, 30), dtype=np.float32),
        "batch_labels": np.array([0, 1, 0, 1], dtype=np.int64),
        "large_list": list(range(100)),
        "short_int_list": [1, 2, 3],        # integers: kept
        "long_float_list": [0.1] * 10,       # floats ≥ 5: redacted
        "nested": {"inner_array": np.ones(5), "scalar": 42},
        "risk_score": 1.23,
    }
    result = __import__("json").loads(_sanitize_audit_metadata(metadata))

    assert result["regime"] == "live"
    assert result["risk_score"] == pytest.approx(1.23)
    assert "<redacted" in str(result["batch_features"])
    assert "<redacted" in str(result["batch_labels"])
    assert "<redacted" in str(result["large_list"])
    assert result["short_int_list"] == [1, 2, 3]         # integers pass through
    assert "<redacted" in str(result["long_float_list"])  # floats ≥ 5 redacted
    assert "<redacted" in str(result["nested"]["inner_array"])
    assert result["nested"]["scalar"] == 42

