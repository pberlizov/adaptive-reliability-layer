from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime.config import RuntimeConfig
from ..runtime.types import OperatingMode, RuntimeBatch
from .config import ServingConfig, load_serving_config_from_yaml
from .loader import build_layer_for_serving, expected_feature_dim
from .security import (
    AdminApiKeyGuard,
    ApiKeyGuard,
    ClientCertGuard,
    DeploymentSecurityError,
    RateLimiter,
    SecurityHeaders,
    TrustedHostGuard,
    check_request_body_size,
    is_admin_request,
    validate_batch_id,
    validate_deployment_security,
)
from .state import ServingState
from .validation import validate_batch_payload, validate_label_payload


def create_app(
    *,
    config_path: str | Path | None = "configs/default.yaml",
    layer: object | None = None,
    serving: ServingConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
):
    """Create a FastAPI production sidecar app."""

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for serving. Install with: pip install -e '.[serving]'"
        ) from exc

    if runtime_config is None:
        if config_path is None:
            raise ValueError("config_path or runtime_config is required")
        raw_config, serving_config = load_serving_config_from_yaml(str(config_path))
        runtime_config = RuntimeConfig.from_mapping(raw_config)
    else:
        serving_config = serving or ServingConfig()
    if serving is not None:
        serving_config = serving

    if os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"}:
        runtime_config = replace(runtime_config, operating_mode=OperatingMode.SHADOW)

    force_shadow = os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"}
    try:
        validate_deployment_security(
            api_key=serving_config.api_key,
            require_api_key=serving_config.require_api_key,
            operating_mode=runtime_config.operating_mode.value,
            environment=runtime_config.governance.environment,
            force_shadow=force_shadow,
            disable_openapi=serving_config.disable_openapi,
        )
    except DeploymentSecurityError as exc:
        raise RuntimeError(str(exc)) from exc

    service_layer = layer or build_layer_for_serving(runtime_config, serving_config)
    state = ServingState(layer=service_layer, serving=serving_config)
    api_guard = ApiKeyGuard(
        api_key=serving_config.api_key,
        header_name=serving_config.api_key_header,
        public_paths=serving_config.public_paths,
        additional_keys=(serving_config.admin_api_key,) if serving_config.admin_api_key else (),
    )
    rate_limiter = RateLimiter(requests_per_minute=serving_config.rate_limit_rpm)
    admin_guard = AdminApiKeyGuard(
        admin_api_key=serving_config.admin_api_key,
        fallback_api_key=serving_config.api_key,
        header_name=serving_config.api_key_header,
    )
    host_guard = TrustedHostGuard(allowed_hosts=serving_config.trusted_hosts)
    cert_guard = ClientCertGuard(
        trusted_client_cn=serving_config.trusted_client_cn,
        header_name=serving_config.client_cn_header,
    )
    security_headers = SecurityHeaders(enabled=serving_config.security_headers_enabled)

    docs_url = None if serving_config.disable_openapi else "/docs"
    redoc_url = None if serving_config.disable_openapi else "/redoc"
    openapi_url = None if serving_config.disable_openapi else "/openapi.json"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        yield

    app = FastAPI(
        title="Adaptive Reliability Layer",
        version="0.5.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    if serving_config.cors_allow_origins:
        try:
            from fastapi.middleware.cors import CORSMiddleware

            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(serving_config.cors_allow_origins),
                allow_methods=["GET", "POST"],
                allow_headers=["Authorization", serving_config.api_key_header, "Content-Type"],
                max_age=600,
            )
        except ImportError:
            pass

    if runtime_config.metrics.enabled and service_layer._metrics.registry is not None:
        try:
            from prometheus_client import make_asgi_app

            metrics_app = make_asgi_app(registry=service_layer._metrics.registry)
            app.mount(serving_config.prometheus_path, metrics_app)
        except ImportError:
            pass

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        if request.url.path.startswith(serving_config.prometheus_path):
            return await call_next(request)
        try:
            host_guard.verify(request)
            cert_guard.verify(request)
            check_request_body_size(request, serving_config.max_request_bytes)
            api_guard.verify(request)
            if is_admin_request(request.method, request.url.path):
                admin_guard.verify(request)
            rate_limiter.check()
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        response = await call_next(request)
        security_headers.apply(response)
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": service_layer._adapter.model_version,
            "operating_mode": service_layer.config.operating_mode.value,
            "force_shadow": os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"},
        }

    @app.get("/v1/ready")
    def ready() -> dict[str, Any]:
        dim = expected_feature_dim(service_layer)
        checks = {
            "layer_ready": service_layer.is_ready(),
            "adapter_kind": service_layer._adapter.adapter_kind,
            "feature_dim": dim,
            "policy_state_loaded": bool(
                service_layer.config.policy_state_path
                or service_layer.config.policy_state_backend == "redis"
            ),
        }
        ok = all(value for key, value in checks.items() if key != "policy_state_loaded")
        if not ok:
            raise HTTPException(status_code=503, detail={"ready": False, "checks": checks})
        return {"ready": True, "checks": checks}

    @app.post("/v1/batch")
    def process_batch(payload: dict[str, Any]) -> dict[str, Any]:
        features, labels, metadata = validate_batch_payload(
            payload,
            max_features=serving_config.max_feature_dim,
            max_batch_rows=serving_config.max_batch_rows,
        )
        batch_id = payload.get("batch_id")
        if batch_id is not None:
            metadata["batch_id"] = str(batch_id)
        batch = RuntimeBatch(
            features=features,
            labels=labels,
            regime=str(payload.get("regime_id", payload.get("regime", "live"))),
            timestamp=payload.get("timestamp"),
            metadata=metadata,
        )
        try:
            return state.process_batch(batch)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/batches/{batch_id}/labels")
    def reveal_labels_by_batch_id(batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        validate_batch_id(batch_id)
        labels = validate_label_payload(payload)
        try:
            return state.reveal_labels(
                batch_id=batch_id,
                step=None,
                labels=labels,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/batch/{step}/labels")
    def reveal_labels_by_step(step: int, payload: dict[str, Any]) -> dict[str, Any]:
        labels = validate_label_payload(payload)
        try:
            return state.reveal_labels(
                batch_id=None,
                step=step,
                labels=labels,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/approve")
    def approve(payload: dict[str, Any]) -> dict[str, Any]:
        if service_layer.config.operating_mode.value != "recommend":
            raise HTTPException(status_code=400, detail="layer is not in recommend mode")
        batch = _payload_to_batch(payload, require_batch_id=False)
        approved_action = str(payload.get("approved_action", "none"))
        approver = str(payload.get("approver", "api"))
        try:
            surface = service_layer.approve_and_apply(
                batch,
                approved_action=approved_action,
                approver=approver,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return surface.decision_record()

    @app.get("/v1/pending")
    def pending_recommendation() -> dict[str, Any]:
        pending = service_layer.pending_recommendation()
        if pending is None:
            return {"pending": False}
        return {"pending": True, "recommendation": pending}

    @app.post("/v1/operating-mode")
    def set_operating_mode(payload: dict[str, Any]) -> dict[str, Any]:
        mode_name = payload.get("mode")
        if mode_name is None:
            raise HTTPException(status_code=400, detail="mode field is required")
        try:
            mode = OperatingMode(str(mode_name))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid mode: {mode_name}") from exc
        return service_layer.set_operating_mode(mode)

    @app.get("/v1/operating-mode")
    def get_operating_mode() -> dict[str, Any]:
        return {
            "operating_mode": service_layer.config.operating_mode.value,
            "force_shadow_active": os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"},
        }

    @app.get("/v1/audit/recent")
    def audit_recent(limit: int = 50) -> dict[str, Any]:
        records = service_layer.governance.audit.fetch_recent(limit=max(1, min(limit, 1000)))
        return {"count": len(records), "records": records}

    @app.post("/v1/audit/export")
    def audit_export(payload: dict[str, Any]) -> dict[str, Any]:
        filename = str(payload.get("filename", "audit_export.jsonl"))
        if "/" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="invalid filename")
        export_dir = Path(serving_config.audit_export_dir)
        export_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = export_dir / filename
        service_layer.export_audit_jsonl(str(target))
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return {"path": str(target), "status": "exported"}

    @app.post("/v1/rollback/{snapshot_id}")
    def rollback(snapshot_id: str, payload: dict[str, Any]) -> dict[str, str]:
        if ".." in snapshot_id or "/" in snapshot_id:
            raise HTTPException(status_code=400, detail="invalid snapshot_id")
        actor = str(payload.get("actor", "api"))
        # Require explicit reason for all rollbacks — creates a clear audit trail.
        reason = payload.get("reason")
        if not reason or not str(reason).strip():
            raise HTTPException(
                status_code=400,
                detail="'reason' field is required for rollback (e.g. 'incident-123: silent corruption detected')",
            )
        # Require explicit acknowledgement that rollback is irreversible in bounded_auto.
        if (
            service_layer.config.operating_mode.value in {"bounded_auto", "recommend"}
            and not payload.get("confirm_irreversible")
        ):
            raise HTTPException(
                status_code=400,
                detail="set confirm_irreversible=true to acknowledge this rollback will discard all adaptations since the snapshot",
            )
        try:
            service_layer.rollback(snapshot_id, actor=f"{actor}:{str(reason)[:80]}")
        except (RuntimeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "rolled_back", "snapshot_id": snapshot_id, "actor": actor, "reason": reason}

    @app.get("/v1/metrics")
    def metrics_debug() -> dict[str, Any]:
        return {
            "revealed_batches": service_layer.revealed_metrics,
            "pending_delayed": service_layer.pending_delayed_count,
            "step": service_layer._step,
            "operating_mode": service_layer.config.operating_mode.value,
            "prometheus_path": serving_config.prometheus_path if runtime_config.metrics.enabled else None,
            "security": {
                "auth_failures": api_guard.auth_failures,
                "auth_successes": api_guard.auth_successes,
                "admin_failures": admin_guard.admin_failures,
                "admin_ops": admin_guard.admin_ops,
                "rate_limit_rpm": rate_limiter.requests_per_minute,
            },
            "governor": {
                "recent_decisions": service_layer._governor.decision_log[-10:],
            },
        }

    app.state.arl_service = state
    app.state.arl_layer = service_layer
    return app


def _payload_to_batch(payload: dict[str, Any], *, require_batch_id: bool) -> RuntimeBatch:
    if "features" not in payload:
        raise ValueError("features field is required")
    features = np.asarray(payload["features"], dtype=np.float32)
    labels = payload.get("labels")
    label_array = None if labels is None else np.asarray(labels, dtype=np.int64)
    metadata = dict(payload.get("metadata") or {})
    batch_id = payload.get("batch_id")
    if batch_id is not None:
        metadata["batch_id"] = str(batch_id)
    elif require_batch_id:
        raise ValueError("batch_id field is required")
    return RuntimeBatch(
        features=features,
        labels=label_array,
        regime=str(payload.get("regime_id", payload.get("regime", "live"))),
        timestamp=payload.get("timestamp"),
        metadata=metadata,
    )
