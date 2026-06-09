from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServingConfig:
    """Production sidecar model loading and request limits."""

    model_bundle: str | None = None
    adapter_kind: str | None = None
    sklearn_model_path: str | None = None
    torch_checkpoint_path: str | None = None
    feature_dim: int | None = None
    reference_batches_path: str | None = None
    max_batch_rows: int = 512
    max_feature_dim: int = 4096
    max_pending_batches: int = 4096
    allow_duplicate_batch_id: bool = True
    prometheus_path: str = "/metrics"
    api_key: str | None = None
    api_key_header: str = "X-API-Key"
    rate_limit_rpm: int | None = None
    audit_export_dir: str = "results/sidecar/audit_exports"
    public_paths: tuple[str, ...] = ("/v1/health", "/v1/ready", "/metrics")
    require_api_key: bool = False
    admin_api_key: str | None = None
    max_request_bytes: int = 4_000_000
    trusted_hosts: tuple[str, ...] | None = None
    cors_allow_origins: tuple[str, ...] | None = None
    security_headers_enabled: bool = True
    disable_openapi: bool = True
    # Centralized audit fanout — optional external sinks
    audit_kafka_bootstrap: str | None = None      # e.g. "kafka:9092"
    audit_kafka_topic: str | None = None           # e.g. "arl.audit"
    audit_jsonl_sink_path: str | None = None       # shared JSONL file path for multi-sidecar
    # mTLS client certificate CN validation (injected by Nginx/LB after TLS termination)
    trusted_client_cn: str | None = None           # e.g. "model-server"
    client_cn_header: str = "X-Client-Cert-CN"    # header Nginx injects: $ssl_client_s_dn_cn

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ServingConfig":
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in data if key in fields})


def load_serving_config_from_yaml(path: str) -> tuple[dict[str, Any], ServingConfig]:
    import os
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.pop("kafka", None)
    serving_data = dict(raw.pop("serving", None) or {})
    if serving_data.get("public_paths") is not None:
        serving_data["public_paths"] = tuple(serving_data["public_paths"])
    if serving_data.get("trusted_hosts") is not None:
        serving_data["trusted_hosts"] = tuple(serving_data["trusted_hosts"])
    if serving_data.get("cors_allow_origins") is not None:
        serving_data["cors_allow_origins"] = tuple(serving_data["cors_allow_origins"])
    env_key = os.environ.get("ARL_API_KEY")
    if env_key and not serving_data.get("api_key"):
        serving_data["api_key"] = env_key
    env_admin = os.environ.get("ARL_ADMIN_API_KEY")
    if env_admin and not serving_data.get("admin_api_key"):
        serving_data["admin_api_key"] = env_admin
    if os.environ.get("ARL_REQUIRE_API_KEY", "").lower() in {"1", "true", "yes"}:
        serving_data["require_api_key"] = True
    serving = ServingConfig.from_mapping(serving_data)
    return raw, serving
