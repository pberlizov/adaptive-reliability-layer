#!/usr/bin/env python3
"""Validate sidecar deployment security settings before go-live."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ARL sidecar security configuration.")
    parser.add_argument("--config", default="configs/serving_pilot_fraud_torch.yaml")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    from adaptive_reliability_layer.runtime.config import RuntimeConfig
    from adaptive_reliability_layer.serving.config import load_serving_config_from_yaml
    from adaptive_reliability_layer.serving.security import validate_deployment_security

    raw, serving = load_serving_config_from_yaml(args.config)
    runtime = RuntimeConfig.from_mapping(raw)
    force_shadow = os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"}

    errors: list[str] = []
    warnings: list[str] = []

    try:
        validate_deployment_security(
            api_key=serving.api_key,
            require_api_key=serving.require_api_key,
            operating_mode=runtime.operating_mode.value,
            environment=runtime.governance.environment,
            force_shadow=force_shadow,
        )
    except Exception as exc:
        errors.append(str(exc))

    if not serving.api_key and runtime.governance.environment not in {"development", "dev", "test", "local"}:
        warnings.append("api_key is not configured for a non-dev environment")

    if not serving.admin_api_key:
        warnings.append("admin_api_key is not set; privileged routes use the regular api_key")

    if "/metrics" in serving.public_paths and serving.api_key:
        warnings.append("/metrics is public; remove from public_paths if scrapers can authenticate")

    if not force_shadow and runtime.operating_mode.value in {"bounded_auto", "recommend"}:
        warnings.append(f"operating_mode={runtime.operating_mode.value} allows mutations")

    for message in warnings:
        print(f"WARN: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors or (args.strict and warnings):
        return 1

    print("Security check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
