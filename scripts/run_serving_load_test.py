#!/usr/bin/env python3
"""Lightweight load test for the ARL sidecar (TestClient, no live server required)."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test ARL sidecar via TestClient.")
    parser.add_argument("--config", default="configs/serving_pilot_fraud_torch.yaml")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from adaptive_reliability_layer.replay.real_data import load_paysim_fraud_torch_bundle
    from adaptive_reliability_layer.runtime.config import load_runtime_config
    from adaptive_reliability_layer.serving.app import create_app
    from adaptive_reliability_layer.serving.config import ServingConfig, load_serving_config_from_yaml

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        raise SystemExit("pip install -e '.[serving]'") from exc

    raw, serving = load_serving_config_from_yaml(args.config)
    from adaptive_reliability_layer.runtime.config import RuntimeConfig

    runtime_config = RuntimeConfig.from_mapping(raw)
    bundle = load_paysim_fraud_torch_bundle(steps=4, batch_size=args.batch_size, stream_cycles=1)
    layer = bundle.build_layer(runtime_config)
    layer._expected_feature_dim = bundle.feature_dim  # type: ignore[attr-defined]
    app = create_app(layer=layer, serving=serving, runtime_config=runtime_config)
    client = TestClient(app)

    records = bundle.stream.records
    features_template = records[0].features.tolist()
    start = time.perf_counter()
    ok = 0
    for _ in range(args.requests):
        batch_id = str(uuid.uuid4())
        response = client.post(
            "/v1/batch",
            json={"batch_id": batch_id, "features": [features_template], "regime": "live"},
        )
        if response.status_code == 200:
            ok += 1
    elapsed = time.perf_counter() - start
    rps = args.requests / max(elapsed, 1e-6)
    print(f"Load test: {ok}/{args.requests} OK in {elapsed:.2f}s ({rps:.1f} req/s)")
    print(f"Pending delayed queue: {layer.pending_delayed_count}")
    if ok < args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
