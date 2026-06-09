#!/usr/bin/env python3
"""Verify HTTP sidecar workflow: batch_id, labels, idempotency, readiness."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from adaptive_reliability_layer.replay.real_data import load_paysim_fraud_torch_bundle
    from adaptive_reliability_layer.runtime.config import load_runtime_config
    from adaptive_reliability_layer.serving.app import create_app
    from adaptive_reliability_layer.serving.config import ServingConfig

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        raise SystemExit("pip install -e '.[serving]'") from exc

    runtime_config = load_runtime_config("configs/serving_pilot_fraud_torch.yaml")
    serving = ServingConfig(model_bundle="paysim_fraud_torch", allow_duplicate_batch_id=False)
    bundle = load_paysim_fraud_torch_bundle(steps=4, batch_size=8, stream_cycles=1)
    layer = bundle.build_layer(runtime_config)
    layer._expected_feature_dim = bundle.feature_dim  # type: ignore[attr-defined]

    app = create_app(config_path="configs/serving_pilot_fraud_torch.yaml", layer=layer, serving=serving)
    client = TestClient(app)

    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/ready").status_code == 200

    records = bundle.stream.records[:16]
    batch_ids: list[str] = []
    for index in range(0, len(records), 8):
        chunk = records[index : index + 8]
        batch_id = str(uuid.uuid4())
        batch_ids.append(batch_id)
        features = [record.features.tolist() for record in chunk]
        labels = [int(record.label) for record in chunk]
        regime = str(chunk[0].metadata.get("regime", "live"))

        response = client.post(
            "/v1/batch",
            json={"batch_id": batch_id, "features": features, "regime": regime},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["batch_id"] == batch_id
        assert "shift_score" in body
        assert "risk_capital" in body

        reveal = client.post(f"/v1/batches/{batch_id}/labels", json={"labels": labels})
        assert reveal.status_code == 200, reveal.text
        assert "batch_accuracy" in reveal.json()

    last_id = batch_ids[-1]
    last_chunk = records[8:16]
    dup = client.post(
        "/v1/batch",
        json={
            "batch_id": last_id,
            "features": [record.features.tolist() for record in last_chunk],
            "regime": "live",
        },
    )
    assert dup.status_code == 200
    assert dup.json().get("idempotent_replay") is True

    print(f"Sidecar workflow OK: {len(batch_ids)} batches scored + labels revealed + idempotency verified")


if __name__ == "__main__":
    main()
