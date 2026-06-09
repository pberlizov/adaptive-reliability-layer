from __future__ import annotations

import uuid

from ..runtime.config import RuntimeConfig
from ..replay.real_data import RealDataBundle
from .config import ServingConfig


def verify_serving_http_workflow(
    bundle: RealDataBundle,
    *,
    runtime_config: RuntimeConfig,
    serving: ServingConfig | None = None,
) -> tuple[bool, str]:
    """Exercise sidecar HTTP workflow on a real-data bundle (TestClient, no live server)."""

    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return False, "fastapi/httpx not installed (pip install -e '.[serving]')"

    from .app import create_app

    serving_config = serving or ServingConfig(allow_duplicate_batch_id=False)
    layer = bundle.build_layer(runtime_config)
    layer._expected_feature_dim = bundle.feature_dim  # type: ignore[attr-defined]
    app = create_app(layer=layer, serving=serving_config, runtime_config=runtime_config)
    client = TestClient(app)

    if client.get("/v1/ready").status_code != 200:
        return False, "ready endpoint failed"

    records = bundle.stream.records[: min(16, bundle.stream_size)]
    batch_size = min(runtime_config.replay.batch_size, len(records))
    if batch_size <= 0:
        return False, "empty stream"

    chunk = records[:batch_size]
    batch_id = str(uuid.uuid4())
    features = [record.features.tolist() for record in chunk]
    labels = [int(record.label) for record in chunk]
    regime = str(chunk[0].metadata.get("regime", "live"))

    score = client.post(
        "/v1/batch",
        json={"batch_id": batch_id, "features": features, "regime": regime},
    )
    if score.status_code != 200:
        return False, f"batch score failed: {score.status_code} {score.text}"

    reveal = client.post(f"/v1/batches/{batch_id}/labels", json={"labels": labels})
    if reveal.status_code != 200:
        return False, f"label reveal failed: {reveal.status_code} {reveal.text}"

    audit = client.get("/v1/audit/recent?limit=5")
    if audit.status_code != 200 or not audit.json().get("records"):
        return False, "audit recent empty after batch"

    return True, f"HTTP batch+labels+audit OK on {bundle.source_id} ({batch_size} rows)"
