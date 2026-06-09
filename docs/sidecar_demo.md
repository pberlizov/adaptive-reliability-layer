# HTTP sidecar demo (Milestone 3)

> **Production guide:** see [sidecar_production.md](sidecar_production.md) for BYOM, batch_id, readiness, Docker, and parity checks.

ARL runs beside your model: score a batch, return a decision record, reveal labels later when they arrive.

## Prerequisites

```bash
cd ~/Documents/GitHub/adaptive-reliability-layer
source .venv/bin/activate
pip install -e ".[serving]"
```

For delayed labels, use a config with `replay.label_delay_steps > 0` (copy `configs/pilot_fraud_sklearn.yaml` or set in `configs/default.yaml`).

## Start server

```bash
python3 scripts/run_serve.py --config configs/default.yaml --port 8080
```

## Curl sequence

**1. Health**

```bash
curl -s http://127.0.0.1:8080/v1/health | python3 -m json.tool
```

**2. Score a batch** (default tabular layer expects **30** features per row)

```bash
curl -s -X POST http://127.0.0.1:8080/v1/batch \
  -H 'Content-Type: application/json' \
  -d '{"features": [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "regime": "live"}' \
  | python3 -m json.tool
```

Note the `"step"` field in the response (e.g. `0`).

**3. Reveal labels** (when `label_delay_steps > 0`)

```bash
curl -s -X POST http://127.0.0.1:8080/v1/batch/0/labels \
  -H 'Content-Type: application/json' \
  -d '{"labels": [0]}' \
  | python3 -m json.tool
```

**4. Metrics**

```bash
curl -s http://127.0.0.1:8080/v1/metrics | python3 -m json.tool
```

## Recommend mode approval

Set `operating_mode: recommend` in your config, restart the server, then:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/approve \
  -H 'Content-Type: application/json' \
  -d '{"features": [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "regime": "live", "approved_action": "label_shift", "approver": "demo-user"}' \
  | python3 -m json.tool
```

## Automated smoke

```bash
pytest tests/test_tier12_product.py::test_fastapi_health_endpoint -q
```
