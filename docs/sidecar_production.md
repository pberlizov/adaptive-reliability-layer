# Production sidecar guide

ARL deploys **beside** your model server: score batches, return decision records, reveal labels when they arrive, optionally apply bounded interventions.

## Quick start (PaySim fraud bundle)

```bash
cd ~/Documents/GitHub/adaptive-reliability-layer
source .venv/bin/activate
pip install -e ".[serving,prometheus]"

# Bundled fraud CSV (first time)
python3 scripts/export_bundled_fraud_data.py

# Shadow mode (zero mutations) — safe default for first deploy
python3 scripts/run_serve.py --config configs/serving_pilot_fraud_torch.yaml --force-shadow

# Or bounded_auto from config (mutations enabled)
python3 scripts/run_serve.py --config configs/serving_pilot_fraud_torch.yaml
```

Docker:

```bash
docker compose up arl-sidecar
# Default compose sets ARL_FORCE_SHADOW=1
```

## Configuration

Primary config: `configs/serving_pilot_fraud_torch.yaml`

### Runtime (same as offline replay)

- `operating_mode`: `shadow` | `recommend` | `bounded_auto`
- `bounded_auto_actions`, `safety_budget`, `policy`, `replay.label_delay_steps`
- `policy_state_path` / `policy_state_save_path` for restart-safe delayed learning
- `governance.audit_db_path`, `governance.snapshot_dir`

### Serving block (new)

```yaml
serving:
  model_bundle: paysim_fraud_torch   # or paysim_fraud, sklearn bundle ids
  # BYOM alternatives:
  # adapter_kind: sklearn
  # sklearn_model_path: /models/fraud.joblib
  # reference_batches_path: /models/reference_batches.pkl
  # adapter_kind: torch_tabular
  # torch_checkpoint_path: /models/checkpoint.pt
  # feature_dim: 42
  max_batch_rows: 256
  max_pending_batches: 4096
  allow_duplicate_batch_id: false
  prometheus_path: /metrics
```

### Kill switch

```bash
export ARL_FORCE_SHADOW=1   # or --force-shadow on run_serve.py
```

Forces shadow mode regardless of config — no model mutations.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/health` | Liveness |
| GET | `/v1/ready` | Readiness (adapter + feature dim) |
| POST | `/v1/batch` | Score batch; returns decision record |
| POST | `/v1/batches/{batch_id}/labels` | Delayed labels (preferred) |
| POST | `/v1/batch/{step}/labels` | Delayed labels by internal step (legacy) |
| POST | `/v1/approve` | Recommend-mode human approval |
| POST | `/v1/rollback/{snapshot_id}` | Roll back adapter to snapshot |
| GET | `/v1/metrics` | Debug JSON metrics |
| GET | `/metrics` | Prometheus scrape (when `metrics.enabled: true`) |

### Phase B: auth, rate limits, mode switch, audit export

Optional production controls in the `serving` block:

```yaml
serving:
  api_key: dev-sidecar-key          # or set ARL_API_KEY env
  api_key_header: X-API-Key
  rate_limit_rpm: 600               # 0 or omit to disable
  audit_export_dir: results/sidecar/audit_exports
  public_paths:
    - /v1/health
    - /metrics
```

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/operating-mode` | Current shadow / recommend / bounded_auto |
| POST | `/v1/operating-mode` | Switch mode at runtime (`{"mode": "bounded_auto"}`) |
| GET | `/v1/pending` | Pending recommend-mode approvals |
| GET | `/v1/audit/recent` | Recent audit records (JSON) |
| POST | `/v1/audit/export` | Export audit log to JSONL (`{"filename": "export.jsonl"}`) |

When `api_key` is set, all routes except `public_paths` require `X-API-Key: <key>` or `Authorization: Bearer <key>`.

Load test (TestClient, no live server):

```bash
python3 scripts/run_serving_load_test.py --requests 100
```

Verification suite includes a global **serving_http_workflow** check (batch → labels → audit) when real-data sources are available.

### Phase C: Kafka ingest, HA policy state, mTLS

**Kafka worker** (consume events without HTTP):

```yaml
# configs/kafka_ingest_pilot.yaml
kafka:
  bootstrap_servers: localhost:9092
  events_topic: arl.events
  labels_topic: arl.labels
  decisions_topic: arl.decisions
  group_id: arl-sidecar-pilot
```

```bash
pip install -e ".[kafka,serving]"

# Live Kafka (requires broker)
python3 scripts/run_kafka_ingest_worker.py --config configs/kafka_ingest_pilot.yaml

# Offline parity (JSONL, no broker)
python3 scripts/run_kafka_ingest_worker.py --jsonl data/events/sample.jsonl --force-shadow
```

Kafka message formats mirror the ingest contract:

- **Score:** `{"event_id": "...", "features": [...]}` or multi-row `{"batch_id": "...", "features": [[...], [...]]}`
- **Labels:** `{"batch_id": "...", "labels": [0, 1, ...]}` on `labels_topic`

**HA policy state** — shared bandit state across restarts / replicas:

```yaml
policy_state_backend: file          # default: atomic JSON file
policy_state_save_path: results/sidecar/policy_state.json

# Or Redis for multi-pod deployments (single writer recommended):
policy_state_backend: redis
policy_state_redis_url: redis://localhost:6379/0
policy_state_redis_key: arl:policy:deployment-1
```

Install Redis backend: `pip install -e ".[redis]"`. File saves use atomic `os.replace` to avoid torn writes.

**mTLS** — terminate TLS at the sidecar or ingress:

```bash
python3 scripts/run_serve.py \
  --config configs/serving_pilot_fraud_torch.yaml \
  --ssl-certfile certs/server.crt \
  --ssl-keyfile certs/server.key \
  --ssl-ca-certs certs/ca.crt
```

Environment equivalents: `ARL_SSL_CERTFILE`, `ARL_SSL_KEYFILE`, `ARL_SSL_CA_CERTS`. Combine with Phase B `api_key` for defense in depth; many deployments terminate mTLS at the ingress and keep HTTP + API key inside the cluster.

### Deployment security checklist

Production sidecar controls (enabled by default where noted):

| Control | Config / env | Purpose |
|---------|----------------|---------|
| API key auth | `api_key`, `ARL_API_KEY` | Block unauthenticated batch/label traffic |
| Require API key | `require_api_key: true`, `ARL_REQUIRE_API_KEY=1` | Fail startup if key missing |
| Admin key | `admin_api_key`, `ARL_ADMIN_API_KEY` | Separate credential for mode switch, rollback, audit export, approve |
| Rate limiting | `rate_limit_rpm` | Abuse / DoS mitigation |
| Request size cap | `max_request_bytes` (default 4MB) | Reject oversized payloads (413) |
| Security headers | `security_headers_enabled: true` | `nosniff`, `DENY` framing, CSP, no-store |
| Hide OpenAPI | `disable_openapi: true` | No `/docs` or `/openapi.json` in production |
| Trusted hosts | `trusted_hosts: [sidecar.internal]` | Reject invalid `Host` headers |
| CORS lockdown | omit `cors_allow_origins` | No cross-origin access unless explicitly allowed |
| Constant-time key compare | built-in | Timing-safe API key verification |
| Batch ID validation | built-in | Reject path traversal in `batch_id` |
| Non-root container | Dockerfile `USER arl` | Reduce container escape blast radius |
| Kafka SASL/TLS | `kafka.security_protocol`, `ARL_KAFKA_SASL_PASSWORD` | Encrypted/authenticated broker access |

Pre-flight check:

```bash
python3 scripts/check_serving_security.py --config configs/serving_pilot_fraud_torch.yaml --strict
```

**Recommended production env:**

```bash
export ARL_API_KEY="<rotate-regularly>"
export ARL_ADMIN_API_KEY="<separate-admin-credential>"
export ARL_REQUIRE_API_KEY=1
export ARL_FORCE_SHADOW=1          # until shadow validation completes
# export ARL_ALLOW_INSECURE=1    # local dev ONLY — never in prod
```

**Privileged routes** (require `admin_api_key` when set):

- `POST /v1/operating-mode`
- `POST /v1/rollback/{snapshot_id}`
- `POST /v1/audit/export`
- `POST /v1/approve`

Regular `api_key` remains sufficient for `/v1/batch`, label reveal, and read endpoints when `admin_api_key` is not configured.

### Score a batch (client-owned batch_id)

```bash
curl -s -X POST http://127.0.0.1:8080/v1/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "batch_id": "evt-2025-001",
    "features": [[0.1, 0.2, ...]],
    "regime": "live"
  }'
```

Response includes `batch_id`, `step`, `recommended_action`, `action_taken`, `risk_capital`, `shift_score`, etc.

With `allow_duplicate_batch_id: false`, repeating the same `batch_id` returns the cached decision (`idempotent_replay: true`) without double-enqueueing pending labels.

### Reveal labels

```bash
curl -s -X POST http://127.0.0.1:8080/v1/batches/evt-2025-001/labels \
  -H 'Content-Type: application/json' \
  -d '{"labels": [0, 1, 0]}'
```

Requires `replay.label_delay_steps > 0` in config when labels were withheld at score time.

### Readiness

```bash
curl -s http://127.0.0.1:8080/v1/ready
```

Returns 503 if the layer failed to initialize (wrong feature dim, missing model path).

## Bring your own model

| Path | Config |
|------|--------|
| PaySim torch (pilot) | `serving.model_bundle: paysim_fraud_torch` |
| PaySim sklearn | `serving.model_bundle: paysim_fraud` |
| Sklearn artifact | `adapter_kind: sklearn` + `sklearn_model_path` + optional `reference_batches_path` |
| Torch checkpoint | `adapter_kind: torch_tabular` + `torch_checkpoint_path` + `feature_dim` |
| Demo tabular (dev only) | omit `serving` block — trains internal 30-dim model |

Reference batches default to internal validation data if `reference_batches_path` is omitted (fine for demos; **required for production** custom models).

## Parity with offline replay

HTTP and offline replay should match on the same stream:

```bash
python3 scripts/run_serving_parity.py
```

Compares HTTP workflow (batch_id, labels, idempotency, readiness) on the PaySim torch bundle. For offline vs HTTP numeric parity, use the same model artifact via `serving.model_bundle` and compare audit logs to `run_ingest_replay.py` on the same CSV.

Ingest CSV path uses the same layer semantics:

```bash
python3 scripts/run_ingest_replay.py --input data/fraud/paysim.csv --dual-mode
```

## Policy persistence across restarts

1. Set `policy_state_save_path` in config (file backend) or `policy_state_backend: redis` with `policy_state_redis_url`.
2. On startup set `policy_state_path` to the same file (or rely on Redis key).
3. Labels revealed via HTTP or Kafka update bandit state and re-save when configured.

**Note:** Bandit policy updates are not safe under concurrent writers. Run a single writer per deployment ID, or use Kafka partition affinity / leader election.

## Observability

- Prometheus: scrape `http://host:8080/metrics` when enabled in config.
- Audit: SQLite at `governance.audit_db_path`.
- Snapshots: JSON files under `governance.snapshot_dir` for rollback via API.

## Rollout checklist

1. **Shadow** with `--force-shadow` on a replay export of customer traffic.
2. Compare dual-metric report (offline) to sidecar audit logs.
3. Switch to `recommend` or `bounded_auto` with safety budget enabled.
4. Calibrate `kpi` weights and alert thresholds with customer ops.

See also: [sidecar_demo.md](sidecar_demo.md) for minimal curl examples.
