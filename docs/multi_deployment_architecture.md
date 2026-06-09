# Multi-Deployment Architecture

*Design sketch — no implementation yet. Last updated: 2026-06-05*

---

## Problem

A large organization runs multiple ML models (fraud scoring, risk rating, claims triage).
Each model has its own deployment, its own label delay, and its own shift profile.
Currently each would run a separate ARL sidecar with no shared state — regime knowledge
learned by one model is invisible to others, and there is no unified governance view.

---

## Proposed architecture

```
                         ┌────────────────────────────────────────┐
                         │          ARL Control Plane              │
                         │                                         │
                         │  ┌───────────────┐  ┌───────────────┐  │
                         │  │  Shared Regime │  │  Central Audit │  │
                         │  │  Registry     │  │  Store (Kafka/ │  │
                         │  │  (Postgres)   │  │  S3 JSONL)     │  │
                         │  └──────┬────────┘  └───────┬────────┘  │
                         │         │                   │            │
                         └─────────┼───────────────────┼────────────┘
                                   │                   │
              ┌────────────────────┼──────────┐        │
              │                    │          │        │
     ┌────────▼──────┐    ┌────────▼──────┐  │  ┌─────▼──────────┐
     │  ARL Sidecar   │    │  ARL Sidecar   │  │  │  Governance    │
     │  (fraud-v1)    │    │  (claims-v2)   │  │  │  Dashboard     │
     │                │    │                │  │  │  (Grafana)     │
     │ /v1/batch      │    │ /v1/batch      │  │  └────────────────┘
     │ /v1/metrics    │    │ /v1/metrics    │  │
     └───────┬────────┘    └───────┬────────┘  │
             │                    │             │
     ┌───────▼────────┐  ┌────────▼───────┐    │
     │ Fraud ML Model  │  │ Claims ML Model │    │
     └────────────────┘  └────────────────┘
```

---

## Components

### Shared Regime Registry
- Stores regime encoder prototypes (centroids, rewards, novelty scores)
- Keyed by `(model_family, feature_schema_hash)` — allows models trained on the same
  feature space to share regime memory
- Write: each sidecar flushes its `StreamingRegimeEncoder` prototypes on policy state save
- Read: new sidecar deployments warm-start from the registry instead of cold-starting

**API (proposed):**
```
POST /regimes/{model_family}   — upsert prototype batch
GET  /regimes/{model_family}   — fetch prototypes for warm-start
```

### Centralized Audit Store
- All sidecars write `AuditRecord` events to a shared Kafka topic or S3 JSONL sink
- Enables cross-model governance: "did any model take a reset action in the last hour?"
- Current per-sidecar SQLite is sufficient for single-model use; Kafka is the multi-tenant path

### Governance Dashboard
- Grafana dashboard reading from Kafka/S3
- Per-model panels: shift_score, risk_capital, action_taken distribution, retrain_recommended rate
- Cross-model panel: correlated shift events (multiple models drifting simultaneously → macro shift)
- Alert rules: `risk_capital > 8.0` for > 10 batches → PagerDuty

### Cross-model regime knowledge sharing
- Blocked on Shared Regime Registry implementation
- Hypothesis: a fraud model that learned "holiday spike = operating condition switch" can share
  that regime prototype with a credit model on the same feature space

---

## Deployment variants

### Variant A: Sidecar per model (current)
- One ARL process per model deployment
- Simplest; no shared state
- Suitable for < 10 models

### Variant B: Shared sidecar pool
- Multiple models share a sidecar pool; routed by model_id in batch payload
- Policy state is isolated per model_id but regime encoder is shared
- Suitable for 10–100 models with the same feature schema

### Variant C: Centralized ARL service
- Single ARL cluster with multi-tenant API
- `POST /v1/models/{model_id}/batch`
- Full shared regime registry and centralized audit
- Required for > 100 models or enterprise governance requirements

---

## Open questions before implementation

1. **Feature schema versioning** — how to handle models with different feature sets sharing regime knowledge?
2. **Prototype privacy** — regime centroids computed from customer data; are they safe to share cross-tenant?
3. **Conflict resolution** — two sidecars writing to the same regime key simultaneously
4. **Latency budget** — shared registry adds a network hop per batch; must be < 1ms

---

*Prerequisites: Centralized Audit API (Blocked), architecture review sign-off*
