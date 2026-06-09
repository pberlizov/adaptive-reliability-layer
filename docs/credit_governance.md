# Regulatory Governance Narrative

*How ARL supports SR 11-7 and TRIM model risk management requirements.*  
*Last updated: 2026-06-05*

---

## Scope

This document maps ARL's technical capabilities to the model risk management (MRM)
requirements of **SR 11-7** (Federal Reserve, 2011) and the ECB's **TRIM** (Targeted
Review of Internal Models). Both frameworks share the same core principle: models used
in consequential decisions must have governed, auditable, and reversible change processes.

ARL addresses the gap between these requirements and the practical reality of online
ML models that silently drift — and currently have no change governance at the
inference layer.

---

## SR 11-7 / TRIM alignment

### 1. Model change governance

**Requirement:** All model changes must be documented, justified, and approved before
deployment. Changes to live models must follow a change-management process.

**ARL capability:**
- `operating_mode=recommend` queues every proposed model adaptation for human review
  before applying. No model change happens without an explicit `POST /v1/approve` call.
- `AuditRecord` captures every proposed and applied change with timestamp, recommended
  action, approved_by, shift_score, risk_capital, and full metadata.
- The audit trail is immutable (SQLite append-only) and exportable to JSONL for
  regulatory examination.

### 2. Model validation and ongoing monitoring

**Requirement:** Banks must validate models before deployment and monitor performance
ongoing. Validators should have access to model behavior under various stress scenarios.

**ARL capability:**
- `shift_score`, `risk_capital`, `parameter_drift`, and `retrain_recommended` are
  emitted on every batch and available via `GET /v1/metrics`.
- Monitor precision/recall benchmarks quantify false-alarm rate and detection latency.
- `docs/risk_metric_spec.md` pre-commits the exact formulas used so validators can
  reproduce the monitoring independently.

### 3. Model inventory and versioning

**Requirement:** Banks must maintain a model inventory. Each model in production must
be versioned, and its behavior at any point in time must be reconstructable.

**ARL capability:**
- `model_version` string in every `AuditRecord` identifies the source model.
- `SnapshotStore` preserves pre- and post-intervention model states with `snapshot_id`
  references in the audit log. A validator can reconstruct the exact model state at any
  past audit step.
- Rollback to any prior snapshot via `POST /v1/rollback/{snapshot_id}`.

### 4. Risk-tiered oversight

**Requirement:** Higher-risk models (credit decisions, capital calculations) require
more conservative change processes and stronger human oversight.

**ARL capability:**
- Three operating modes provide a risk-tiered control surface:
  - `shadow`: zero model mutation — suitable for high-risk models during validation
  - `recommend`: human-in-the-loop approval — suitable for live production under SR 11-7
  - `bounded_auto`: constrained automation with safety budget — suitable for lower-risk
    domains or ops-approved automation

### 5. Stress testing and limit breach

**Requirement:** Models must be stress-tested against adverse scenarios and must have
defined escalation paths when limits are breached.

**ARL capability:**
- Martingale risk capital (`risk_capital`) is a sequential statistical alarm that fires
  when the cumulative evidence of shift exceeds a threshold. It serves as the limit breach signal.
- `retrain_recommended=True` is the formal escalation signal.
- False alarm flood stress test (`test_stress.py::TestFalseAlarmFlood`) demonstrates the
  governor prevents limit-breach flooding from triggering unbounded auto-adaptations.
- The incident response playbook (`docs/incident_response.md`) defines P0–P3 severity
  levels and escalation contacts.

---

## Gaps and limitations

| Requirement | Status |
|---|---|
| Independent model validation | **Gap** — ARL provides observability but not independent validation. A separate validator must review audit records. |
| Pre-approval for automated actions | `bounded_auto` with safety budget provides *constrained* automation without pre-approval. Full SR 11-7 compliance requires `recommend` mode. |
| Model retirement governance | **Gap** — ARL signals `retrain_recommended` but does not govern model retirement. |
| Multi-model inventory | **Gap** — Centralized audit API is not yet implemented (see multi-deployment sketch). |
| Encryption at rest | Policy state encryption (AES-256-GCM) is not yet implemented for Redis backend. |

---

## Recommended deployment configuration for regulated use

```yaml
operating_mode: recommend          # require human approval for all adaptations
governance:
  environment: production
  persist_snapshots_on_mutation: true
  persist_snapshots_on_recommend: true
  max_snapshots: 500               # keep ≥ 1 year of snapshots at daily frequency
safety_budget:
  enabled: true
  max_auto_actions_per_window: 0   # zero in recommend mode — all blocked until approved
  downgrade_to_recommend: true
policy:
  name: delayed_bandit             # generate recommendations; never auto-apply
sota:
  asr_reset_enabled: false         # no auto-resets in regulated use
  deferred_adaptation_enabled: false
```

---

*For questions on regulatory compliance, contact your data protection officer.*
