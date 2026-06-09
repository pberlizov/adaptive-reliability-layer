# ARL Incident Response Playbook

*Version 1.0 — 2026-06-05*

---

## Severity levels

| Level | Description | Response time |
|---|---|---|
| P0 | Model producing wrong outputs at scale; safety-critical domain | Immediate (< 15 min) |
| P1 | Drift alarm sustained > 10 batches; retrain_recommended firing continuously | < 1 hour |
| P2 | Unexpected action taken; audit anomaly; auth failure spike | < 4 hours |
| P3 | Performance regression in non-critical domain; config drift | Next business day |

---

## Scenario 1: Auth breach / credential leak

**Symptoms:** `auth_failures` counter spikes on `/v1/metrics`; requests from unexpected IPs.

**Steps:**
1. Rotate `api_key` and `admin_api_key` immediately (redeploy sidecar).
2. Export audit log: `POST /v1/audit/export {"filename": "breach_export.jsonl"}`.
3. Review audit records for unauthorized `action_taken` values (adapt, reset, rollback).
4. If any unauthorized adaptation occurred: roll back to last known-good snapshot via `POST /v1/rollback/{snapshot_id}`.
5. Set `operating_mode` to shadow: `POST /v1/operating-mode {"mode": "shadow"}`.
6. File incident report; preserve audit export as evidence.

---

## Scenario 2: Model producing unexpected outputs (silent corruption)

**Symptoms:** Downstream accuracy drops; `parameter_drift` on `/v1/metrics` is high; predictions are systematically wrong.

**Steps:**
1. Check `action_taken` history in `/v1/audit/recent` — look for `adapt` or `reset` actions that shouldn't have fired.
2. Identify the last "good" snapshot: look for `snapshot_id_before` on the step before the drift started.
3. Roll back: `POST /v1/rollback/{snapshot_id} {"actor": "operator"}`.
4. Set operating mode to `recommend`: `POST /v1/operating-mode {"mode": "recommend"}`.
5. Monitor for 10+ batches to confirm rollback restored correct behavior.
6. Review governance config: tighten `safety_budget.max_auto_actions_per_window` if needed.
7. Root cause: check `governor.decision_log` from `/v1/metrics` — were actions allowed that should have been blocked?

---

## Scenario 3: Sustained drift alarm / retrain_recommended flooding

**Symptoms:** `risk_alert=true` for > 10 consecutive batches; `retrain_recommended=true`; `shift_score > severe_threshold`.

**Steps:**
1. Check if drift is genuine: compare `feature_shift_score` vs `output_shift_score`.
   - Feature-dominant: covariate shift (new operating conditions, seasonality). Consider waiting.
   - Output-dominant: concept drift (label distribution changed). Retrain is likely needed.
2. If genuine concept drift:
   - Export audit for provenance: `POST /v1/audit/export`.
   - Trigger offline retrain with recent labeled data.
   - Deploy new model, reset ARL session.
3. If benign condition switch (high accuracy, stable positive rate):
   - Check combined gate in `_resolve_bounded_actions` — it should have blocked adaptation.
   - If gate is firing incorrectly, review `MonitorConfig.alert_threshold` and `severe_threshold`.
4. Temporary mitigation while investigating: switch to shadow mode.

---

## Scenario 4: Data exfiltration / raw features in audit log

**Symptoms:** Audit export JSONL contains raw feature arrays; large numeric lists visible in `metadata_json`.

**Steps:**
1. Stop export: do not distribute the compromised JSONL file.
2. Confirm: run `grep -c "redacted_array" audit_export.jsonl` — if zero, sanitizer is not running.
3. Verify ARL version has `_sanitize_audit_metadata` in `runtime/audit.py` (added v0.3.3+).
4. If old version: upgrade immediately; re-export — new exports will be sanitized.
5. Assess data exposure: which batches were exported? Were feature values PII (e.g. health/financial)?
6. Notify data protection officer if PII was exposed per applicable regulation (GDPR, HIPAA).

---

## Scenario 5: Rate limit bypass / DoS

**Symptoms:** Service unresponsive; `rate_limit_rpm` counter exceeded; 429s not being returned.

**Steps:**
1. Check rate limiter config: `serving_config.rate_limit_rpm` should be set (not None).
2. Verify IP-level rate limiting is enforced upstream (load balancer / API gateway) — ARL's rate limiter is in-process and doesn't block at the network level.
3. Scale horizontally or temporarily restrict access via firewall rule.
4. Check `shift_score` in recent audit records — DoS may be accompanied by synthetic inputs designed to trigger expensive computations.

---

## Quick reference: key endpoints

| Action | Endpoint |
|---|---|
| Check health | `GET /v1/health` |
| See security counters + governor log | `GET /v1/metrics` |
| Recent audit records | `GET /v1/audit/recent?limit=100` |
| Export full audit | `POST /v1/audit/export {"filename": "incident.jsonl"}` |
| Roll back model | `POST /v1/rollback/{snapshot_id} {"actor": "operator"}` |
| Switch to shadow | `POST /v1/operating-mode {"mode": "shadow"}` |
| Switch to recommend | `POST /v1/operating-mode {"mode": "recommend"}` |

---

## Escalation contacts

*(Fill in before deploying to production)*

| Role | Contact |
|---|---|
| On-call ML engineer | `<oncall@your-org.example>` |
| Data protection officer | `<dpo@your-org.example>` |
| CISO | `<security@your-org.example>` |
