# Product milestones — prove impact (first five)

This doc turns the **first impact milestones** into concrete work: what each one proves, why a buyer should care, how to run it in this repo, and how to know you passed.

**North star:** High value is shown when ARL **reduces operational risk and unnecessary interventions** on a **delayed-label, production-shaped stream**—with governance and restart safety—not when it wins raw accuracy on a static test set.

**Repo root:** `~/Documents/GitHub/adaptive-reliability-layer` (adjust paths if yours differs).

**Setup:**

```bash
cd ~/Documents/GitHub/adaptive-reliability-layer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,prometheus,serving]"
```

---

## Milestone 1 — Dual-metric “ops story” on one fraud stream

### What this proves

The product is a **controller over bounded interventions**, not a replacement trainer. On the **same chronological stream** you run:

- **Shadow** — monitor shift/risk and recommend actions, but **do not mutate** the model.
- **Bounded auto** — apply only low-risk actions from an allowlist, with safety budget.

A design partner should see: “we can start in shadow, then turn on automation without guessing the blast radius.”

### Why it matters (buyer language)

- **Risk / alert rate** — fewer noisy escalations vs a frozen baseline that fires on every drift blip.
- **Intervention budget** — mutations per 1k steps stay under a cap you can quote in a pilot SOW.
- **Ops deferral** — retrain recommendations or harmful drift episodes avoided vs frozen (see `dual_metric_report.md` and buyer KPI block).

Accuracy may stay flat; **utility and risk capital** are the headline metrics.

### How to run

**Recommended path (PaySim torch + regime-aware delayed bandit, dual report built in):**

```bash
# Bundled PaySim CSV if needed:
python3 scripts/export_bundled_fraud_data.py

python3 scripts/run_pilot_torch.py
```

**Artifacts:** `results/pilot_torch/dual_metric_report.md`, `dual_metric_report.json`, `pilot_report.md`.

**Longer public fraud suite (multiple sources, good for appendix):**

```bash
python3 scripts/run_fraud_public_benchmark.py --stream-cycles 6
# Artifacts: results/fraud_public_benchmark/
```

**Sklearn / tabular pilot (synthetic or CSV):**

```bash
arl-pilot --config configs/pilot_fraud_sklearn.yaml --output-dir results/pilot_sklearn
# or
python3 scripts/run_pilot_case_study.py --config configs/pilot_fraud_tabular.yaml
```

**Ingest file → dual replay:**

```bash
python3 scripts/run_ingest_replay.py \
  --input data/openml/credit_german.csv \
  --config configs/default.yaml \
  --dual-mode \
  --output-dir results/ingest_dual
```

### Pass criteria (checklist)

Tune thresholds with your pilot customer; defaults below are **starting targets** for internal sign-off.

- [ ] **Same stream, two modes:** `dual_metric_report.md` contains both `shadow` and `bounded_auto` sections.
- [ ] **Risk story:** In `bounded_auto`, `risk_reduction` vs frozen is **≥ 0%** (positive = lower mean risk capital than frozen on that replay). Prefer **≥ 10%** before external claims.
- [ ] **Intervention cap:** `bounded_interventions_per_1000` for the controller strategy is **≤ 50** (or a cap you document in the pilot config). If over cap, safety budget should show `budget_limited` events.
- [ ] **Buyer headline exists:** At least one sentence in the report from `compute_buyer_kpis` (harmful alert reduction, retrain deferral, or utility delta)—not only raw tables.
- [ ] **Stream length:** `max_steps × batch_size` (or recorded row count) is **≥ 2,000 labeled events** for external-facing pilots; use `stream_cycles` on fraud benchmark or raise `replay.max_steps` in `configs/pilot_fraud_torch.yaml`.
- [ ] **Label delay documented:** Config shows `replay.label_delay_steps` (e.g. `4` in torch pilot) and the report states delayed supervision.

### Record for outreach

| Field | Where to copy from |
|--------|-------------------|
| Shadow vs bounded risk | `dual_metric_report.json` → `modes.bounded_auto.risk_reduction` |
| Utility delta vs frozen | `modes.bounded_auto.utility_delta` |
| Harmful alert reduction | Buyer block / `business_kpis.harmful_alert_reduction_pct` |
| Interventions / 1k steps | Strategy summary in technical table |

---

## Milestone 2 — Delayed-label fidelity (commercial path)

### What this proves

Fraud and ops workflows get labels **late**. ARL must:

1. Score batches **without** labels at decision time (`process_batch` with `labels=None` when delay &gt; 0).
2. Apply learning when labels arrive (`reveal_labels(step, labels)`).
3. Improve or match **frozen** while beating naive “pretend labels are instant” baselines.

This separates ARL from benchmarks that cheat with immediate supervision.

### Why it matters

If delayed feedback does not move the controller state, the product is only a monitor. If it does, you can credibly sell **learning under label latency**—core to chargeback review, case disposition, and retrain triggers.

### How to run

**Torch pilot (delay + policy persistence path configured):**

```bash
python3 scripts/run_pilot_torch.py
# Policy state written to results/pilot_torch/policy_state.json when policy_state_save_path is set
```

**Replay API smoke (unit-level behavior):**

```bash
pytest tests/test_tier12_product.py::test_reveal_labels_updates_delayed_bandit_policy -q
```

**Manual check — policy state changed after reveals:**

```bash
# Before/after: matrices in policy_state.json should differ after a full pilot run
cat results/pilot_torch/policy_state.json | head -20
```

**Extend delay / steps in config** (`configs/pilot_fraud_torch.yaml`):

```yaml
replay:
  label_delay_steps: 8   # stress longer lag
  max_steps: 96
```

Re-run `run_pilot_torch.py` and compare `regime_aware_delayed_bandit` vs `frozen` in `pilot_report.md`.

### Pass criteria

- [ ] **`reveal_labels` works:** No `KeyError: no pending batch` during replay when `label_delay_steps > 0` (pilot completes cleanly).
- [ ] **Controller vs frozen utility:** `controller_vs_frozen_utility_delta` **≥ 0** on the chosen fraud stream (same or better utility than frozen).
- [ ] **Risk not worse:** `controller_vs_frozen_risk_reduction` is **not materially negative** (e.g. not worse than −5% vs frozen unless interventions explain it in the report).
- [ ] **Policy state is real:** `results/pilot_torch/policy_state.json` exists and `kind` is `regime_aware_delayed_bandit` or `bandit` with non-empty `matrices` / `vectors`.
- [ ] **Restart sanity (stretch):** Run replay in two segments with `policy_state_path` pointing at saved JSON; second segment intervention mix within reason vs one continuous run (document ε in pilot notes).

### Record for outreach

> “Under N-step label delay, the controller matched or beat frozen on utility while keeping sequential risk at or below baseline; bandit state persisted across the stream.”

---

## Milestone 3 — Sidecar integration smoke (HTTP)

### What this proves

ARL deploys as a **sidecar**: your model stays upstream; ARL receives batch features (and optional regime), returns a **decision record**, and accepts **delayed labels** on a separate call. No requirement to retrain inside ARL for the demo.

### Why it matters

Buyers need to see **integration surface area** (days to wire, not months to rewrite). HTTP + audit/snapshots is the commercial shape; offline replay alone is not.

### How to run

```bash
pip install -e ".[serving]"
python3 scripts/run_serve.py --config configs/default.yaml --port 8080
```

In another terminal:

```bash
# Health
curl -s http://127.0.0.1:8080/v1/health | python3 -m json.tool

# One batch (feature_dim must match the default layer’s adapter — default tabular build uses 30 features)
curl -s -X POST http://127.0.0.1:8080/v1/batch \
  -H 'Content-Type: application/json' \
  -d '{"features": [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "regime": "live"}' \
  | python3 -m json.tool

# Note the "step" in the response, then reveal labels for that step (if label_delay > 0 in config)
curl -s -X POST http://127.0.0.1:8080/v1/batch/0/labels \
  -H 'Content-Type: application/json' \
  -d '{"labels": [0]}' \
  | python3 -m json.tool

curl -s http://127.0.0.1:8080/v1/metrics | python3 -m json.tool
```

**Automated smoke:**

```bash
pytest tests/test_tier12_product.py::test_fastapi_health_endpoint -q
```

For **recommend** mode approval path, set `operating_mode: recommend` in config and use `POST /v1/approve` with `approved_action` and `approver`.

### Pass criteria

- [ ] `GET /v1/health` returns `status: ok` and a `model_version`.
- [ ] `POST /v1/batch` returns a full **decision record** (`shift_score`, `recommended_action`, `action_taken`, `risk_capital`, etc.).
- [ ] With `replay.label_delay_steps > 0` in config, `POST /v1/batch/{step}/labels` returns metrics (e.g. `batch_accuracy`) and `/v1/metrics` shows `revealed_batches` non-empty.
- [ ] Audit DB path from config receives rows after batches (check `governance.audit_db_path` in YAML).
- [ ] **Demo script exists:** You can hand a partner a 5-step curl sequence (or `docs/sidecar_demo.md` once written) without starting from source code.

### Record for outreach

> “Sidecar HTTP API: batch scoring, delayed label reveal, optional human approve path; same semantics as offline replay.”

---

## Milestone 4 — Real-data verification scorecard

### What this proves

The commercial runtime is not a single sklearn toy path. The **same 8 engineering priorities** (deployment surface, operating modes, replay, adapters, maturity, observability, governance, KPI evidence) are exercised across **multiple public sources**, with one **fraud wedge** highlighted.

### Why it matters

Design partners and security reviewers ask: “Does it only work on your synthetic stream?” This milestone is the **release gate** before you claim “production-oriented.”

### How to run

```bash
python3 scripts/export_bundled_real_data.py   # if OpenML/bundled CSVs missing
python3 scripts/export_bundled_fraud_data.py  # PaySim / IEEE-CIS samples

arl-real-data-verification \
  --config configs/real_data_verification.yaml \
  --output-dir results/real_data_verification

# Fraud-focused subset:
arl-real-data-verification \
  --sources paysim_fraud,ieee_cis_fraud,openml_credit_g \
  --output-dir results/real_data_verification_fraud
```

Add fraud sources to `configs/real_data_verification.yaml` under `sources:` if not already listed:

```yaml
sources:
  - paysim_fraud
  - ieee_cis_fraud
  - openml_credit_g
```

**Artifacts:** `results/real_data_verification/verification_suite.md`, per-source JSON, priority check results.

### Pass criteria

- [ ] Suite completes without `--fail-fast` abort.
- [ ] **All 8 priorities** pass for: `breast_cancer`, `digits`, and **one fraud source** (`paysim_fraud` or `ieee_cis_fraud`).
- [ ] **Fraud wedge called out** in the summary markdown (which source is the pilot anchor and why).
- [ ] Offline replay uses **label delay** where configured (`label_delay_steps: 2` in verification config).
- [ ] Governance paths write **audit** and optional **snapshots** without error.
- [ ] One-line scorecard at top of `verification_suite.md`: `PASS x/y sources` (you can add this manually until automated).

### Record for outreach

Table: source × pass/fail × primary KPI (risk reduction or utility delta). Link to `verification_suite.md` in data room.

---

## Milestone 5 — Buyer KPI calibration (one pilot, their weights)

### What this proves

Technical metrics translate to **language the buyer already uses**. The `kpi` block in runtime YAML weights accuracy, false alerts, abstention, reset cost, and retrain recommendations into a **business score**; dual-metric reports surface that alongside buyer KPI headlines.

### Why it matters

Without this step, you sell “shift score” and “risk capital.” With it, you sell **fewer false escalations**, **less retrain churn**, or **lower review load**—mapped from the same replay artifacts.

### How to run

1. Pick **one wedge KPI** with the customer (see mapping below).
2. Edit weights in `configs/pilot_fraud_torch.yaml` (or a customer-specific copy):

```yaml
kpi:
  accuracy_weight: 1.0
  false_alert_cost: 0.08      # raise if they care about alert fatigue
  drift_cost: 0.03
  abstention_cost: 0.10       # raise if analyst queue cost is high
  reset_cost: 0.05
  retrain_recommendation_cost: 0.10  # raise if retrain downtime is expensive
```

3. Re-run the dual-metric pilot:

```bash
python3 scripts/run_pilot_torch.py
```

4. Review `dual_metric_report.md` **business_kpis** lines and the buyer headline from `buyer_kpis.py`.

**Reference outreach framing:** `docs/outreach_one_pager.md`.

### KPI mapping (pick one primary per pilot)

| Buyer cares about | Raise / tune in `kpi` | Evidence in report |
|-------------------|------------------------|-------------------|
| Chargeback / fraud loss | `accuracy_weight`, intervention allowlist | Utility delta, batch accuracy after reveal |
| Alert fatigue / false escalations | `false_alert_cost` | `harmful_alert_reduction_pct` vs frozen shadow |
| Analyst queue / abstention | `abstention_cost` | Abstention rate in surfaces |
| Model ops / retrain churn | `retrain_recommendation_cost` | `controller_retrain_deferral_steps_vs_frozen` |
| Incident severity | `drift_cost`, risk alert rate | Mean risk capital, risk reduction % |

### Pass criteria

- [ ] Customer **primary KPI** named in writing (one sentence in pilot charter or email).
- [ ] Config weights changed **deliberately** from defaults (documented in commit message or pilot README).
- [ ] `dual_metric_report.md` includes **business score** and at least one metric that moves in the **direction the buyer cares about** when comparing shadow → bounded_auto.
- [ ] Buyer agrees the headline is **directionally right** (“not wrong”), even if weights are still provisional.
- [ ] Caveat documented: weights are **engineering-tuned** until production telemetry calibrates them (`docs/outreach_one_pager.md`).

### Record for outreach

Single slide: **Primary KPI** → **ARL lever** → **number from last replay** → **caveat (offline, replay weights)**.

---

## Suggested order (weeks 1–4)

| Week | Focus | Milestones |
|------|--------|------------|
| 1 | Fraud dual-metric + verification | **M1**, **M4** |
| 2 | Delay + policy state | **M2** |
| 3 | HTTP sidecar demo | **M3** |
| 4 | Customer weights + narrative | **M5** (depends on M1 artifacts) |

M1 and M4 can run in parallel; M5 should use the latest M1 run.

---

## Quick status template

Copy into PR or weekly update:

```markdown
## Product milestone status

- [ ] M1 Dual-metric fraud ops story — artifact: results/pilot_torch/dual_metric_report.md
- [ ] M2 Delayed-label fidelity — policy_state.json + utility/risk deltas
- [ ] M3 HTTP sidecar smoke — curl demo / test pass
- [ ] M4 Real-data verification — verification_suite.md (x/y pass)
- [ ] M5 Buyer KPI calibration — primary KPI: __________ ; weights file: __________
```

---

## What comes after these five

Research and moat milestones (temporal grid stability, Kafka ingest, baseline comparison pack, recurrence-gated specialists) are intentionally **out of scope** for this doc. See `docs/next_step_decision_memo.md` and `docs/status_paper_commercial_outreach.md` when you are ready to narrow the science track again.

---

## Run all milestones (automation)

```bash
python3 scripts/run_product_milestones.py
```

Writes `results/pilot_sklearn/milestone_status.json`, `results/pilot_torch/milestone_status.json`, and `results/product_milestones/run_summary.json`.

## Related commands (cheat sheet)

| Goal | Command |
|------|---------|
| All milestones | `python3 scripts/run_product_milestones.py` |
| Sklearn fraud dual pilot (fast) | `python3 scripts/run_pilot_sklearn.py` |
| Torch fraud dual pilot | `python3 scripts/run_pilot_torch.py` |
| Public fraud benchmark | `python3 scripts/run_fraud_public_benchmark.py` |
| Verification suite | `arl-real-data-verification` |
| Ingest + dual replay | `python3 scripts/run_ingest_replay.py --input … --dual-mode` |
| HTTP serve | `python3 scripts/run_serve.py` |
| Bounded auto demo | `python3 scripts/run_bounded_auto_demo.py` |
| Unit tests (product) | `pytest tests/test_tier12_product.py tests/test_commercial_runtime.py -m 'not slow'` |
