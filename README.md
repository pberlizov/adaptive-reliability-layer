# Adaptive Reliability Layer

**Stop retraining your fraud model every time a drift alarm fires.**

ARL sits between your inference pipeline and your monitoring layer. It detects distribution shift, decides whether adaptation is actually warranted, learns from delayed revealed labels (chargebacks, disputes — weeks after inference), and takes the smallest bounded steering step that stabilizes the model: correction first, explicit mutation only when needed. Most drift alarms don't require a retrain. ARL tells you which ones do, proves it, and logs every decision. It does not replace your model.

```bash
pip install "adaptive-reliability-layer[torch,serving]"
arl-demo   # ~2 min, no downloads, runs on your machine
```

---

## Why this exists

Public fraud streams (ULB, IEEE-CIS, PaySim) show **94–99% frozen accuracy** — accuracy looks fine. But the model is quietly degrading on shifted segments, and the standard fix (scheduled retrain) fires too late and adapts too broadly.

For AML and fraud teams, a retrain isn't a free operation. It means: retraining on millions of labeled transactions, engineering the dataset, running backtests, getting model risk sign-off (SR 11-7 / TRIM in regulated institutions), staging a deployment, and holding a rollback window. Teams that retrain on a fixed schedule — or every time a drift alarm fires — are paying that cost repeatedly for shifts that either resolve on their own or don't affect decision quality. ARL's primary job is to distinguish "this shift requires intervention" from "this shift can be safely held," and to take the smallest bounded action that stabilizes the model — deferring unnecessary retrains and their associated compliance overhead.

The core problem is harder than it looks: **labels arrive weeks after inference** (fraud chargebacks, engine failures, clinical outcomes). You can't do standard online learning. You need a controller that learns from delayed feedback, knows when to hold, and can prove it didn't harm anything.

---

## Numbers

### Fraud (3 public temporal streams, torch adapters, 12-step label delay)

| Stream | Risk ↓ vs frozen | Utility Δ vs scheduled retrain | Beats naive adapt |
|--------|------------------|-------------------------------|-------------------|
| ULB credit card | **7.2%** | **+0.54** | ✓ |
| IEEE-CIS | **8.7%** | **+0.51** | ✓ |
| PaySim | **6.0%** | **+0.52** | ✓ |

**Suite: 3/3 PASS** (`require_beat_baselines: true`). All baselines evaluated with the same torch adapter, same temporal split, same 12-step label delay.

**Risk** = composite proxy: the strongest reduction among martingale capital, drift-alert rate, and retrain recommendations versus frozen. Raw fraud accuracy is flat across all methods (94–99%) — these are operational metrics, not detection quality metrics.

**Utility** = replay accuracy minus operational penalties such as sustained risk alerts, parameter drift, abstention, and resets. Retrain deferral is reported separately in the risk / operations story. Full spec: [docs/risk_metric_spec.md](docs/risk_metric_spec.md).

### Predictive maintenance (NASA CMAPSS turbofan degradation, real data)

Frozen model accuracy degrades 8–13 pp as engines approach failure. ARL result on 4 sub-datasets:

| Dataset | Conditions | Fault modes | Best controller | vs frozen | Result |
|---------|-----------|-------------|----------------|-----------|--------|
| FD001 | 1 | 1 | `delayed_bandit` | **+2.3 pp** | PASS |
| FD002 | 6 | 1 | `delayed_hybrid` | **+2.1 pp** | PASS |
| FD003 | 1 | 2 | `delayed_bandit` | **+1.6 pp** | PASS |
| FD004 | 6 | 2 | `delayed_bandit` | **+0.0 pp** | HOLD ✓ |

**3/4 PASS. FD004 is a correct hold, not a failure.** On FD004 (2 fault modes × 6 conditions, 495 test batches), all unsupervised adaptation strategies hurt: naive −12.7 pp, rule-based −3.0 pp, unsupervised bandit −2.6 pp. The production controller — learning from delayed labels — correctly identified that no available action improves this dataset and held at frozen accuracy. The governance layer is doing its job.

### vs in-repo policy baselines (CMAPSS FD001)

| Method | Final accuracy Δ vs frozen |
|--------|---------------------------|
| **ARL `delayed_bandit`** | **+2.3 pp** |
| BN-only adaptation | −1.2 pp |
| Naive (always adapt) | −9.1 pp |
| Frozen (no adaptation) | 0.0 pp (baseline) |

All results are in-repo benchmarks on CMAPSS FD001 under identical conditions (temporal split, 12-step label delay). BN-only adaptation = `TentTabularPolicy` (entropy minimization + BN refresh). Naive = `NaiveTabularPolicy` (adapt whenever shift score exceeds threshold). Reproduce with `arl-hn-launch`.

---

## Install

```bash
pip install "adaptive-reliability-layer[torch,serving]"
```

Requires Python 3.10+. PyPI **0.3.4** is the launch-sync release for `arl-demo`, `arl-hn-launch`, and `arl-serve`.

---

## License

ARL is **source-available** under **BUSL-1.1**, not open source. In plain
English: you can inspect the code, run the demo, benchmark it, evaluate it
internally, and review it for research or security work. **Production use,
managed-service use, and customer-facing deployment require a commercial
license.**

If there is any conflict between this summary and the license text, the
[LICENSE](LICENSE) file controls. See [docs/licensing.md](docs/licensing.md)
for the repo-specific usage guide.

---

## Try it

**Quick demo — 2–5 min, no downloads, synthetic data only:**

```bash
arl-demo
# same as: arl-hn-launch --quick
```

Runs PaySim synthetic stream → production benchmark → hard-slice benchmark → writes `results/hn_launch/comparison_table_quick.md`. **This is a 1-source toy run.** The three-source numbers in this README require the full suite below.

**Full five-dataset suite — 30–90 min:**

```bash
arl-hn-launch
```

Runs ULB, IEEE-CIS, PaySim, Elliptic (graph), BAF on torch adapters with temporal splits and delayed labels. Artifacts land in `results/hn_launch/`.

**Export-only — ~1 min, no training:**

```bash
arl-hn-launch --export-only
```

**HTTP sidecar API:**

```bash
arl-serve --config serving_pilot_fraud_torch.yaml --force-shadow
curl -s http://127.0.0.1:8080/v1/health
curl -s http://127.0.0.1:8080/v1/batch -d '{"features": [[...]]}'
```

Full curl flow: [docs/sidecar_demo.md](docs/sidecar_demo.md)

---

## How it works

```
inference pipeline
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │            ReliabilityLayer                  │
 │                                              │
 │  shift monitor → risk capital → governor     │
 │         │               │           │        │
 │  feature score    martingale    action gate  │
 │  output score     sequential    (hold/adapt) │
 │  collapse risk    test                       │
 │                                              │
 │  delayed bandit ← label reveals (weeks later)│
 │  specialist pool  regime encoder             │
 └─────────────────────────────────────────────┘
        │
        ▼
  predictions + audit trail + rollback metadata
```

**Shift detection.** Three signals: feature distribution shift (normalized Mahalanobis), output distribution shift (KL from source), collapse risk (martingale capital sequential test). Each triggers different actions.

**Steering library.** ARL has two layers of control: narrow probability / threshold correction on most batches, and explicit actions when warranted. Explicit actions include `none`, `bn_refresh`, `label_shift`, `bbse_label_shift`, `recalibrate`, `cool_confidence`, `adapt`, `reset`. The controller selects from them; the governor gates the selection.

**Benign-shift gate.** When revealed accuracy > 0.92 AND revealed positive rate is stable, the controller holds regardless of detected shift — the shift is benign and adaptation would hurt. This is the mechanism behind FD004's correct hold: without it, all unsupervised strategies harm performance on that dataset.

**Delayed bandit.** LinUCB (28D context: shift signals + temporal state + regime features) learns from delayed revealed labels. Reward = `utility + 0.15 × (revealed_accuracy − baseline_accuracy)` — a counterfactual lift signal that lets the controller distinguish "this steering step helped" from "things were already good."

**Specialist reservoir.** Up to 4 model snapshots, each with per-regime behavior signatures (40% feature + 60% confidence/entropy). Routing uses blended distance; a staleness gate skips snapshots whose creation positive rate diverges from current by more than 0.15. Regime encoder tracks per-prototype revealed positive rate EMA so the bandit can distinguish regimes by label distribution, not just feature fingerprint.

**Governance.** Every decision is logged to SQLite with action, reason, regime_id, risk_score, rollback_eligible. Rollback restores a prior snapshot deterministically. Operating modes: `shadow` (observe only), `recommend` (human approval), `bounded_auto` (autonomous within budget caps).

---

## Honest limits

- **This is an ops/reliability layer, not a fraud detector.** ARL wraps your existing model. Raw detection accuracy on public streams is already 94–99% frozen — there is essentially no headroom to improve, and that's not what ARL is measuring. It measures operational reliability: risk capital, retrain deferral, utility under governance costs.
- **Elliptic (Bitcoin blockchain) is an extended tier, not a core claim.** The Elliptic dataset has fundamentally different temporal structure — illicit clusters are time-localized by exposure windows in the blockchain, not by the gradual covariate drift ARL is designed for. It also loses to naive on utility on the extended stream. It is not included in the 3/3 core claim for this reason, and the controller correctly holds on the hard tail rather than harm.
- **FD004 is a correct hold.** The 2-fault-mode × 6-condition structure needs fault-mode-specific interventions the current action library doesn't have. All unsupervised strategies hurt on FD004; the production controller (learning from delayed labels) held at frozen accuracy. The governance layer is doing its job.
- **Label delay ≤ 30 days.** Beyond that, the bandit reward signal is too stale to be useful.
- **Binary / low-cardinality classification.** Specialist routing doesn't generalize to 100-class problems without modification.
- **Single-run CMAPSS variance ≈ 1–5 pp.** Temporal folds with statistical significance tests are the reliable estimate; single-run numbers are directionally correct but noisy.

---

## SDK

Three lines to wrap your model:

```python
from adaptive_reliability_layer import build_session_from_sklearn

session = build_session_from_sklearn(clf, X_reference, y_reference)

# Each batch:
result = session.predict(X_batch)           # get predictions + shift score
session.reveal(step, y_delayed_labels)      # tell it what actually happened
```

Also: `build_session_from_torch`, `build_session_from_predict_fn`. Full quickstarts: `notebooks/`.

---

## Evidence and docs

| Document | Contents |
|----------|----------|
| [docs/current_findings.md](docs/current_findings.md) | Full benchmark evidence with methodology notes |
| [docs/positioning.md](docs/positioning.md) | Head-to-head comparison table vs TENT, ADWIN, Evidently, River |
| [docs/risk_metric_spec.md](docs/risk_metric_spec.md) | Formal definition of every metric |
| [docs/sidecar_user_guide.md](docs/sidecar_user_guide.md) | API contract for HTTP sidecar |
| [docs/credit_governance.md](docs/credit_governance.md) | SR 11-7 / TRIM mapping for regulated deployments |
| [docs/security_threat_model.md](docs/security_threat_model.md) | Threat model and deployment checklist |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/   # 152 tests, ~2 min
```

CI runs the 152-test suite on Python 3.11, and local verification also passed on Python 3.14.

---

## Research benchmarks

The repo includes additional benchmarks beyond the fraud/maintenance core: temporal image shift (Fashion-MNIST), graph-native drift (Elliptic Bitcoin), WILDS CivilComments (NLP/moderation), UCI gas sensor drift, and OpenML electricity. Install with `pip install -e ".[research,dev]"`.

---

## Citation

If you use ARL in research, a citation to this repo is appreciated. Academic write-up in progress (ICML DistShift workshop path, target Aug 2026).
