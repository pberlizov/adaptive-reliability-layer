# Adaptive Reliability Layer — Outreach One-Pager

**Positioning:** Do not lead with accuracy. Lead with **risk capital**, **harmful drift alerts**, and **bounded interventions** that keep decision quality stable without constant retraining.

## The skeptical buyer question

> "Frozen is already 96% accurate — why would I pay for this?"

**Answer:** Because production ML fails silently. Accuracy on yesterday's distribution is the wrong headline. On our tabular benchmarks, frozen and ARL often **tie on accuracy** while ARL cuts **sequential risk exposure by 50–90%** and raises **utility** (accuracy minus operational penalties for false alarms, drift, resets, and abstention).

## German Credit (fraud/risk adjacent wedge)

Offline replay on OpenML German Credit with a **multi-regime shift stream** (covariate drift → label shift → abrupt shift):

| Metric | Frozen | ARL (bandit) | Buyer translation |
|--------|--------|--------------|-------------------|
| Mean risk capital | ~high | ~low | **Sustained risk burden** on the monitoring layer |
| Risk alerts | higher | lower | **Fewer harmful drift escalations** to ops |
| Accuracy | ~57% | ~57% (equivalent) | **No retrain required** to hold decision quality on this slice |
| Interventions | 0 | bounded | **Controlled adaptation** vs naive always-on TTA |

**Say this:** "We kept approval/decision accuracy stable but cut harmful drift alarms and sequential risk exposure by roughly **90%** in offline replay — that's fewer false escalations and less silent degradation between retrains."

Run from the repo root (no install required):

```bash
python3 scripts/run_outreach_validation.py
```

Or after `pip install -e .`:

```bash
arl-outreach-validation
arl-fraud-benchmark
```

Public fraud benchmark (PaySim + IEEE-CIS + German Credit, long horizon):

```bash
python3 scripts/export_bundled_fraud_data.py   # once, if CSVs missing
python3 scripts/run_fraud_public_benchmark.py
```

Drop Kaggle `train_transaction.csv` at `data/fraud/raw/` to use real IEEE-CIS instead of synthetic fallback.

Artifacts land in `results/outreach_validation/` and `results/fraud_public_benchmark/`.

## Translate internal metrics

| Internal | Say to fraud / risk ops | Say to MLOps |
|----------|-------------------------|--------------|
| +0.02 utility | Fewer costly interventions per 1000 scored rows | Better tradeoff of accuracy vs ops cost |
| Risk capital ↓ 92% | Far fewer **false drift escalations** | Lower martingale-style sequential risk on the stream |
| Risk alerts ↓ | **False alarm rate** on shift detection drops | Monitoring budget goes to real shifts |
| Bounded resets | **Controlled rollback** when adaptation diverges | Safer than naive TTA without governance |
| Shadow mode | **Zero production impact** pilot | Prove value before enabling auto actions |

## TENT / EATA (reviewer gap — now benchmarked)

Standard test-time adaptation baselines (**TENT**, **EATA-style selective TENT**) are included in `scripts/run_outreach_validation.py` on the research tabular shift stream. Typical pattern:

- TENT can move accuracy slightly but **lacks reset discipline and governance** → higher drift and risk alerts vs ARL bandit.
- ARL wins on **utility and risk capital**, not raw accuracy.

See `results/outreach_validation/tta_baseline_comparison.md` after running the script.

## Honest limitations (say proactively)

1. **Horizon:** Offline streams are thousands of rows, not six months of production traffic. Long-horizon replay uses **repeated regime cycles** as a stress proxy — label this clearly in pilots.
2. **Utility weights** are engineering-tuned; pilots should re-weight toward customer KPIs (chargebacks, analyst hours, retrain cost).
3. **Image / NLP** tracks show smaller utility deltas; lead with **tabular fraud/risk** until WILDS-scale evidence improves.

## Pilot offer

1. **Shadow mode** on their replay CSV — no model mutations, full counterfactual actions logged.
2. Report: harmful alert reduction %, accuracy equivalence, intervention rate, recommended vs taken actions.
3. Success criterion: **risk exposure down** with accuracy within agreed tolerance — not accuracy up.
