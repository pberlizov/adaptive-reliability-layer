# Discrimination benchmark

When production fraud benchmarks show **flat accuracy** (~99.9% ULB, ~96% PaySim, ~94% IEEE), method comparison on accuracy alone will falsely look like a tie. This suite uses:

1. **Harder temporal slices** — train on the first half of time, replay only the latest holdout tail.
2. **Imbalance-aware metrics** — balanced accuracy, PR-AUC, recall at precision ≥ 0.80, cost-weighted error.
3. **Natural drift datasets** — gas sensor and electricity streams where frozen models typically degrade across batches.

## Why this exists

| Problem on core fraud suite | Discrimination suite response |
|-----------------------------|-------------------------------|
| Accuracy ceiling | Balanced accuracy + PR-AUC + recall@precision |
| Strong frozen baseline on mild tail | 50/50 temporal split + latest-tail stream only |
| Utility ties hide mechanism differences | Explicit **metric spread** table per source |
| Fraud class imbalance | Cost-weighted error (default FN cost = 10× FP) |
| Proxy risk flat | Late-stream recall delta (1st half vs 2nd half) |

## Data tiers

### Tier D1 — Hard fraud slices (same public CSVs, harder evaluation)

| Source ID | Split | Stream |
|-----------|-------|--------|
| `ieee_cis_fraud_torch_hard` | Temporal 50/50 train/test | Latest 50% of holdout only |
| `ulb_creditcard_fraud_torch_hard` | Temporal 50/50 | Latest 50% of holdout only |
| `paysim_fraud_torch_hard` | Temporal 50/50 | Latest 50% of holdout, no synthetic shift |

### Tier D2 — Natural drift (non-fraud or sensor)

| Source ID | Why it discriminates |
|-----------|----------------------|
| `uci_gas_sensor_drift_torch` | Sensor batch ordering; covariate drift across batches |
| `openml_electricity_torch` | Chronological market stream |

### Future Tier D3 — ingest from Kaggle

| Source ID | Ingest |
|-----------|--------|
| `elliptic_fraud_torch_hard` | `python3 scripts/ingest_elliptic_kaggle_zip.py --zip ~/Downloads/elliptic-data-set.zip` |
| `baf_fraud_torch_hard` | `python3 scripts/ingest_baf_kaggle_zip.py --zip ~/Downloads/bank-account-fraud-dataset-neurips-2022.zip` |

Synthetic fallbacks (no Kaggle): `python3 scripts/export_elliptic_baf_fraud_data.py`

### Future Tier D3 — add when ready

- Customer replay — chargeback-weighted costs from pilot CSV

## Metrics

Per strategy, per source:

| Metric | Use |
|--------|-----|
| `balanced_accuracy` | Primary when classes are imbalanced |
| `pr_auc` | Ranking quality for rare fraud class |
| `recall_at_precision_80` | Ops-constrained: "how much fraud caught at 80% precision?" |
| `cost_weighted_error` | `(FP×1 + FN×10) / N` by default |
| `second_half_recall` | Late-stream performance |
| `recall_delta_late_minus_early` | Degradation detector |
| `retrain_recommendation_rate` | Ops churn (still matters) |

**Rankable:** spread across strategies ≥ `min_metric_spread` (default 0.005).

**Headroom:** frozen baseline must not already be saturated on imbalance-aware metrics (see report).

## Run

```bash
python3 scripts/run_discrimination_benchmark.py
# or one source:
python3 scripts/run_discrimination_benchmark.py --source ieee_cis_fraud_torch_hard
```

Artifacts:

- `results/discrimination_benchmark/discrimination_report.md`
- `results/discrimination_benchmark/discrimination_report.json`

## Interpretation

- **High metric spread + headroom:** good testbed for comparing controllers/correction variants.
- **Low headroom:** dataset slice still too easy — tighten `stream_tail_fraction` or add Tier D3 data.
- **High spread on cost/recall but flat accuracy:** expected; use these metrics in ablation docs instead of accuracy.

## Relation to production suite

| Suite | Question it answers |
|-------|---------------------|
| `production_benchmark_sota_suite` | Does ARL beat scheduled retrain on realistic fraud ops replay? |
| `discrimination_benchmark_suite` | Can we **tell methods apart** and measure fraud-native quality? |

Run discrimination benchmarks when iterating on correction/governor mechanisms. Run production SOTA when validating outreach claims.
