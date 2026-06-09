# Production evidence bar

This document defines when Adaptive Reliability Layer (ARL) is **testing-grade** vs **production-grade**.

Testing-grade means the system runs correctly (CI, sidecar parity, milestones). Production-grade means a buyer can expect measurable outcome improvement on realistic delayed-label streams.

## Evidence layers

| Layer | Testing-grade (today) | Production-grade (target) |
|-------|----------------------|---------------------------|
| Engineering | Tests pass, HTTP parity, audit writes | Same + security strict mode in prod config |
| Economic | utility ≥ frozen OR risk ↓ 50% on one source | utility **+0.5–2 pp** or utility Δ ≥ **+0.01** on **core** sources |
| Risk | Any non-negative risk delta | **≥ 5–10%** alert/capital reduction **or** harmful events avoided > 0 |
| Delay | 2–4 batch steps | **≥ 12** steps on fraud core (chargeback proxy) |
| Stream size | 2k+ rows | **≥ 20k** labeled events on core fraud sources |
| Robustness | One lucky slice | **≥ 2 of 3 core** open fraud sources pass |
| Deployment | Shadow smoke | bounded_auto shows **executed** interventions when policy recommends |

## Open dataset tiers

### Tier A — Core fraud (production claims)

These are large, public, fraud-adjacent streams. Claims should be anchored here.

| Source ID | Dataset | Typical size | Time order | Notes |
|-----------|---------|--------------|------------|-------|
| `ulb_creditcard_fraud` | ULB credit card fraud (OpenML / bundled CSV) | ~285k tx | `Time` column | Natural chronological drift; **no synthetic shift** |
| `ieee_cis_fraud` | IEEE-CIS (Kaggle raw → `ieee_cis_full.csv`) | up to ~590k tx | `TransactionDT` | Place raw files under `data/fraud/raw/` |
| `ieee_cis_fraud_torch` | Same, torch adapter | same | same | For full-intervention path |

### Tier B — Auxiliary fraud / risk

| Source ID | Dataset | Role |
|-----------|---------|------|
| `paysim_fraud` | PaySim-inspired synthetic | Plumbing + regime-shift stress (synthetic shift OK) |
| `openml_credit_g` | OpenML German Credit | Risk-adjacent tabular |
| `paysim_fraud_torch` | PaySim + torch | Sidecar / bounded_auto demo |

### Tier C — Non-fraud wedges (capability, not fraud ROI)

| Source ID | Wedge |
|-----------|-------|
| `openml_electricity` | Predictive maintenance / sensor-safe |
| `breast_cancer` | General tabular sanity |
| `wilds_civilcomments_csv` | Public NLP shift |

## Pass/fail thresholds (suite defaults)

Configured in `configs/production_benchmark_suite.yaml` and enforced by `scripts/run_production_benchmark_suite.py`:

- **Core source pass:** bounded_auto controller vs frozen:
  - `utility_delta ≥ 0.005` **OR** `risk_reduction ≥ 5%`
  - `stream_records ≥ 20_000` (after export + replay sizing)
  - `label_delay_steps ≥ 12` on core fraud
- **Suite pass:** at least **2 of 3** core fraud sources pass
- **Stretch (external marketing):** utility Δ ≥ **0.01** and risk reduction ≥ **10%** on PaySim-class pilot

## How to run

```bash
# 1. Download / export open datasets into data/
python3 scripts/export_open_datasets.py

# 2. Run production benchmark suite (slow — trains models per source)
python3 scripts/run_production_benchmark_suite.py

# 3. Review
cat results/production_benchmark/suite_report.md
```

### IEEE-CIS full (Kaggle)

1. Download [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) (`ieee-fraud-detection.zip`).
2. Ingest in one step:

```bash
python3 scripts/ingest_ieee_kaggle_zip.py --zip ~/Downloads/ieee-fraud-detection.zip
```

Or manually place `train_transaction.csv` (+ optional `train_identity.csv`) in `data/fraud/raw/` and run `python3 scripts/export_open_datasets.py`.

### Strong baselines + SOTA torch suite

Sklearn suite (temporal train split, scheduled-retrain + naive baselines):

```bash
python3 scripts/run_production_benchmark_suite.py
```

SOTA-integrated torch suite (`delayed_hybrid` + `sota:` extensions + baselines):

```bash
python3 scripts/run_production_benchmark_sota_suite.py
```

Reports include utility Δ vs frozen, vs `scheduled_retrain`, vs `naive`, adaptation safety rate, and execution rate.

### ULB credit card fraud (recommended)

```bash
python3 scripts/export_open_datasets.py
```

Downloads from [Zenodo mirror](https://zenodo.org/records/7395559) when available, else OpenML id 1597, else reproducible fallback. Output: `data/fraud/creditcard.csv` (~285k rows real).

## Discrimination benchmark (mechanism comparison)

When Tier A accuracy saturates, use the discrimination suite for method comparison:

```bash
python3 scripts/run_discrimination_benchmark.py
```

See [discrimination_benchmark.md](discrimination_benchmark.md). Hard temporal slices (`*_torch_hard`) and imbalance-aware metrics (PR-AUC, recall@precision, cost-weighted error) are designed to separate controllers when raw accuracy ties.

## What we are not claiming yet

- Customer-specific historical replay
- Calendar-time chargeback delay (weeks/months) — batch-step delay is a proxy
- Live A/B deployment outcomes
- sklearn adapter + `delayed_hybrid` (torch-only specialist routing)
