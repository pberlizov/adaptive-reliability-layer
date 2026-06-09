# Accuracy experiment campaign (2026-06-03)

Bounded mechanism experiments on hard fraud temporal slices. Full artifacts: [results/accuracy_experiment_campaign/](../results/accuracy_experiment_campaign/).

## Verdict

**Abandon flagship accuracy track.** Stop rule 3 triggered after 4 configs (~12 min wall clock): no balanced-accuracy gain ≥1pp on IEEE hard or ULB hard vs frozen.

## Frozen baseline (hard slices)

| Source | bal_acc | PR-AUC | recall | R@P≥0.8 |
| --- | ---: | ---: | ---: | ---: |
| `ieee_cis_fraud_torch_hard` | 0.529 | 0.073 | 0.082 | 0.000 |
| `ulb_creditcard_fraud_torch_hard` | 0.833 | 0.668 | 0.667 | 0.667 |
| `paysim_fraud_torch_hard` | 0.500 | 0.103 | 0.000 | 0.000 |

## Experiments run (4/6)

| ID | Mechanism | Best Δ bal_acc (IEEE) | Best Δ bal_acc (ULB) | Notes |
| --- | --- | ---: | ---: | --- |
| exp01 | correction_plus_governor | −0.0005 | +0.0002 | PR-AUC +0.007 IEEE; correction 59% steps |
| exp02 | sensitive monitor + cpg | −0.0005 | +0.0002 | Identical to exp01 |
| exp03 | recall-oriented KPI + cpg | −0.0005 | +0.0002 | KPI does not feed decision path |
| exp04 | 3× threshold learning rate + cpg | −0.0005 | +0.0002 | Threshold **↑** to 0.55 IEEE; still 0 recall delta |

Skipped: exp05 (label_shift bounded), exp06 (proactive drift) — early stop.

## Code added

- `scripts/run_accuracy_experiment_campaign.py` — bounded runner with stop rules
- `configs/accuracy_experiment/base_hard_fraud.yaml` — hard-fraud-only discrimination template
- `policy.threshold_learning_rate` config knob (default 0.10) wired through delayed bandit

## Recommendation for main thread

| Track | Action |
| --- | --- |
| Flagship accuracy | **Stop** — hard slices show no rankable bal_acc headroom for controller tweaks |
| Commercial story | **Continue** — utility vs scheduled retrain + correction+governor |
| Next research | Revealed-loss KPIs, customer replay, Elliptic/BAF ingest for slice quality |

Re-run: `PYTHONPATH=src python3 scripts/run_accuracy_experiment_campaign.py`
