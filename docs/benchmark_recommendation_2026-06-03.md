# Benchmark Recommendation (2026-06-03)

## Purpose

The repo now has enough evidence to separate two different benchmark jobs:

1. **claim validation**
2. **method discrimination**

Those are not the same thing.

The current production fraud suite is still useful for showing that the flagship path beats strong baselines. But several of those same datasets are too saturated or too low-signal to tell nearby methods apart.

This note defines which datasets we should:

- keep for headline claims
- keep for mechanism comparison
- demote from ranking decisions
- add next

## Core Conclusion

We should stop expecting one benchmark family to do everything.

The right evaluation structure now is:

1. **Flagship claim suite**
2. **Research discrimination suite**
3. **Safety / mismatch suite**

## 1. Flagship Claim Suite

Use this suite to answer:

**Does the current flagship architecture beat frozen and scheduled retrain on realistic temporal fraud replay?**

Primary artifact:

- `results/production_benchmark_sota/suite_report.md`

Keep:

- `ulb_creditcard_fraud_torch`
- `ieee_cis_fraud_torch`
- `paysim_fraud_torch`

Why keep them:

- they are the current best support for the repo's fraud/reliability claim
- they use temporal split plus delayed labels
- they are already part of the production-style story

How to use them:

- use for outreach claims
- use for product positioning
- use for “beats scheduled retrain” statements

How **not** to use them:

- do **not** use them as the main fine-grained architecture-selection suite
- do **not** expect them to cleanly separate nearby correction variants

## 2. Research Discrimination Suite

Use this suite to answer:

**Can we actually tell candidate methods apart on metrics with headroom?**

Primary artifact:

- `results/discrimination_benchmark/discrimination_report.md`

### Keep as active discriminators

#### `ieee_cis_fraud_torch_hard`

Keep as the main fraud-side discriminator.

Why:

- among the fraud sources, it currently has the clearest genuine headroom
- the latest report shows measurable spread on at least some rankable metrics
- it is the best current place to compare nearby fraud mechanisms

Primary metrics:

- `pr_auc`
- `cost_weighted_error`
- `balanced_accuracy`

#### `openml_electricity_torch`

Keep as a non-fraud discriminator.

Why:

- it clearly separates good and bad controller behavior
- `delayed_hybrid` and `regime_aware_delayed_bandit` are **not** tied here
- this is useful as a counterexample and stress case

Primary metrics:

- `balanced_accuracy`
- `cost_weighted_error`

#### `uci_gas_sensor_drift_torch`

Keep as a maintenance-side discriminator.

Why:

- it sharply reveals when a method is actively harmful
- it is still a good mechanism filter even when the best methods tie

Primary metrics:

- `balanced_accuracy`
- `pr_auc`
- `cost_weighted_error`

### Keep as active secondary fraud discriminator

#### `elliptic_fraud_torch_hard`

This is now available and should stay in the fraud-side comparison portfolio.

Why:

- temporal
- graph/risk adjacent
- harder than current easy fraud baselines
- more informative than ULB or PaySim-hard on recall-style metrics

Expected role:

- fraud mechanism comparison
- temporal delayed-feedback ranking

### Demote from mechanism ranking

#### `ulb_creditcard_fraud_torch_hard`

Demote from primary ranking decisions.

Why:

- still too close to ceiling
- almost no metric spread on the latest report
- not a good place to compare nearby methods

Allowed use:

- regression check
- headline stability check

#### `paysim_fraud_torch_hard`

Demote from primary ranking decisions.

Why:

- the current hard slice still yields almost no recall movement
- many metrics remain flat across methods
- useful for sanity, weak for research discrimination

Allowed use:

- regression check
- production-story continuity

#### `baf_fraud_torch_hard`

Demote until we have a more informative slice or different metric setup.

Why:

- currently near-zero useful spread
- not pulling its weight as a discriminator

## 3. Safety / Mismatch Suite

Use this suite to answer:

**Where does a method fail, stand down, or expose a profile mismatch?**

Keep:

- `openml_electricity_torch`
- `uci_gas_sensor_drift_torch`

These are not just alternative benchmarks. They are where we can see whether the system is:

- robust
- over-aggressive
- profile-mismatched

That is useful even when they are not the primary commercial story.

## Recommended New Default Comparison Suite

For day-to-day architecture iteration, the main comparison suite should be:

1. `ieee_cis_fraud_torch_hard`
2. `elliptic_fraud_torch_hard`
3. `openml_electricity_torch`
4. `uci_gas_sensor_drift_torch`

Canonical config now:

- `configs/discrimination_trio_suite.yaml`
- `configs/discrimination_quartet_suite.yaml` for fraud-side iteration with an added Elliptic check
- `configs/discrimination_fraud_context_suite.yaml` for the parallel context-augmented fraud lane

This gives us:

- two fraud discriminators
- one operational drift discriminator
- one maintenance/sensor discriminator
- one graph-adjacent temporal fraud check

### Current read on the context-augmented fraud lane

We now also have a fully parallel fraud-context path:

- `ieee_cis_fraud_torch_context_hard`
- `elliptic_fraud_torch_context_hard`
- controller: `fraud_context_delayed_bandit`

This path is useful because it keeps the existing fraud loaders and controllers intact while testing whether a richer temporal feature surface changes the ranking story.

Current conclusion from `results/discrimination_fraud_context/discrimination_report.md`:

1. the extra temporal-context features make IEEE hard **more discriminating**
2. but `fraud_context_delayed_bandit` still does **not** beat `delayed_hybrid` or `frozen` on the key fraud-side metrics
3. so the next fraud bottleneck is probably **not just representation flattening**

## Metrics To Center

For the comparison suite, stop centering raw accuracy alone.

Use:

- `balanced_accuracy`
- `pr_auc`
- `recall_at_precision_80`
- `cost_weighted_error`
- `late-stream recall delta`

Use raw accuracy only as a supporting metric.

## Practical Decision Rule

When evaluating a new method:

1. First run it on the **research discrimination suite**.
2. If it does not improve at least one rankable metric on `ieee_cis_fraud_torch_hard` or one non-fraud discriminator, do not promote it.
3. If it does improve there, then run it on the **flagship claim suite**.
4. Only use the flagship suite to decide whether the improvement is commercially relevant.

## What To Do Next

### Immediate

1. Make `ieee_cis_fraud_torch_hard`, `openml_electricity_torch`, and `uci_gas_sensor_drift_torch` the default mechanism-comparison trio.
2. Stop using `ulb_creditcard_fraud_torch_hard`, `paysim_fraud_torch_hard`, and `baf_fraud_torch_hard` as primary method-ranking signals.
3. Evaluate new interventions first on the discrimination trio.

### Next ingest target

1. `elliptic_fraud_torch_hard`

That is the most attractive next data addition if we want a stronger fraud-side discriminator.

## Final Read

The current issue is **not** that every method is truly identical.

The issue is that several of the current fraud slices are poor discriminators for nearby methods.

So the right response is:

- do **not** panic about flat top-line ties
- do **not** overfit to saturated fraud slices
- do **tighten the benchmark portfolio** so the right datasets answer the right questions
