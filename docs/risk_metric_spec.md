# ARL Risk Metric Specification

*Version 1.0 — 2026-06-05*

This document pre-commits the exact formulas used in the ARL risk monitor before benchmarks are (re-)run.  Any change to these formulas must update this document and bump the version before results are re-reported.

---

## 1. Shift signal

Computed in `TabularShiftMonitor.evaluate` (`tabular_benchmark.py`).

### Feature score

```
normalized_mean_gap  = mean( |batch_mean - ref_mean| / sqrt(ref_var + 1e-6) )
normalized_var_gap   = mean( |batch_var  - ref_var|  / (ref_var + 1e-6) )
feature_score        = normalized_mean_gap + 0.5 * normalized_var_gap
```

### Output score

```
output_score = entropy_gap
             + 0.75 * |mean(batch_probs) - ref_mean_prob|
             + |positive_rate - ref_positive_rate|
             + 0.50 * |mean_confidence - ref_mean_confidence|
```

where `entropy_gap = |mean_entropy(batch) - ref_mean_entropy|` and
`mean_entropy(p) = -p*log(p) - (1-p)*log(1-p)`.

### Collapse risk

```
collapse_risk = max(0, ref_mean_entropy - batch_mean_entropy)
              + max(0, |positive_rate - 0.5| - |ref_positive_rate - 0.5|)
```

### Composite score

```
score = feature_score + 0.75 * output_score + 0.65 * collapse_risk
```

### Thresholds

| Signal | Alert | Severe |
|---|---|---|
| `score` | ≥ 1.10 | ≥ 1.75 |
| `collapse_risk` | — | ≥ 0.30 |

---

## 2. Martingale risk capital

Computed in `MartingaleRiskMonitor.update` (`risk.py`).

```
p_value(t) = (#{s in ref : s >= raw_score(t)} + 1) / (|ref| + 1)
e_value(t) = epsilon * p_value(t)^(epsilon - 1)      (epsilon = 0.5)
capital(t) = clip( capital(t-1) * decay * e_value(t), 1.0, max_capital )
```

| Parameter | Default |
|---|---|
| `epsilon` | 0.5 |
| `decay` | 0.92 |
| `alert_threshold` | 8.0 |
| `max_capital` | 100.0 |

The `raw_score` fed to the risk monitor is:

```
raw_score = output_score + 0.5 * feature_score + collapse_risk
```

(Note: this is intentionally output-heavy; the risk monitor is designed to detect degradation in model outputs, not just covariate shift.)

---

## 3. Controller utility (replay benchmark)

```
utility(t) = batch_accuracy(t)
           - 0.06 * risk_alert(t)
           - 0.03 * min(1, parameter_drift(t))
           - 0.10 * abstained(t)
           - 0.04 * (action_taken(t) == "reset")
```

### Proxy risk reduction (production / HN benchmark)

For the fraud production suite and `arl-hn-launch`, the reported proxy risk
reduction is the strongest available reduction among three operational signals
relative to the frozen baseline:

```
capital_reduction =
    1 - mean_risk_capital(controller) / mean_risk_capital(frozen)
    if mean_risk_capital(frozen) > 0

alert_reduction =
    1 - risk_alert_rate(controller) / risk_alert_rate(frozen)
    if risk_alert_rate(frozen) > 0

retrain_reduction =
    1 - retrain_recommendation_count(controller)
        / retrain_recommendation_count(frozen)
    if retrain_recommendation_count(frozen) > 0

risk_reduction = max(
    capital_reduction,
    alert_reduction,
    retrain_reduction,
)
```

where:

- `mean_risk_capital(strategy)` is the mean martingale capital over replay batches
- `risk_alert_rate(strategy)` is `risk_alert_count / steps`
- `retrain_recommendation_count(strategy)` is the count of batches flagged for retrain escalation

This is intentionally an **operations / reliability** metric, not a raw
accuracy metric.

---

## 4. Production evidence bar

For the fraud production benchmark, a source passes when it clears the configured
evidence bar in `production_benchmark.py`:

```
stream_size >= min_stream_records
AND
(
  utility_delta_vs_frozen >= min_utility_delta
  OR
  risk_reduction >= min_risk_reduction_pct
)
AND
(if enabled) utility_delta_vs_each_baseline >= beat_baselines_min_delta
AND
controller steering is observable on the stream
```

“Controller steering” here includes:

- explicit interventions (`adapt`, `reset`, `recalibrate`, etc.)
- recommendation execution
- non-explicit correction / threshold steering recorded on replay surfaces

---

## 5. CMAPSS Gate B pass condition

A controller strategy **passes Gate B** on a CMAPSS dataset when:

```
ctrl_delta(delayed_bandit) > 0   OR   ctrl_delta(delayed_hybrid) > 0
```

i.e., at least one production-path controller beats frozen at the terminal window.

The stretch bar is `ctrl_delta ≥ 0.086` (10% relative improvement over frozen's terminal gap from 1.0).

---

## Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-06-05 | Initial commit; formulas extracted from code as-is |
