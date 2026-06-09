# Current findings (living summary)

> **Canonical HN launch run (2026-06-08):** Full recorded results → [hn_launch_results_2026-06-08.md](hn_launch_results_2026-06-08.md). Artifacts: [results/hn_launch/comparison_table.md](../results/hn_launch/comparison_table.md).

**Headline:** `delayed_hybrid` passes **3/3 core** fraud sources (ULB, IEEE-CIS, PaySim) vs scheduled retrain + naive. Utility Δ vs scheduled: **+0.51 to +0.54**. Proxy risk ↓: **7.2% (ULB), 8.7% (IEEE), 6.0% (PaySim)**. Extended Elliptic **FAIL** (loses to naive); BAF passes on utility only (0% risk). Discrimination: **1/5 rankable** on hard fraud slices.

**Mechanism note:** on the flagship fraud streams, the win is primarily **narrow controller steering** (correction / threshold path) with explicit mutate-the-model actions used sparingly. That is intentional product behavior, not a benchmark caveat.

---

> **Historical note:** older sections below are retained as research snapshots and may differ slightly from the canonical HN run above.

Canonical bar: [production_evidence_bar.md](production_evidence_bar.md).

## Sklearn production suite (`regime_aware_delayed_bandit` vs frozen)

Artifacts: [results/production_benchmark/suite_report.md](../results/production_benchmark/suite_report.md)


| Source        | Utility Δ vs frozen | Risk ↓ |
| ------------- | ------------------- | ------ |
| ULB           | +0.007              | 0%     |
| IEEE-CIS full | **+0.165**          | 0%     |
| PaySim        | +0.067              | 0%     |


Random 75/25 train split (not temporal). IEEE headline vs frozen is inflated relative to temporal + scheduled baselines.

## SOTA torch production suite (`delayed_hybrid` + SOTA, temporal split)

Artifacts: [results/production_benchmark_sota/suite_report.md](../results/production_benchmark_sota/suite_report.md)

Historical re-run **2026-06-06** after bandit lift signal + behavior routing + prototype label signal improvements.

| Source       | Risk ↓ | Beats scheduled | Beats naive | Suite |
| ------------ | ------ | --------------- | ----------- | ----- |
| ULB torch    | **7.2%** | Yes | Yes | PASS |
| IEEE torch   | **8.7%** | Yes | Yes | PASS |
| PaySim torch | **6.0%** | Yes | Yes | PASS |

Suite **passed 3/3** with `require_beat_baselines: true`. Risk reduction is the stable primary metric. Use the canonical HN run at the top of this file for current public numbers.

## Direct controller head-to-head (2026-06)

Same torch replay, temporal split, identical SOTA config. Primary `delayed_hybrid` vs comparison `regime_aware_delayed_bandit`.

Artifacts: [results/production_benchmark_head_to_head/head_to_head_report.md](../results/production_benchmark_head_to_head/head_to_head_report.md)


| Source       | delayed_hybrid | regime_aware_delayed_bandit | Δ     |
| ------------ | -------------- | --------------------------- | ----- |
| ULB torch    | 0.942          | 0.942                       | 0.000 |
| IEEE torch   | 0.885          | 0.885                       | 0.000 |
| PaySim torch | 0.905          | 0.905                       | 0.000 |


**3/3 ties** on mean utility. Both beat frozen and scheduled; specialist/coreset routing adds **no measurable utility** over the regime-aware delayed bandit on these streams. Lift vs strong baselines is from **shared delayed bandit + residual correction**, not hybrid routing.

## Implications

1. Production claims should cite **temporal split + scheduled/naive baselines** (SOTA suite), not sklearn-vs-frozen alone.
2. Architecture direction: **correction-first core + hybrid governor shell**; mechanism ablation and revealed-loss KPIs are the next evidence levers.
3. Risk story: composite proxy risk reduction is now **7–9% vs frozen** on core fraud sources (2026-06-03 re-run); stretch bar (10%) still open on IEEE.

## Discrimination benchmark (2026-06)

When core fraud accuracy saturates, use the discrimination suite for mechanism comparison.

- Doc: [discrimination_benchmark.md](discrimination_benchmark.md)
- Portfolio recommendation: [benchmark_recommendation_2026-06-03.md](benchmark_recommendation_2026-06-03.md)
- Config: `configs/discrimination_benchmark_suite.yaml`
- Default comparison config: `configs/discrimination_trio_suite.yaml`
- Run: `python3 scripts/run_discrimination_benchmark.py`

Adds hard temporal slices (`ieee_cis_fraud_torch_hard`, etc.), natural drift sources (gas sensor, electricity), and metrics: balanced accuracy, PR-AUC, recall@precision≥0.80, cost-weighted error, late-stream recall delta, explicit metric-spread tables.

### Fraud-ranking + segment-aware quartet follow-up

Artifacts:

- [results/discrimination_quartet/discrimination_report.md](../results/discrimination_quartet/discrimination_report.md)
- [configs/discrimination_quartet_suite.yaml](../configs/discrimination_quartet_suite.yaml)

We then implemented three bigger fraud-side changes in parallel:

1. a **true pairwise ranking update** inside the torch head+adapter path
2. **segment-aware fraud ranking features** in the delayed rank corrector
3. a stronger fraud-side comparison suite that adds **`elliptic_fraud_torch_hard`**

The result is a useful negative one:

- `fraud_rank_delayed_bandit` did **not** improve the main fraud discriminators
- on `ieee_cis_fraud_torch_hard`, it underperformed `regime_aware_delayed_bandit` on PR-AUC (`0.077` vs `0.080`) and recall (`0.066` vs `0.082`)
- on `elliptic_fraud_torch_hard`, it also underperformed both `regime_aware_delayed_bandit` and `delayed_hybrid` on balanced accuracy, recall, and cost-weighted error

At the same time, the new path was **not** uniformly bad:

- on `openml_electricity_torch`, it actually became the strongest controller in the quartet (`0.878` balanced accuracy vs `0.848` for `delayed_hybrid`)
- on `uci_gas_sensor_drift_torch`, it behaved identically to the failing `regime_aware_delayed_bandit` path

So the updated interpretation is:

1. the new pairwise/segment-aware ranking path is **not yet a fraud win**
2. `delayed_hybrid` remains the safest general robustness shell
3. `regime_aware_delayed_bandit` remains the better simple fraud-side ranking specialist than the new fraud-rank variant
4. the next fraud bottleneck is probably **feature/slice quality**, not just “add a stronger ranking loss”

### Fraud-context feature follow-up

Artifacts:

- [results/discrimination_fraud_context/discrimination_report.md](../results/discrimination_fraud_context/discrimination_report.md)
- [configs/discrimination_fraud_context_suite.yaml](../configs/discrimination_fraud_context_suite.yaml)

We then tried the next parallel step without replacing any existing path:

1. append **causal temporal-context features** to the hard fraud datasets (`ieee` and `elliptic`)
2. add a separate **`fraud_context_delayed_bandit`** controller that segments on those prepended context features
3. evaluate it against:
   - `regime_aware_delayed_bandit`
   - `fraud_rank_delayed_bandit`
   - `delayed_hybrid`

This produced a very useful result:

- the new context features made the fraud benchmarks themselves more discriminating, especially IEEE hard
- but the **specialized fraud controllers still did not win**

On `ieee_cis_fraud_torch_context_hard`:

- `frozen`: PR-AUC `0.109`, balanced accuracy `0.556`, recall `0.148`
- `delayed_hybrid`: PR-AUC `0.100`, balanced accuracy `0.555`, recall `0.148`
- `regime_aware_delayed_bandit`: PR-AUC `0.088`, balanced accuracy `0.549`, recall `0.131`
- `fraud_context_delayed_bandit`: PR-AUC `0.087`, balanced accuracy `0.532`, recall `0.098`

So the temporal-context representation **increased headroom**, but the new fraud-context controller still underperformed the safer baseline and even the frozen model.

On `elliptic_fraud_torch_context_hard`:

- `delayed_hybrid` remained strongest overall on balanced accuracy, recall, and cost-weighted error
- `fraud_context_delayed_bandit` ended up essentially tied with `fraud_rank_delayed_bandit`

So the updated interpretation is:

1. the fraud-side bottleneck is **not just missing temporal context**
2. the fraud-side bottleneck is also **not just missing ranking loss**
3. `delayed_hybrid` remains the best overall robustness shell
4. the next fraud-specific gain probably requires:
   - better benchmark slices,
   - richer raw fraud features or segment metadata,
   - or a different objective than these bounded local ranking/correction updates

## Correction-centric parallel path (2026-06)

Artifacts: [results/correction_path_evaluation/correction_path_evaluation.md](../results/correction_path_evaluation/correction_path_evaluation.md)

We evaluated two explicit parallel variants against the same fraud SOTA suite:

- `correction_only`: keep delayed correction, suppress policy actions
- `correction_plus_governor`: keep delayed correction and governor logic, suppress explicit actions

Verdict:

- `correction_only` **fails hard**. Mean utility delta vs frozen is `+0.000`, and it trails the full hybrid by about `0.063` utility on average.
- `correction_plus_governor` **passes cleanly**. It matches the current full hybrid on all three core fraud sources while using `0.000` mean explicit action rate.

Per-source utility for `correction_plus_governor`:


| Source       | correction+governor | full hybrid | Δ vs scheduled |
| ------------ | ------------------- | ----------- | -------------- |
| ULB torch    | 0.942               | 0.942       | +0.049         |
| IEEE torch   | 0.885               | 0.885       | +0.049         |
| PaySim torch | 0.902               | 0.902       | +0.047         |


Current best interpretation:

1. The project should **not** pivot to pure correction-only.
2. The evidence **does** support a narrower architecture: **delayed correction as the main mechanism, with the governor retained as the safety/control shell**.
3. Explicit action machinery is currently not carrying the flagship fraud win.

### Regime-specific residual correction follow-up

We upgraded the delayed correction path to include a small **regime-local residual head** plus a short-horizon residual bias. This is an accuracy-seeking intervention, not a governance change.

Latest read:

- `correction_only` is still not viable as a standalone architecture.
- `correction_plus_governor` still passes cleanly and remains the best simplified parallel path.
- the new local residual head produced only a **small lift**, mainly on IEEE (`0.885 -> 0.886` utility on the saved run).

So this was directionally right, but **not a breakthrough**. The conclusion is that regime-local residual correction helps a bit, but not enough by itself to change the project thesis.

### Correction-expert mixture follow-up

We then upgraded the same fraud correction path again with a small **correction-expert mixture**:

- `recurring` expert
- `transition` expert
- `high_risk` expert

These are gated by the same delayed regime state and pending-feedback context already used by the controller.

Result:

- still **no meaningful change** to the main story
- `correction_only` remains a clear failure
- `correction_plus_governor` remains the right simplified path
- the only measurable move was another tiny IEEE improvement (`0.824 -> 0.825` on correction-only, while full/governor paths stayed effectively unchanged)

That means “better residual mixing” is probably **not** the next big lever. The next accuracy-seeking intervention should likely be stronger than residual shaping alone, such as threshold correction or bounded revealed-label head updates.

### Threshold-correction follow-up

We then added a learned **decision-threshold correction** path to the same delayed fraud controller and wired it through the runtime so predictions no longer have to use a fixed `0.5` cutoff.

Result:

- no meaningful change in the flagship fraud suite
- `correction_only` still fails
- `correction_plus_governor` still passes, but with effectively unchanged top-line metrics

So the current evidence is:

1. residual shaping alone is too weak
2. residual expert mixing is also too weak
3. threshold correction, in this bounded form, is also too weak

That strongly points to the next accuracy intervention being a **stronger model-side update**, most likely a bounded revealed-label head update rather than more post-hoc correction variants.

### Bounded revealed-label head-update follow-up

We then implemented a bounded **supervised head-only update** on the torch fraud path, triggered only when delayed revealed labels arrive and the correction stack is not already clearly outperforming the baseline. The update is:

- head-only
- anchor-regularized back to the source model
- drift-projected to stay within the existing bounded-adaptation budget
- applied only on sufficiently large revealed batches

Result:

- still **no meaningful flagship change**
- `correction_only` remains a clear failure
- `correction_plus_governor` still passes cleanly and remains effectively identical to the full hybrid on all three core fraud sources
- the top-line correction-path report is unchanged at the level that matters:
  - mean utility delta vs frozen for `correction_plus_governor`: `+0.064`
  - mean utility delta vs scheduled retrain: `+0.049`
  - mean utility gap vs full hybrid: `-0.000`

So this is a very useful negative result. It means the next bottleneck is probably **not** “we need a slightly stronger bounded head update.” It is more likely one of:

1. the fraud win is already close to the ceiling available under this replay setup
2. the biggest leverage is still in **correction + governor timing**, not in small supervised weight updates
3. any further accuracy jump will require a stronger intervention class than incremental bounded head-only updates

That makes the current picture sharper:

- correction-only is too weak
- correction + governor is the right simplified flagship path
- explicit actions are not carrying the fraud win
- incremental post-hoc and head-only accuracy tweaks are now showing diminishing returns

### Bounded revealed-label head+adapter update follow-up

We then stepped up one intervention class further and tested a bounded **supervised head+adapter update** on the torch fraud path:

- revealed-label supervised
- adapter + head, not just head-only
- anchor-regularized to source
- projected back into the same bounded parameter-drift budget
- triggered only when delayed outcomes suggest the correction stack is not already comfortably ahead

Result:

- still **no meaningful change** to the flagship correction-path evaluation
- `correction_only` still fails
- `correction_plus_governor` still passes cleanly with the exact same top-line metrics as before:
  - mean utility delta vs frozen: `+0.064`
  - mean utility delta vs scheduled retrain: `+0.049`
  - mean utility gap vs full hybrid: `-0.000`

This is an important negative result. It means we have now tried:

1. local residual correction
2. correction-expert mixtures
3. threshold correction
4. bounded head-only revealed-label updates
5. bounded head+adapter revealed-label updates

and none of them materially improved the flagship fraud lane.

So the current best interpretation is that the fraud path is either:

- already near the ceiling available under this replay setup, or
- bottlenecked by something more structural than “slightly stronger bounded correction/adaptation”

That makes the next move less about another local intervention tweak and more about deciding whether to:

- change the benchmark/problem contract,
- use a stronger adaptation class than our current bounded budget allows,
- or accept `correction + governor` as the real flagship and optimize around that simpler mechanism.

### Trusted-sample subspace adapter update follow-up

Based on the 2025–2026 tabular TTA literature, we then implemented a more targeted intervention:

- use only high-confidence revealed-label samples
- restrict adapter updates to a compact bottleneck subspace chosen by source-normalized activation shift
- keep source anchoring and bounded drift projection
- scale aggressiveness by pending-feedback pressure

This was intended to be meaningfully different from the earlier blunt head-only and head+adapter updates.

Result:

- the flagship fraud comparison is still effectively unchanged
- `correction_only` still fails
- `correction_plus_governor` still passes with the same top-line metrics:
  - mean utility delta vs frozen: `+0.064`
  - mean utility delta vs scheduled retrain: `+0.049`
  - mean utility gap vs full hybrid: `-0.000`

This is another important negative result. It suggests the current fraud lane is not bottlenecked simply by:

- trusted sample selection
- compact adapter subspace restriction
- or slightly better supervised adaptation targeting

So the evidence is now pushing toward a stronger conclusion:

1. `correction + governor` is the real flagship path today
2. bounded local adaptation tweaks are showing strong diminishing returns
3. the next useful leap is more likely to come from a bigger architecture change, a stronger adaptation class, or a harder / more discriminating benchmark contract rather than another nearby intervention variant
