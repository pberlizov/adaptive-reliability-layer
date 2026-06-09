# Scale-Up Recommendation

## Bottom Line

The architecture is now mature enough for a **real offline model-and-dataset evaluation**, but it is still too early for a **live database or production integration**.

That is the right boundary.

The current system already has:

- multiple intervention primitives
- learned and memory-based controllers
- sequential risk monitoring
- reset / rollback-like safety behavior
- multi-seed evaluation
- ablation support

What it does **not** yet have is:

- delayed-label handling from a real operational backend
- production logging / replay infrastructure
- realistic latency budgets
- human-review / abstention workflow integration
- evidence on larger public benchmarks with stronger source models

So the best next move is:

**offline real benchmark -> stronger public model -> only then production-style data plumbing**

## What The New Suites Say

### Multi-seed benchmark suite

Saved outputs:

- `results/benchmark_suite.json`
- `results/benchmark_suite.md`

Key signals:

- On the real tabular stream, `frozen` still has the highest raw accuracy mean at `0.961`, but `bandit` and `hybrid` have much better utility/risk tradeoffs:
  - `bandit`: accuracy `0.956`, utility `0.941`, risk capital `4.124`
  - `hybrid`: accuracy `0.959`, utility `0.943`, risk capital `4.118`
  - `frozen`: accuracy `0.961`, utility `0.920`, risk capital `61.292`

- On the harder digits-shift benchmark, controllers consistently improve on frozen:
  - `frozen`: accuracy `0.648`, utility `0.599`, risk capital `77.262`
  - `bandit`: accuracy `0.663`, utility `0.639`, risk capital `5.689`
  - `specialist_memory`: accuracy `0.665`, utility `0.639`, risk capital `7.976`

Interpretation:

- the control layer is clearly real
- the action library is doing meaningful work
- the best architecture is still controller-centric

### Ablation suite

Saved outputs:

- `results/ablation_suite.json`
- `results/ablation_suite.md`

Key signals:

- **Reset is essential.**
  - Tabular: `multi_action_no_reset` drops to `0.673` accuracy with risk capital `58.493`
  - Digits: `multi_action_no_reset` drops to `0.455` accuracy with risk capital `77.598`
  - Bandit without reset is also much worse on both benchmarks

- **Tabular gains rely heavily on label-shift handling.**
  - `multi_action_no_label_shift` is materially worse than `multi_action`

- **Digits gains rely less on label-shift correction and more on safe adaptation plus reset.**
  - `multi_action_no_label_shift` is nearly unchanged from `multi_action`

- **Specialist memory helps, but huge reservoirs are not necessary yet.**
  - Small specialist reservoirs are already competitive

Interpretation:

- reset / recovery behavior is not a side detail
- “which action matters” is benchmark-dependent
- learned control and specialist memory are both worth keeping

## Recommended Next Target

The next experiment should use a **larger public benchmark and a stronger real model**, not a live database.

Best candidates:

1. **A torchvision image backbone with built-in batch norm**
   - example target: a small ResNet on a corruption or domain-shift benchmark
   - why: our current intervention menu naturally supports BN refresh, recalibration, and bounded head/adapter updates

2. **A real temporal or streaming tabular benchmark**
   - example target types: electricity demand, airline delay, sensor drift, fraud-like delayed labels
   - why: it tests whether the controller remains useful under longer-horizon nonstationarity

3. **A graph benchmark after the above**
   - why: graph-native shift remains the strongest long-term differentiator, but the controller abstraction should be firmer first

## Recommendation On “Real Database”

Do **not** wire this into a live database yet.

Reasons:

- the system still needs stronger evidence on public benchmarks
- the intervention API is still research-grade, not operationally hardened
- the most important open questions are scientific, not infrastructure questions

If we want something more “real” immediately, the right compromise is:

**use a real public dataset plus a real pretrained model in an offline replay setup**

That will tell us much more than adding production-style storage right now.
