# Temporal Image Track

## Purpose

This track extends the image benchmark into a delayed-feedback setting.

The main question is:

**what should a controller do when the world is shifting now but the labels that would confirm or refute its behavior arrive only later?**

That is much closer to many real deployments than immediate-label evaluation.

## Current Design

The benchmark is built on the Fashion-MNIST stream and adds:

- a configurable label reveal delay
- separate reporting for:
  - online overall accuracy
  - revealed accuracy within the evaluation horizon
  - revealed coverage
  - eventual accuracy once all delayed labels would arrive

This gives us a place to study how delayed supervision should influence:

- future controller learning
- reliability updates
- reset logic
- specialist creation or retirement

The benchmark now also includes a true delayed-learning controller:

- `delayed_bandit`: chooses actions online like the standard bandit controller, but only updates its linear-UCB reward model when the corresponding labels are revealed
- `delayed_hybrid`: routes through specialist memory and also delays controller credit assignment until labels mature

The delayed controllers now learn from a richer retrospective reward at reveal time that combines:

- immediate batch utility
- batch accuracy relative to the stream's revealed baseline
- a calibration penalty for reliability hindsight error
- revealed-coverage awareness

Those delayed rewards are now also:

- smoothed over recent revealed outcomes instead of being consumed as isolated one-batch targets
- trust-weighted based on reveal age, revealed coverage, and reliability mismatch

## Representative Result

Short 24-step run with `reveal_delay_steps=4`:

- `frozen`: overall `0.816`, revealed coverage `0.833`
- `controller`: overall `0.821`, revealed coverage `0.833`
- `bandit`: overall `0.819`, revealed coverage `0.833`

## Interpretation

The main importance of this track is architectural rather than raw performance.

It proves that the project can now express:

- online decision-making
- delayed supervision
- partial feedback visibility

That matters because many promising future controller ideas, especially learned controllers, will need this kind of evaluation rather than pure fully-observed offline replay.

## Delayed-Feedback Controller Update

Quick 36-step confirmation run with `reveal_delay_steps=6`, `source_train_size=2500`, and `source_epochs=3`:

- `frozen`: overall `0.791`, utility `0.762`, mean risk capital `36.414`
- `controller`: overall `0.809`, utility `0.796`, mean risk capital `4.200`
- `bandit`: overall `0.791`, utility `0.770`, mean risk capital `4.902`
- `delayed_bandit`: overall `0.788`, utility `0.767`, mean risk capital `6.633`

Interpretation:

- the delayed bandit is now genuinely different from the immediate-feedback bandit
- delayed reward arrival modestly hurts controller quality in this confirmation run
- the safety-gated controller still looks strongest on this temporal slice
- this creates the right research pressure for the next step: better delayed-feedback controller learning rather than assuming instant supervision

## Harder Temporal Stress Test

Longer 72-step run with `severity=extreme`, `reveal_delay_steps=12`, `source_train_size=2000`, and `source_epochs=2`:

- `frozen`: overall `0.752`, utility `0.708`, mean risk capital `68.202`
- `controller`: overall `0.748`, utility `0.726`, mean risk capital `5.897`
- `bandit`: overall `0.738`, utility `0.720`, mean risk capital `4.714`
- `delayed_bandit`: overall `0.715`, utility `0.688`, mean risk capital `7.254`
- `delayed_hybrid`: overall `0.715`, utility `0.688`, mean risk capital `7.254`

Delayed-controller diagnostics from that run:

- `delayed_bandit` mean retrospective reward: `0.583`
- `delayed_hybrid` mean retrospective reward: `0.583`
- `delayed_hybrid` specialist count: `1`

Interpretation:

- longer label delays and harsher shifts do create a meaningful penalty for delayed learned control
- the retrospective reward machinery is active and measurable
- the delayed hybrid architecture is now implemented end to end, but it is **not yet** opening useful specialists on this temporal image setup
- that makes specialist-routing quality, not just delayed reward learning, the next bottleneck on this branch

## Temporal Suite Update

We now also have a saved temporal suite at:

- [temporal_benchmark_suite.md](../results/temporal_benchmark_suite.md)

The current reduced confirmation grid uses:

- seeds: `7`, `11`
- severities: `standard`, `extreme`
- reveal delays: `2`, `6`, `12`

Most important pattern from the current suite:

- the new `regime_aware_delayed_bandit` is **not** uniformly best
- and the upgraded `delayed_hybrid` is now a genuinely distinct controller rather than a delayed-bandit clone
- but neither delayed variant is uniformly best across the full grid

Representative wins:

- `standard`, delay `12`:
  - `delayed_bandit`: accuracy `0.596`, utility `0.580`
  - `regime_aware_delayed_bandit`: accuracy `0.610`, utility `0.594`
  - `delayed_hybrid`: accuracy `0.610`, utility `0.594`

- `extreme`, delay `12`:
  - `delayed_bandit`: accuracy `0.545`, utility `0.523`
  - `regime_aware_delayed_bandit`: accuracy `0.566`, utility `0.545`
  - `delayed_hybrid`: accuracy `0.566`, utility `0.545`

- `extreme`, delay `2`:
  - `delayed_bandit`: accuracy `0.527`, utility `0.506`
  - `delayed_hybrid`: accuracy `0.548`, utility `0.527`

Representative failure:

- `extreme`, delay `6`:
  - `delayed_bandit`: accuracy `0.545`, utility `0.524`
  - `regime_aware_delayed_bandit`: accuracy `0.515`, utility `0.494`
  - `delayed_hybrid`: accuracy `0.515`, utility `0.493`

Interpretation:

- adding temporal regime state is promising, especially as delay grows
- reward smoothing and trust weighting did not erase the signal; they made the delayed-learning path more faithful to partial supervision
- the delayed hybrid now sometimes opens specialists and sometimes beats the plain delayed bandit
- but the delayed controller family is still unstable across conditions
- that means the next question is no longer “does temporal context matter at all?”
- it is now “what temporal context and delayed routing/credit rule make delayed control robust across regimes?”

## Recurrence-First Benchmark Update

We now also have a recurrence-focused temporal benchmark at:

- [recurrence_temporal_benchmark.md](../results/recurrence_temporal_benchmark.md)

Representative 72-step run with `reveal_delay_steps=8`, `severity=standard`, `source_train_size=2000`, and `source_epochs=2`:

- `frozen`: overall `0.690`, utility `0.643`, mean risk capital `72.327`
- `controller`: overall `0.696`, utility `0.671`, mean risk capital `6.562`
- `regime_aware_delayed_bandit`: overall `0.699`, utility `0.671`, mean risk capital `7.578`
- `hybrid`: overall `0.695`, utility `0.673`, mean risk capital `5.746`
- `delayed_hybrid`: overall `0.686`, utility `0.659`, mean risk capital `7.416`

Specialist diagnostics from that run:

- `hybrid`: `specialist_count=2`, `reuse_ratio=0.972`
- `delayed_hybrid`: `specialist_count=3`, `reuse_ratio=0.972`

Interpretation:

- the recurrence-first design is finally making specialist memory measurable rather than theoretical
- the delayed hybrid now does open and reuse multiple specialists on this temporal image path
- but it still turns that extra structure into slightly worse overall utility than the non-delayed `hybrid`
- the next bottleneck is therefore no longer “can delayed specialist memory form?” but “can delayed specialist routing and credit assignment make that memory worthwhile?”
