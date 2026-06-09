# Next-Step Decision Memo

## Executive Call

The project should now **narrow**, not broaden.

The strongest next move is to make the **temporal delayed-feedback controller** the primary research track, while keeping the graph and specialist-memory branches alive as secondary validation tracks.

In other words:

- **primary track:** delayed-feedback controller learning
- **secondary track:** graph-aware safety and shift diagnosis
- **tertiary track:** specialist memory, but only after we have a benchmark where recurrence truly rewards it

This is the best match to the current evidence.

## Status Update

All three immediate recommendations from the previous memo now exist:

- a **temporal benchmark suite** over multiple delay/severity settings
- a **regime-aware delayed bandit** with short-horizon temporal state
- a **recurrence-first temporal benchmark** for testing specialist reuse under returning regimes

The current saved suite is here:

- [temporal_benchmark_suite.md](../results/temporal_benchmark_suite.md)

Current takeaway from that suite:

- regime-aware delayed control looks promising in some long-delay settings
- but it is not yet stable across the full temporal grid

We also now have a saved recurrence-focused result here:

- [recurrence_temporal_benchmark.md](../results/recurrence_temporal_benchmark.md)

That narrows the next step further: the bottlenecks are now **robust temporal-state learning** and **delayed specialist credit/routing**, not just missing infrastructure.

## Why This Is The Right Call

### 1. The controller thesis is already real

Across tabular, image, and graph settings, the stable result is:

- naive continual adaptation is brittle
- safety-gated control reduces risk substantially
- multi-action control is better than one always-on update rule

That is not speculative anymore in this codebase.

### 2. Delayed feedback is now the most scientifically interesting pressure test

The temporal image benchmark now exposes something important:

- immediate-feedback learned control and true delayed-feedback control are not the same
- longer delays measurably hurt learned controller quality
- the current controller stack remains strongest when it uses explicit safety logic rather than relying purely on delayed learning

This is a meaningful research problem rather than just an implementation detail.

### 3. The delayed-hybrid branch is implemented, but it is only partially differentiated

That is useful information.

The current specialist-memory idea is not failing because the code is missing. It is limited because the **delayed controller is still weaker at opening and crediting specialists than the non-delayed hybrid**.

The recurrence-first benchmark is now giving a more precise read:

- the regular `hybrid` controller can open a second specialist and reuse it
- the `delayed_hybrid` controller can now also open multiple specialists
- but it still converts that extra structure into weaker utility than the non-delayed `hybrid`

So specialist-memory should not be the main branch until we either:

- build a benchmark with cleaner recurring regimes and stronger regime identity
- or design a better routing mechanism that can actually separate recurring states

### 4. The graph track is promising, but it is still a safety testbed more than a performance-recovery benchmark

The graph benchmark now clearly justifies topology-aware monitoring, which is good.

But the graph branch is not yet where we should spend the majority of cycles if the near-term goal is a strong paper-quality result. It is better used as a **second-domain validation track** after the temporal controller thesis is stronger.

## Recommended Main Thesis

The best thesis to optimize for now is:

**A risk-aware controller over bounded interventions can maintain better deployment utility than both frozen inference and naive continual adaptation under delayed feedback and nonstationary shift.**

That is narrower and better supported than:

**online unlabeled adaptation will reliably recover the best raw accuracy**

The first thesis is strong, credible, and aligned with what the system is already doing well.

## What We Should Build Next

### Priority 1: Regime-aware delayed controller

The next controller should not just see drift and risk. It should also get a simple regime state.

Best next implementation:

- add a lightweight temporal regime state to the controller context
- include recurrence-sensitive features:
  - recent shift trajectory
  - recent action history
  - recent reliability trend
  - recent risk-capital trend
  - current regime embedding or cluster id

The goal is to help delayed learners distinguish:

- transient noise
- persistent covariate drift
- recurring regimes
- collapse-prone failure states

### Priority 2: Better delayed-feedback learning rule

The current delayed bandit is a good scaffold, but it is still simple.

Most promising next upgrades:

- reward smoothing over a reveal window instead of single-batch reward
- contextual bandit with short temporal state rather than per-batch iid context
- trust weighting on revealed outcomes based on reveal age and reliability mismatch
- separate reward terms for:
  - accuracy gain
  - risk suppression
  - calibration quality
  - unnecessary intervention cost

### Priority 3: Temporal benchmark suite

Before more architecture expansion, build a proper temporal suite around:

- multiple delay settings: `2`, `4`, `8`, `12`
- multiple severities: `standard`, `harsh`, `extreme`
- at least 3 seeds per setting
- summary outputs for:
  - accuracy
  - utility
  - mean risk capital
  - revealed accuracy
  - reveal coverage
  - retrospective reward

This will tell us whether the delayed-learning story is stable or only anecdotal.

### Priority 4: Delayed specialist-memory improvement on the recurrence benchmark

Do **not** tune specialist memory blindly on the generic temporal stream.

Instead, use the recurrence-first benchmark that now exists and improve:

- delayed specialist creation thresholds
- delayed specialist routing features
- delayed specialist credit assignment
- recurrence-sensitive state for specialist reuse

The target is no longer “can specialist memory ever form under delayed feedback?”

It is now:

**can delayed specialist control close the quality gap to the non-delayed hybrid once specialist memory actually forms?**

### Priority 5: Use graph as validation, not as the main optimization loop

Once the temporal controller improves, re-run it on the graph benchmark and ask:

- does topology-aware monitoring improve regime diagnosis?
- does the controller choose different actions under topology shift than under pure feature drift?
- does abstention/reset behavior improve under structural rewiring?

That is a strong “second domain” validation story.

## What We Should Not Do Right Now

### 1. Do not broaden the action library much further

The controller already has enough actions to study the core hypothesis.

Adding more actions now will mostly increase complexity before we know the delayed controller is truly working.

### 2. Do not chase raw accuracy as the only headline metric

The best story in this repo is utility under risk, not absolute top-line accuracy alone.

Accuracy still matters, but the system is most differentiated when it wins on:

- utility
- risk suppression
- calibration/reliability behavior
- robustness to delayed supervision

### 3. Do not make graph the main branch yet

Graph remains promising, but it is not yet the shortest path to a convincing central result.

### 4. Do not over-invest in specialist memory until recurrence is benchmarked properly

Right now the delayed-hybrid result is telling us:

- the idea is implementable
- the current setup does not reward it yet

That means the bottleneck is experimental design or routing quality, not “more generic memory code”.

## Recommended 3-Step Immediate Plan

### Step 1

Refine the **regime-aware delayed bandit** so it behaves more consistently across temporal conditions.

### Step 2

Use the **recurrence-first temporal benchmark** to improve delayed specialist routing and credit assignment.

### Step 3

Expand the **temporal benchmark suite** once the controller changes are in place, rather than broadening the architecture again immediately.

If these three steps go well, then the next paper-quality story becomes much clearer.

## Best Paper-Style Story From Here

If the next phase works, the strongest paper framing is probably:

**Safe Continual Adaptation Under Delayed Feedback: A Controller-Based Approach to Nonstationary Model Reliability**

That is cleaner than trying to tell three stories at once about:

- graph adaptation
- delayed learning
- specialist memory
- general TTA

## Bottom Line

At this point, the highest-value move is:

**turn the temporal delayed-feedback track into a rigorous benchmark-and-controller story, and use graph plus specialist memory as secondary validation branches rather than the main research loop.**
