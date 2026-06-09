# State-of-the-Art Search

Date: 2026-06-02

## Goal

This note summarizes recent primary-source literature most relevant to the current bottlenecks in the Adaptive Reliability Layer:

- safe continual test-time adaptation
- delayed-feedback control
- memory / retrieval under recurring regimes
- maintenance-style covariate drift
- uncertainty under ongoing shift

The goal is not to list every adjacent paper. It is to identify the papers that most directly imply architectural changes for this repository.

## High-Level Read

The literature is getting stronger on:

- continual test-time adaptation under long streams
- delayed / online feedback in bandit-style decision making
- memory and retrieval for recurring domains
- online conformal methods under shift
- graph-specific structural adaptation

But there is still not a clean, dominant systems design for:

**safe closed-loop adaptation under delayed feedback with multiple drift families**

That is good news for this project. It means we should borrow aggressively from the best pieces of the literature, but the repo still has room to be architecturally distinct.

The most useful conclusion from this search is:

**we should push toward a two-level architecture:**

1. a common adaptive runtime shell
2. profile-specific adaptation modules chosen by shift type and recurrence state

## Near-Term Implementation Backlog (2026-06)

The current fraud flagship is now clear enough that the backlog should be narrower and more intervention-oriented. Based on the latest papers and the repo's current negative results on local correction tweaks, the highest-priority implementation bets are:

1. **Trusted-sample subspace adapter update**
   - inspired by recent tabular TTA work such as `PFT3A` and `OT3A`
   - use only high-confidence, regime-consistent, revealed-label samples
   - update a compact adapter subspace rather than the full bounded adaptation space
   - keep source anchoring and drift projection

2. **Multi-timescale correction/adaptation experts**
   - inspired by online nonstationary black-box adaptation work
   - maintain short / medium / long horizon correction or update experts
   - choose among them online instead of using a single blended residual path

3. **Outstanding-feedback-aware update budgeting**
   - inspired by delayed-bandit work
   - scale adaptation aggressiveness by missing-feedback mass, not just nominal delay steps
   - skip or sharply downweight updates when the pending queue is stale or overloaded

4. **Richer score recalibration**
   - move beyond a single global threshold shift
   - test piecewise / quantile / tail-specific fraud score recalibration

5. **Feature-refresh + label-shift pairing**
   - especially for fraud-like prior drift with mild representation mismatch
   - couple output correction with a lightweight source-anchored feature refresh, rather than treating them as independent paths

Implementation order:

1. trusted-sample subspace adapter update
2. outstanding-feedback-aware update budgeting
3. multi-timescale correction experts

Success bar:

- beat the current `correction_plus_governor` fraud flagship on the temporal torch suite
- or explain cleanly that the current replay setup is already near its ceiling

## Most Relevant Papers

### 0. 2026 update

The initial version of this memo leaned heavily on 2024–2025 work because that is where much of the core CTTA literature clustered. Since the current date is **June 2, 2026**, we also did a targeted 2026 pass. The most important 2026 additions are:

- drift-aware resetting and recovery (`RDUMB++`)
- asynchronous / decoupled inference-time adaptation (`Caravan`)
- tabular-specific online TTA (`PFT3A`, `OT3A`)
- time-series output-space adaptation with observed labels (`COSA`)
- graph structural alignment at test time (`TSA`)
- specialist-style expert reuse (`IMSE`)

These do not overturn the earlier recommendations, but they sharpen them substantially.

### 1. The Entropy Enigma: Success and Failure of Entropy Minimization

Source:
[ICML 2024](https://proceedings.mlr.press/v235/press24a.html)

Why it matters:

- shows why entropy minimization helps briefly and then fails
- explains over-adaptation as representation drift away from source structure
- strongly supports bounded, early-stop, source-anchored adaptation rather than prolonged unsupervised updates

Architectural implication:

- keep adaptation short-horizon and reversible
- measure representation displacement, not just output confidence
- add explicit stop conditions based on source-distance growth

### 2. Reshaping the Online Data Buffering and Organizing Mechanism for Continual Test-Time Adaptation

Source:
[arXiv 2407.09367](https://arxiv.org/abs/2407.09367)

Why it matters:

- frames online buffering and data organization as a first-class CTTA problem
- directly addresses error accumulation and catastrophic forgetting

Architectural implication:

- our replay / support-state / specialist memory work is on the right path
- memory should be organized, not just stored
- specialist support batches should probably become clustered, windowed memories with recency and recurrence scores

### 3. DPCore: Dynamic Prompt Coreset for Continual Test-Time Adaptation

Source:
[ICML 2025 / OpenReview](https://openreview.net/forum?id=E5MQRICtwq)

Why it matters:

- pushes the field toward dynamic coresets rather than naive memory
- emphasizes selecting a compact but representative adaptation substrate

Architectural implication:

- our specialist memory should likely move from “few snapshots + support batches” toward:
  - small recurrence-aware coresets
  - compact regime descriptors
  - explicit reuse-quality tracking

### 4. Adaptive Retention & Correction: Test-Time Training for Continual Learning

Source:
[ICLR 2025](https://openreview.net/forum?id=9bLdbp46Q1)

Why it matters:

- introduces explicit out-of-task detection plus separate retention and correction mechanisms
- conceptually close to our emerging split between regime detection and action choice

Architectural implication:

- separate:
  - recurrence / task identity detection
  - representation retention
  - prediction correction
- this supports a three-part runtime:
  - regime detector
  - memory selector
  - bounded intervention controller

### 5. Online Feature Updates Improve Online (Generalized) Label Shift Adaptation

Source:
[NeurIPS 2024 / OpenReview](https://openreview.net/forum?id=HNH1ykRjXf)

Why it matters:

- directly relevant to our fraud path
- argues that online label-shift adaptation should not only update the output layer or priors, but also improve feature representations using unlabeled data

Architectural implication:

- our fraud profile should evolve from:
  - `label_shift` correction only
to:
  - `label_shift` correction
  - optional feature-refresh or boundary-refresh when the label-shift signal is persistent

### 6. Adapting to Online Distribution Shifts in Deep Learning: A Black-Box Approach

Source:
[AISTATS 2025](https://proceedings.mlr.press/v258/baby25a.html)

Why it matters:

- proposes a meta-algorithm that maintains only `O(log T)` online learners with different attention spans
- chooses the effective history length adaptively under nonstationarity

Architectural implication:

- we should seriously consider replacing part of the current maintenance controller with:
  - multi-timescale experts
  - short / medium / long attention spans
  - online selection among them

This is especially promising for the maintenance branch, where “how much history matters right now?” is clearly one of the core unresolved questions.

### 7. Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data

Source:
[NeurIPS 2024 / OpenReview](https://openreview.net/forum?id=p3tSEFMwpG)

Why it matters:

- treats temporal drift in tabular data as an in-context inference problem
- uses structural causal model priors that evolve over time

Architectural implication:

- not a drop-in replacement for the current runtime
- but conceptually very important:
  - regime identity may be better represented by recent labeled/unlabeled context windows than by scalar drift scores alone

This suggests the regime encoder should eventually consume:

- recent batch summaries
- action history
- recent prediction behavior
- perhaps a small context window of actual examples

### 8. TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules

Source:
[ICML 2024](https://proceedings.mlr.press/v235/ren24b.html)

Why it matters:

- one of the few strong tabular-specific TTA papers
- argues that tabular adaptation should exploit structured feature knowledge, not only generic entropy minimization

Architectural implication:

- for tabular maintenance streams, pure generic feature correction is likely too weak
- we should consider feature-group-aware or rule-aware bounded corrections

This is especially relevant for electricity-like streams where our current actions are safe but not semantically targeted.

### 9. Test-Time Calibration: A Framework for Personalized Test-Time Adaptation in Real-World Biosignals

Source:
[CHIL 2025](https://proceedings.mlr.press/v287/jo25a.html)

Why it matters:

- real-world biosignal setting is close in spirit to our maintenance branch
- emphasizes calibration and personalization as central test-time mechanisms

Architectural implication:

- maintenance streams may benefit more from calibration-style adaptation than generic adaptation
- supports adding profile-specific “calibration first” behavior for sensor domains

### 10. A Best-of-both-worlds Algorithm for Bandits with Delayed Feedback with Robustness to Excessive Delays

Source:
[NeurIPS 2024 / OpenReview](https://openreview.net/forum?id=LDzrQB4X5w)

Why it matters:

- delayed-bandit literature is more mature than CTTA on how to reason about missing feedback
- key idea: what matters is the amount of missing information, not just elapsed delay
- includes adaptive skipping of excessively delayed observations

Architectural implication:

- our delayed controllers should track:
  - outstanding feedback count
  - feedback staleness
  - skipped / downweighted late observations

This suggests a stronger delayed controller state than just `reveal_delay_steps`.

### 11. A Conformal Martingales Approach for Recurrent Concept Drift

Source:
[COPA 2025](https://proceedings.mlr.press/v266/eliades25a.html)

Why it matters:

- directly relevant to recurrent regimes
- selects an earlier model when recent data looks exchangeable with the post-training window of that model
- only trains a new model if no stored model passes the recurrence checks

Architectural implication:

- this is one of the best external validations of our “specialist memory should be conditional” thesis
- we should likely add a recurrence gate based on:
  - exchangeability or drift test with prior windows
  - historical specialist success threshold

### 12. Online Conformal Prediction via Online Optimization

Source:
[ICML 2025](https://proceedings.mlr.press/v267/areces25a.html)

Why it matters:

- provides online conformal algorithms with coverage guarantees in adversarial and stochastic settings

Architectural implication:

- our uncertainty layer should evolve from static conformal recalibration toward online conformal optimization
- this is especially attractive for the runtime layer because it gives us a principled online uncertainty substrate independent of adaptation details

### 13. Adapting Prediction Sets to Distribution Shifts Without Labels

Source:
[UAI 2025](https://proceedings.mlr.press/v286/kasa25a.html)

Why it matters:

- improves conformal prediction under shift using unlabeled target data
- very aligned with our “bounded adaptation + uncertainty” setup

Architectural implication:

- uncertainty adaptation should be treated as its own action family
- not just a report layer on top of model adaptation

### 14. Conformal Predictive Systems Under Covariate Shift

Source:
[COPA 2024](https://proceedings.mlr.press/v230/jonkers24a.html)

Why it matters:

- extends predictive distributions under covariate shift using weighting ideas

Architectural implication:

- maintenance domains may benefit from weighted predictive systems rather than only threshold recalibration

### 15. GCAL: Adapting Graph Models to Evolving Domain Shifts

Source:
[ICML 2025](https://proceedings.mlr.press/v267/qiao25a.html)

Why it matters:

- one of the clearest graph papers on evolving domain shifts
- uses bilevel adaptation and generated graph memories to mitigate forgetting

Architectural implication:

- validates our view that the graph branch should be memory-based, not just statistic-based
- suggests the graph track should eventually maintain compressed structural memories, not only topological monitors

### 16. RDUMB++: Drift-Aware Continual Test-Time Adaptation

Source:
[ICLR 2026 TTU Workshop](https://openreview.net/forum?id=lWu9V8z4cf)

Why it matters:

- one of the clearest 2026 validations of reset-heavy long-horizon CTTA
- uses entropy- and KL-based drift scoring with adaptive resets
- specifically targets long streams where adaptation collapse accumulates over time

Architectural implication:

- strengthens our existing “reset logic is core” conclusion
- suggests we should explicitly add:
  - drift-specific reset thresholds
  - reset-strength policies
  - reset diagnostics in the runtime

### 17. Caravan: Asynchronous Test-Time Adaptation for Faster Inference

Source:
[ICLR 2026 TTU Workshop](https://openreview.net/forum?id=lyGzOZH4at)

Why it matters:

- separates inference from adaptation updates
- uses sample filtering plus gradient-consistency filtering to make lagged updates safer

Architectural implication:

- supports a future product/runtime split between:
  - fast inference path
  - lower-priority update path
- this is especially relevant for production latency once we move beyond offline replay

### 18. Prior-free Tabular Test-time Adaptation (PFT3A)

Source:
[ICLR 2026 Poster](https://openreview.net/forum?id=BgSDPE24pa)

Why it matters:

- directly focused on tabular TTA without source access or prior knowledge
- explicitly addresses simultaneous label and feature shift

Architectural implication:

- our tabular branch should continue treating mixed feature/label drift as first-class
- strengthens the case for separate fraud-style and maintenance-style tabular profiles

### 19. Online Test-Time Adaptation in Tabular Data with Minimal High-Certainty Samples (OT3A)

Source:
[ICML 2026 FMSD Workshop](https://openreview.net/forum?id=rmpLcZtJ4l)

Why it matters:

- very recent, tabular-specific, and online
- combines target label-distribution estimation with self-training only on high-confidence and domain-consistent pseudo-labels

Architectural implication:

- for fraud-like tabular streams, we should likely add:
  - a domain-consistency filter
  - a high-confidence target prior estimator
  - a stricter pseudo-label gate before any self-training-style update

### 20. NEO: No-Optimization Test-Time Adaptation through Latent Re-Centering

Source:
[ICLR 2026 Poster](https://openreview.net/forum?id=mVlIKLiizr)

Why it matters:

- argues for latent re-centering without online optimization
- highly relevant to our maintenance branch where gradient-based updates have been brittle

Architectural implication:

- we should consider a maintenance-side action that is closer to:
  - latent/feature recentering
  - distribution alignment
than to gradient updates

This may be a better fit than the current `covariate_refresh` implementation.

### 21. COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting

Source:
[ICLR 2026 Poster](https://openreview.net/forum?id=L7Z5wBMPrW)

Why it matters:

- very relevant to delayed-feedback temporal settings
- treats recent observed labels as context for a lightweight residual correction head
- updates only the output-space adapter under leakage-free delayed supervision

Architectural implication:

- this is one of the strongest direct inspirations for our delayed branch
- we should likely add a small residual correction head driven by:
  - recent revealed labels
  - recent prediction statistics
  - regime context

This may be more stable than broader adaptation for delayed-feedback temporal tasks.

### 22. TRUST: Trajectory-guided State-Space Temporal Test-Time Adaptation

Source:
[OpenReview 2026](https://openreview.net/forum?id=CSking4YcX)

Why it matters:

- uses temporal smoothness, cached state, and filtering instead of generic online optimization

Architectural implication:

- more evidence that temporal adaptation should look like:
  - state estimation
  - filtering
  - trajectory-aware correction
not just parameter updates

### 23. IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation

Source:
[ICLR 2026 Poster](https://openreview.net/forum?id=eZO38vANPM)

Why it matters:

- directly about reuse of knowledge from earlier domains
- another strong signal that expert/specialist mixtures are a real state-of-the-art direction

Architectural implication:

- validates keeping the specialist-memory branch alive
- but pushes us toward more explicit expert mixture / gating machinery

### 24. Structural Alignment Improves Graph Test-Time Adaptation (TSA)

Source:
[AISTATS 2026 Poster](https://openreview.net/forum?id=8Q3qQxmlkJ)

Why it matters:

- graph-specific 2026 paper focused on structure shifts
- combines uncertainty-aware neighborhood weighting, adaptive self-vs-neighbor balance, and decision-boundary refinement

Architectural implication:

- when we return to the graph branch, it should not stop at topology alarms
- the graph path should eventually have structure-aware bounded adaptation actions

## What The Literature Suggests We Should Change

## A. Make recurrence detection first-class

Most useful sources:

- ARC
- recurrent concept drift with conformal martingales
- DPCore

Recommended change:

- create a dedicated recurrence module that decides:
  - known regime
  - uncertain recurrence
  - novel regime

Inputs:

- regime embedding
- recent drift signature trajectory
- recent confidence / calibration trend
- specialist historical success
- exchangeability-style recurrence test

Output:

- `recurrence_confidence`
- `reuse_candidate_ids`
- `should_create_new_specialist`

## B. Add multi-timescale control

Most useful source:

- black-box online shift adaptation

Recommended change:

- maintain several lightweight controller states with different effective history lengths
- choose among them online

Why:

- this directly addresses one of our clearest empirical bottlenecks:
  - maintenance streams seem highly sensitive to attention span

This could be much more important than another action primitive.

## C. Separate uncertainty adaptation from model adaptation

Most useful sources:

- online conformal prediction via online optimization
- adapting prediction sets without labels
- conformal predictive systems under covariate shift

Recommended change:

- add a parallel uncertainty controller with actions like:
  - online conformal update
  - weighted conformal / weighted predictive system
  - abstention threshold update

This should become its own runtime lane, not just a passive output layer.

## D. Make delayed control aware of outstanding feedback

Most useful source:

- best-of-both-worlds delayed bandits

Recommended change:

- add controller state:
  - number of pending outcomes
  - age distribution of pending outcomes
  - stale-feedback skip / downweight logic

Current delayed control is modeled mostly through fixed reveal delays. The literature suggests we should explicitly reason about missing-information volume.

## E. Make maintenance adaptation more structured

Most useful sources:

- TabLog
- Test-Time Calibration for biosignals

Recommended change:

- for maintenance / sensor profiles, add feature-group-aware or calibration-first bounded actions
- avoid over-relying on generic entropy or raw feature refresh

This is the branch where our current architecture is still weakest.

## F. Turn memory into compact regime substrates

Most useful sources:

- DPCore
- reshaped buffering for CTTA
- GCAL

Recommended change:

- store for each specialist:
  - compact support coreset
  - regime descriptor
  - historical utility / reuse quality
  - exchangeability anchor window

This is stronger than the current “snapshot plus support-state” design.

## Ranked Next Architectural Bets

### Highest-value now

1. **Recurrence gate with exchangeability-style retrieval**
2. **Delayed-feedback residual correction head**
3. **Multi-timescale controller / attention-span selection**
4. **Outstanding-feedback-aware delayed control**

These three are the best fit for our current bottlenecks and current codebase.

### Strong second wave

5. **Uncertainty controller using online conformal optimization**
6. **Maintenance-side latent re-centering / recentering-style action**
7. **Maintenance-specific structured bounded actions**

### Longer-term / differentiated moat

8. **Graph memory with structural recurrence retrieval**
9. **Asynchronous inference/update execution path**

## What I Would Build Next

If choosing only one serious architecture change, the best next move is:

**add a recurrence gate that combines regime embeddings with an exchangeability-style retrieval test**

Why this first:

- it directly strengthens specialist memory
- it is supported by the concept-drift literature
- it fits our current runtime and replay setup
- it is likely easier to validate than a full uncertainty-controller overhaul

If choosing two:

1. recurrence gate
2. delayed-feedback residual correction head

If choosing three:

1. recurrence gate
2. delayed-feedback residual correction head
3. delayed controller with pending-feedback awareness

## Bottom Line

The literature does not suggest that we should abandon the current architecture.

It suggests that the repo is already pointed in the right direction:

- bounded interventions
- delayed-feedback control
- conditional memory reuse
- profile-specific runtime behavior

But to get closer to real state-of-the-art behavior, the architecture should become more explicit about:

- recurrence
- timescale selection
- uncertainty control
- delayed-feedback load

Those appear to be the most promising paths from “interesting adaptive runtime” to “genuinely state-of-the-art adaptive runtime.”

## Implemented (runtime/sota, May 2026)

The following SOTA items are wired into `ReliabilityLayer` under `runtime/sota/` and `sota:` config (see `configs/default.yaml`, `configs/serving_pilot_fraud_torch.yaml`):

| Feature | Module | Config flag |
|---------|--------|-------------|
| ASR collapse + selective reset advice | `collapse_asr.py`, `asr_reset.py` | `asr_reset_enabled` |
| CDSeer-style drift detector | `drift_detector.py` | `drift_detector_enabled` |
| Online conformal uncertainty lane | `online_conformal.py` | `online_conformal_enabled` |
| Multi-timescale controller | `timescale.py` | `timescale_enabled` |
| Proactive drift hold | `proactive_drift.py` | `proactive_drift_enabled` |
| RCCDA loss-slope budget gate | `rccda_budget.py` | `rccda_budget_enabled` |
| Deferred adaptation queue (Caravan) | `deferred_adaptation.py` | `deferred_adaptation_enabled` |
| Adaptation safety tracker | `adaptation_safety.py` | `adaptation_safety_enabled` |
| Maintenance `latent_recenter` action | `torch_model.py`, `model_adapter.py` | `maintenance_latent_recenter` |

**Excluded (you are implementing separately):** recurrence gate, delayed-feedback residual head, outstanding-feedback-aware delayed control.

New decision-record fields: `asr_class_concentration`, `drift_detector_score`, `timescale_expert`, `uncertainty_action`, `conformal_alpha`, `adaptation_safety_ok`, `proactive_hold`, `deferred_adaptation`.

Verification suite global check: `adaptation_safety` (priority 10).

## Supplement: May 29, 2026 deep pass

This pass focused on **2024–2026 primary sources** not fully covered above, especially: long-horizon TTA collapse, reset science, delayed-feedback bandits, online conformal under shift, proactive drift, and governance-first sidecars.

### New high-signal papers

| Paper | Venue / year | Why it matters for ARL |
|-------|----------------|------------------------|
| [ADAPT](https://arxiv.org/abs/2508.15568) (backprop-free TTA) | NeurIPS 2025 | Closed-form, no-gradient adaptation with a **historical knowledge bank** — aligns with sidecar latency + bounded updates |
| [Buffer layers for TTA](https://neurips.cc/virtual/2025/poster/115695) | NeurIPS 2025 | **Modular buffer** on frozen backbone — same pattern as ARL sidecar: don’t mutate core weights blindly |
| [TTVD](https://openreview.net/forum?id=4af827e7d0b7bdae6097d44977e87534) | ICLR 2025 | Geometry-guided neighbor/prototype routing under shift — strengthens **regime routing + specialist selection** |
| [When and Where to Reset (ASR)](https://arxiv.org/abs/2603.03796) | 2026 | **Adaptive + selective reset** from collapse risk (class concentration), not fixed cooldown — direct upgrade to `reset` action |
| [ReservoirTTA](https://openreview.net/forum?id=XewZ4rJYKZ) | NeurIPS 2025 | **Reservoir of domain specialists** with online clustering — validates `DelayedHybridBanditSpecialistPolicy` direction |
| [Self-Normalized Resets (SNR)](https://arxiv.org/abs/2410.20098) | NeurIPS 2025 | Hypothesis-test–driven **neuron-level reset** for plasticity — alternative to full `model.reset()` |
| [NCTTA](https://arxiv.org/abs/2512.10421) | 2025 | Feature–classifier alignment under shift — fraud path should track **calibration collapse**, not only shift score |
| [Test-Time Training Undermines Safety Guardrails](https://arxiv.org/html/2605.22984v1) | 2026 | TTT/TTA can **break safety** — mandatory pairing of adaptation with **shadow + budget + rollback** (already in ARL; make it a first-class metric) |
| [Neural Contextual Bandits Under Delayed Feedback](https://arxiv.org/html/2504.12086v1) | 2025 | Delayed NeuralUCB/TS with regret tied to **effective delay mass** — formal basis for pending-queue state |
| [Regret Bounds for Adversarial CMAB with Delayed Feedback](https://openreview.net/forum?id=jlnA0ZRfv1) | NeurIPS 2025 | Adversarial delays + FIFO — supports **stale-label downweight / skip** in delayed controller |
| [Delay-as-payoff contextual linear bandits](https://openreview.net/pdf?id=zL8O4wWQps) | 2025 | Treat delay as part of reward — useful for **KPI design** when labels arrive late |
| [Distribution-informed Online Conformal (COP)](https://arxiv.org/html/2512.07770v1) | 2025 | Tighter sets under shift via CDF-informed updates — top candidate for **uncertainty lane** |
| [DtACI / online conformal under arbitrary shift](https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf) | JMLR 2024 | Locally adaptive coverage — pairs with delayed label reveal stream |
| [CDSeer](https://arxiv.org/html/2410.09190v1) | 2024 | **Model-agnostic** drift detection with few labels — plug-in above `shift_score` |
| [Proceed (proactive drift)](https://arxiv.org/html/2412.08435v4) | 2024–25 | Adapt **before** performance cliff — preemptive `hold` / light recalibrate |
| [RCCDA](https://arxiv.org/pdf/2505.24149) | 2025 | **Lyapunov drift-plus-penalty** update scheduling under resource caps — formalizes `safety_budget` |
| [Arbiter-K](https://arxiv.org/html/2604.18652) | Apr 2026 | Governance-first kernel around probabilistic model — architectural cousin of ARL **sidecar** (policy at sinks, rollback) |

### Revised priority stack (after this pass)

The earlier ranked bets still hold. This pass **raises** three items and **adds** one:

1. **ASR-style reset policy** — replace fixed cooldown reset with collapse-risk trigger + partial/layer-selective reset (ASR, SNR, RDUMB++ line).
2. **Pending-feedback controller** — track outstanding label mass, max age, skip/downweight stale reveals (delayed bandit + Caravan filtering).
3. **Reservoir / recurrence gate** — exchangeability test + specialist reservoir (ReservoirTTA + conformal martingales + existing `StreamingRegimeEncoder`).
4. **Safety-under-adaptation metric** — report guardrail pass rate and invalid adaptation rate alongside dual-metric KPIs (TTT-jailbreak line).

**New (sidecar product):** treat ARL as **governance kernel** (Arbiter-K / flux7-mesh): deterministic policy at mutation sinks, audit graph, rollback — not “smart retrain wrapper.”

### Concrete mapping → current code

| Literature idea | Today in ARL | Suggested tweak |
|-----------------|--------------|-----------------|
| ASR reset | `MultiActionTabularPolicy` fixed cooldown + risk threshold | `collapse_risk` from label concentration + selective reset scope |
| ReservoirTTA | `DelayedHybridBanditSpecialistPolicy` | Online cluster on regime embedding → route batch to specialist |
| Delayed bandit | `reveal_labels` + pending queue | `pending_mass`, `max_staleness`, IPW-style reward update |
| COP / DtACI | static thresholds in monitor | parallel **uncertainty controller** action family |
| Buffer / ADAPT | full adapter mutations | **buffer-only** or closed-form path for hot latency |
| CDSeer | `shift_score` | second detector signal → `should_retrain` vs `should_intervene` |
| RCCDA | `safety_budget` | prove budget as explicit resource constraint, not heuristic cap |
| TTT safety paper | shadow + bounded_auto | add **adaptation safety** eval to verification suite |

### Open gap (still no dominant paper)

No 2024–2026 paper cleanly solves: **tabular fraud + delayed chargebacks + bounded interventions + specialist memory + production sidecar** in one system. That remains the differentiation target.

### Additional sources (this pass)

- [ADAPT (NeurIPS 2025)](https://arxiv.org/abs/2508.15568)
- [Buffer TTA (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/115695)
- [TTVD (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/4af827e7d0b7bdae6097d44977e87534-Paper-Conference.pdf)
- [ASR long-term TTA](https://arxiv.org/abs/2603.03796)
- [ReservoirTTA (NeurIPS 2025)](https://openreview.net/forum?id=XewZ4rJYKZ)
- [SNR resets (NeurIPS 2025)](https://arxiv.org/abs/2410.20098)
- [NCTTA](https://arxiv.org/abs/2512.10421)
- [TTT vs safety guardrails](https://arxiv.org/html/2605.22984v1)
- [Delayed neural contextual bandits](https://arxiv.org/html/2504.12086v1)
- [Adversarial CMAB + delayed feedback (NeurIPS 2025)](https://openreview.net/forum?id=jlnA0ZRfv1)
- [COP online conformal](https://arxiv.org/html/2512.07770v1)
- [CDSeer drift detection](https://arxiv.org/html/2410.09190v1)
- [Proceed proactive adaptation](https://arxiv.org/html/2412.08435v4)
- [RCCDA resource-aware updates](https://arxiv.org/pdf/2505.24149)
- [Arbiter-K governance-first agents](https://arxiv.org/html/2604.18652)

- [Reshaping the Online Data Buffering and Organizing Mechanism for Continual Test-Time Adaptation](https://arxiv.org/abs/2407.09367)
- [DPCore: Dynamic Prompt Coreset for Continual Test-Time Adaptation](https://openreview.net/forum?id=E5MQRICtwq)
- [Adaptive Retention & Correction: Test-Time Training for Continual Learning](https://openreview.net/forum?id=9bLdbp46Q1)
- [Online Feature Updates Improve Online (Generalized) Label Shift Adaptation](https://openreview.net/forum?id=HNH1ykRjXf)
- [Adapting to Online Distribution Shifts in Deep Learning: A Black-Box Approach](https://proceedings.mlr.press/v258/baby25a.html)
- [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://openreview.net/forum?id=p3tSEFMwpG)
- [TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules](https://proceedings.mlr.press/v235/ren24b.html)
- [Test-Time Calibration: A Framework for Personalized Test-Time Adaptation in Real-World Biosignals](https://proceedings.mlr.press/v287/jo25a.html)
- [A Best-of-both-worlds Algorithm for Bandits with Delayed Feedback with Robustness to Excessive Delays](https://openreview.net/forum?id=LDzrQB4X5w)
- [A Conformal Martingales Approach for Recurrent Concept Drift](https://proceedings.mlr.press/v266/eliades25a.html)
- [Online Conformal Prediction via Online Optimization](https://proceedings.mlr.press/v267/areces25a.html)
- [Adapting Prediction Sets to Distribution Shifts Without Labels](https://proceedings.mlr.press/v286/kasa25a.html)
- [Conformal Predictive Systems Under Covariate Shift](https://proceedings.mlr.press/v230/jonkers24a.html)
- [GCAL: Adapting Graph Models to Evolving Domain Shifts](https://proceedings.mlr.press/v267/qiao25a.html)
- [RDUMB++: Drift-Aware Continual Test-Time Adaptation](https://openreview.net/forum?id=lWu9V8z4cf)
- [Caravan: Asynchronous Test-Time Adaptation for Faster Inference](https://openreview.net/forum?id=lyGzOZH4at)
- [Prior-free Tabular Test-time Adaptation](https://openreview.net/forum?id=BgSDPE24pa)
- [Online Test-Time Adaptation in Tabular Data with Minimal High-Certainty Samples](https://openreview.net/forum?id=rmpLcZtJ4l)
- [NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering](https://openreview.net/forum?id=mVlIKLiizr)
- [COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting](https://openreview.net/forum?id=L7Z5wBMPrW)
- [TRUST: Trajectory-guided State-Space Temporal Test-Time Adaptation](https://openreview.net/forum?id=CSking4YcX)
- [IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](https://openreview.net/forum?id=eZO38vANPM)
- [Structural Alignment Improves Graph Test-Time Adaptation](https://openreview.net/forum?id=8Q3qQxmlkJ)
