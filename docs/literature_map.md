# Literature Map: Safe Continual Test-Time Adaptation Under Distribution Shift

## Working Problem Frame

We are not just interested in `monitor + recommend`, which already exists in many commercial ML observability products. The research goal is stronger:

- detect meaningful shift early
- decide whether adaptation is safe
- adapt online without labels when possible
- avoid catastrophic forgetting and collapse over long streams
- quantify uncertainty after adaptation
- handle graph-structured or relational shift, not only flat feature drift

The strongest research framing is:

**A safe controller for continual test-time adaptation under structured distribution shift.**

## High-Level Takeaways

1. Test-time adaptation is already a real and active research area.
2. The field has good methods for narrow settings, especially vision benchmarks.
3. Long-horizon continual adaptation remains unstable and failure-prone.
4. Memory, restoration, replay, and reset logic are central.
5. Graph-specific adaptation under shift is much less mature than standard image-based TTA.
6. Uncertainty under online adaptation is promising but not solved; formal guarantees degrade once exchangeability breaks or the base model changes over time.

## Core Literature

### 1. Foundational Test-Time Adaptation

These papers establish the modern TTA setup and baseline methods.

- **Tent: Fully Test-Time Adaptation by Entropy Minimization**  
  Wang et al., 2020  
  Link: https://arxiv.org/abs/2006.10726  
  Why it matters: canonical fully test-time adaptation method; updates normalization statistics and affine parameters using entropy minimization on unlabeled test batches.

- **Test-Time Adaptation to Distribution Shift by Confidence Maximization and Input Transformation**  
  Mummadi et al., 2021  
  Link: https://arxiv.org/abs/2106.14999  
  Why it matters: improves on raw entropy minimization with a more stable objective and an input transformation module.

- **MEMO: Test Time Robustness via Adaptation and Augmentation**  
  Zhang et al., NeurIPS 2022  
  Link: https://proceedings.neurips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html  
  Why it matters: uses augmentation consistency at test time; broadens the adaptation toolkit beyond simple entropy minimization.

- **MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption**  
  Bartler et al., AISTATS 2022  
  Link: https://proceedings.mlr.press/v151/bartler22a.html  
  Why it matters: introduces a meta-learning angle for test-time training.

- **Meta-TTT: A Meta-learning Minimax Framework For Test-Time Training**  
  Tao et al., 2024  
  Link: https://arxiv.org/abs/2410.01709  
  Why it matters: a more recent example of meta-learning for robust test-time training on BN layers.

### 2. Continual Test-Time Adaptation and Failure Modes

These are essential because they show where adaptation methods break in realistic streams.

- **NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation**  
  Gong et al., NeurIPS 2022  
  Link: https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae6c7dbd9429b3a75c41b5fb47e57c9e-Abstract-Conference.html  
  Why it matters: focuses on non-i.i.d. streams and temporal correlation, which are much closer to deployment reality.

- **On Pitfalls of Test-Time Adaptation**  
  Zhao et al., ICML 2023  
  Link: https://proceedings.mlr.press/v202/zhao23d.html  
  Why it matters: strong reality check on instability, collapse modes, and evaluation weaknesses in TTA.

- **RDumb: A simple approach that questions our progress in continual test-time adaptation**  
  Yuan et al., NeurIPS 2023  
  Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d640f377893fc5f22b5610e175ef7c3-Abstract-Conference.html  
  Why it matters: shows many CTTA methods eventually collapse and can underperform a non-adapting model.

- **SoTTA: Robust Test-Time Adaptation on Noisy Data Streams**  
  Liang et al., NeurIPS 2023  
  Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html  
  Why it matters: addresses noisy streams and adaptation robustness.

- **Continual Test-Time Adaptation: A Comprehensive Survey**  
  OpenReview/TMLR submission, 2026  
  Link: https://openreview.net/forum?id=mM3r03Xw1V  
  Why it matters: good map of method families, problem settings, and open problems.

### 3. Memory, Replay, Restoration, and Controlled Adaptation

This cluster is closest to the safety layer we are likely to need.

- **Effective Restoration of Source Knowledge in Continual Test Time Adaptation**  
  2023  
  Link: https://arxiv.org/abs/2311.04991  
  Why it matters: emphasizes domain-change detection and resets/restoration to preserve source knowledge.

- **Leveraging Proxy of Training Data for Test-Time Adaptation**  
  Kang et al., ICML 2023  
  Link: https://proceedings.mlr.press/v202/kang23a.html  
  Why it matters: uses proxy information about source data to stabilize adaptation without direct source access.

- **ReservoirTTA: Prolonged Test-time Adaptation for Evolving and Recurring Domains**  
  OpenReview, 2025  
  Link: https://openreview.net/forum?id=XewZ4rJYKZ  
  Why it matters: explicitly handles recurring domains and prolonged adaptation.

- **DPCore: Dynamic Prompt Coreset for Continual Test-Time Adaptation**  
  Zhang et al., ICML 2025  
  Link: https://proceedings.mlr.press/v267/zhang25bf.html  
  Why it matters: memory/core-set ideas remain important even in newer parameter-efficient adaptation regimes.

- **Source-Free Controlled Adaptation of Teachers for Continual Test-Time Adaptation**  
  OpenReview, 2025  
  Link: https://openreview.net/forum?id=nymWIrCIhF  
  Why it matters: controlled adaptation is very aligned with a safety-gated adaptation engine.

### 4. Graph Shift, OOD, and Graph Test-Time Adaptation

This is the most promising differentiation layer for the project.

- **Incremental Unsupervised Domain Adaptation on Evolving Graphs**  
  Chung and Ghosh, CoLLAs 2023  
  Link: https://proceedings.mlr.press/v232/chung23a.html  
  Why it matters: directly motivated by evolving graph settings such as fraud detection under changing patterns.

- **Pairwise Alignment Improves Graph Domain Adaptation**  
  Liu et al., ICML 2024  
  Link: https://proceedings.mlr.press/v235/liu24ci.html  
  Why it matters: addresses graph structure shift more explicitly than generic domain adaptation methods.

- **GCAL: Adapting Graph Models to Evolving Domain Shifts**  
  Qiao et al., ICML 2025  
  Link: https://proceedings.mlr.press/v267/qiao25a.html  
  Why it matters: one of the clearest recent references for continual graph adaptation with memory.

- **Test-time Adaptation on Graphs via Adaptive Subgraph-based Selection and Regularized Prototypes**  
  OpenReview, 2024  
  Link: https://openreview.net/forum?id=lC40m2jjUO  
  Why it matters: graph TTA remains underexplored; this is directly relevant.

- **Structural Alignment Improves Graph Test-Time Adaptation**  
  OpenReview, 2026 cycle  
  Link: https://openreview.net/forum?id=8Q3qQxmlkJ  
  Why it matters: recent work focused specifically on structure shift in graph TTA.

- **Out-of-Distribution Detection on Graphs: A Survey**  
  2025  
  Link: https://arxiv.org/abs/2502.08105  
  Why it matters: useful background for graph OOD taxonomies, metrics, and evaluation setups.

### 5. Uncertainty and Conformal Methods Under Shift

This area matters if the system must say not only "I adapted" but also "here is how much to trust the adapted output."

- **Adaptive Conformal Inference Under Distribution Shift**  
  Gibbs and Candès, 2021  
  Link: https://arxiv.org/abs/2106.00170  
  Why it matters: foundational online conformal reference for nonstationary settings.

- **Test-time Recalibration of Conformal Predictors Under Distribution Shift Based on Unlabeled Examples**  
  2022  
  Link: https://arxiv.org/abs/2210.04166  
  Why it matters: directly relevant if we want uncertainty adaptation with limited or delayed labels.

- **Improved Online Conformal Prediction via Strongly Adaptive Online Learning**  
  2023  
  Link: https://arxiv.org/abs/2302.07869  
  Why it matters: useful for faster online recalibration under drift.

- **Adapting Conformal Prediction to Distribution Shifts Without Labels**  
  2024  
  Link: https://arxiv.org/abs/2406.01416  
  Why it matters: especially relevant to unlabeled target streams.

- **Adapting Prediction Sets to Distribution Shifts Without Labels**  
  Kasa et al., UAI 2025  
  Link: https://proceedings.mlr.press/v286/kasa25a.html  
  Why it matters: strong modern reference for label-free adaptation of prediction sets.

- **Conformal Prediction Under Generalized Covariate Shift with Posterior Drift**  
  Wang and Qiao, AISTATS 2025  
  Link: https://proceedings.mlr.press/v258/wang25l.html  
  Why it matters: useful when covariate shift is mixed with posterior drift, which is closer to production reality.

- **Online Conformal Inference with Retrospective Adjustment for Faster Adaptation to Distribution Shift**  
  Jun and Ohn, 2025  
  Link: https://arxiv.org/abs/2511.04275  
  Why it matters: recent proposal for faster recalibration in online settings.

## What Seems Saturated vs Open

### More Saturated

- TTA on image corruption benchmarks
- entropy-minimization variants
- parameter-efficient adaptation tricks on standard vision datasets

### More Open

- adaptation controllers that decide when to adapt, when to reset, and when to abstain
- long-horizon continual adaptation under recurring and shifting regimes
- graph-structured test-time adaptation
- principled replay/coreset logic tied to safety rather than just accuracy
- uncertainty calibration for an online-adapting model
- evaluation protocols that resemble deployment rather than benchmark domain shifts

## Likely Research Gap

The strongest gap appears to be the integration problem:

- drift detection exists
- adaptation methods exist
- graph OOD methods exist
- online conformal methods exist

But there is no clean, mature, widely accepted framework that combines:

1. structured shift diagnosis
2. safe continual adaptation
3. source-knowledge preservation
4. graph-aware monitoring
5. uncertainty recalibration
6. rollback or abstention when adaptation is unsafe

That integration gap is likely the most interesting place to contribute.

## Candidate Research Thesis

Possible thesis statement:

**We propose a safe continual test-time adaptation controller that combines shift diagnosis, bounded parameter updates, memory-based retention, and uncertainty-aware rollback for models deployed under structured nonstationary distributions.**

Possible novelty levers:

- graph-aware shift features in the controller state
- explicit gating policy for `adapt vs recalibrate vs abstain vs reset`
- replay or coreset anchoring during adaptation
- uncertainty-aware rollback after harmful updates
- evaluation on recurring and evolving domains rather than one-shot corruption benchmarks

## Prototype Direction Suggested by the Literature

A realistic first prototype should avoid trying to solve every open problem at once.

### Recommended V1

- base model: standard classifier or GNN
- shift monitor: latent MMD plus rolling change detection
- adaptation: BN-stat updates plus TENT-style adaptation
- safety: anchor replay or source proxy, plus reset logic
- uncertainty: adaptive conformal or threshold recalibration
- evaluation: continual stream with gradual, abrupt, recurring, and structural shifts

### Recommended V2

- graph topology drift features
- controller that chooses among multiple adaptation actions
- memory buffer with coreset selection
- rollback after post-update degradation tests
- shadow-mode vs live-mode adaptation comparison

## Key Risks the Prototype Should Measure

1. Detection quality: does the monitor detect harmful shift early enough to matter?
2. Actionability: does detecting shift actually help choose the right intervention?
3. Safety: how often does adaptation make performance worse?
4. Retention: how much source-domain competence is lost over time?
5. Recovery: can the system recover after a bad adaptation cycle?
6. Calibration: do uncertainty estimates remain informative after repeated updates?
7. Graph value-add: do graph-specific signals detect failure earlier than flat feature statistics?

## Shortlist for Immediate Deep Reading

If we only read a dozen papers first, the best starting set is:

- Tent: Fully Test-Time Adaptation by Entropy Minimization
- MEMO: Test Time Robustness via Adaptation and Augmentation
- NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation
- On Pitfalls of Test-Time Adaptation
- RDumb
- Leveraging Proxy of Training Data for Test-Time Adaptation
- Effective Restoration of Source Knowledge in Continual Test Time Adaptation
- Incremental Unsupervised Domain Adaptation on Evolving Graphs
- GCAL: Adapting Graph Models to Evolving Domain Shifts
- Out-of-Distribution Detection on Graphs: A Survey
- Adaptive Conformal Inference Under Distribution Shift
- Adapting Prediction Sets to Distribution Shifts Without Labels

## Next Recommended Documents

After this literature map, the next useful artifacts would be:

1. a taxonomy table with columns for assumptions, update target, memory use, graph support, and failure modes
2. an architecture note for our proposed system
3. an evaluation plan with benchmark candidates and metrics
4. a prototype spec with a concrete repo layout

---

## ARL-specific positioning within this literature (added 2026-06-05)

### Where ARL sits

ARL is a **runtime safety layer** for TTA, not a new TTA method. It addresses the
failure modes identified in the literature (RDumb, NOTE, CoTTA) by building an
explicit gating controller above the adaptation primitives.

| Literature finding | ARL response |
|---|---|
| Entropy minimization collapses on long streams (RDumb) | `max_parameter_drift` cap; anchor regularization; `reset` action with snapshot restore |
| TTA without labels is unstable (Zhao et al. pitfalls) | Delayed-label feedback via `DelayedCorrectionEngine`; adaptation only after enough reveals |
| Source knowledge degrades over time (CoTTA) | Specialist warm-start from source anchor; `creation_positive_rate` staleness gate |
| Temporal correlation breaks i.i.d. assumptions (NOTE) | Regime encoder tracks recurrence; bandit learns per-regime policy |
| Uncertainty after adaptation is unreliable | Online conformal controller (`OnlineConformalController`) tracks empirical coverage |

### Key papers for the Gate B story

- **TENT** — ARL includes it as a strategy; empirically it hurts on tabular CMAPSS (−1.2 pp)
- **EWC** — ARL's `anchor_strength` parameter plays the same role without modifying training
- **Adaptive conformal inference** (Gibbs & Candès) — theoretical basis for `OnlineConformalController`
- **ADWIN** (Bifet & Gavalda) — included as a comparison baseline; ARL's monitor outperforms it
  on the synthetic fraud stream (zero false alarms, 0-latency abrupt detection vs ADWIN's ~5-batch lag)

### What ARL does NOT address (honest gaps)

- **No formal convergence guarantees** — the bandit converges in theory but sample complexity under
  delayed feedback with distribution shift is uncharacterized
- **No adversarial robustness** — inputs crafted to trigger adaptation could corrupt the model
- **Tabular-only model** — `TorchTabularAdapterModel` uses a small MLP; vision/NLP require new adapters
- **Single-model** — no cross-model knowledge sharing; each session is independent
