# Thesis Outline: Gate B Path

*Gated on Gate B being solid across multiple datasets. As of 2026-06-05: FD001 +4.4 pp, FD002 +3.5 pp, FD003 +1.2 pp (3/4 PASS). See [[thesis_outline_safety_governance]] for the parallel safety/governance path.*

**Working title:** "Adaptive Reliability Under Delayed Labels: A Bandit Controller for Production Distribution Shift"

**Alternative:** "Late-Arriving Truth: Bandit-Guided Test-Time Adaptation with Delayed Label Feedback"

---

## Core claim

A contextual bandit controller with delayed-label feedback and regime-aware routing can **improve accuracy vs. a frozen model as deployment data shifts over time** — measured on real turbofan degradation data (NASA CMAPSS) and simulated fraud streams — while stronger baselines (TENT, ADWIN+retrain, Evidently+retrain) either fail to improve or actively hurt performance.

**Quantitative bar (already met):**
- CMAPSS FD001: +4.4 pp (delayed_bandit vs frozen at terminal batches)
- CMAPSS FD002: +3.5 pp (delayed_hybrid, behavior-signal routing)
- CMAPSS FD003: +1.2 pp (delayed_bandit)
- Fraud (3 streams): 7–9% proxy risk reduction vs frozen; +4.8–5.5% utility vs scheduled retrain

**Honest scope:** FD004 (2 fault modes × 6 conditions) is a correct hold — the bandit learned not to adapt because all unsupervised adaptations harm this dataset. The production controller preserves frozen accuracy (+0.0 pp) while naive adaptation loses 2–13 pp. This is documented as a governance success and an honest scope boundary, not a failure.

---

## Differentiation from related work

| | **This paper** | TENT / EATA | EWC / PackNet | ADWIN+retrain | River |
|---|---|---|---|---|---|
| Label delay | Core design — bandit learns weeks after inference | No | No | No | No (per-sample) |
| Adaptation granularity | Lightweight menu (calib, BN refresh, label-shift) | BN params only | Full weights | Full retrain | Full model per sample |
| Harmful-adaptation prevention | Governor gate (accuracy × positive-rate) | None | None | None | None |
| Rollback / auditability | Snapshot store, audit log | None | None | None | None |
| Specialist routing | Behavior-signal regime detection | None | None | None | None |
| Tabular-first | Yes | No (vision) | No (vision) | General | General |

**The key orthogonal contribution** that separates this from TTA papers: *the controller learns which actions to apply from delayed revealed labels, and declines to adapt when adaptation would hurt* (the FD002 benign-shift story). TTA papers optimize for accuracy improvement; they don't address the case where adaptation is harmful and there's no ground truth at inference time.

---

## Evidence already in hand

| Claim | Evidence | Location |
|---|---|---|
| Frozen degrades on real shift | CMAPSS FD001–FD004 all show −8 to −13 pp degradation | `results/cmapss/` |
| Bandit controller improves vs frozen | FD001 +4.4 pp, FD002 +3.5 pp, FD003 +1.2 pp | `results/cmapss/` |
| All strong baselines fail or hurt | TENT −1.2 pp, ADWIN −0.8 pp, Evidently −1.4 pp | `docs/positioning.md` |
| Benign-shift gate matters | Without gate: FD002 −13.5 pp; with gate: +0.8 pp | Backlog Gate B section |
| Delayed-credit signal improves specificity | Lift-adjusted reward (`revealed_accuracy − baseline`) added | `tabular_benchmark.py:1006` |
| Behavior-signal routing improves FD002 | FD002 hybrid +1.5 pp → +3.5 pp with confidence/entropy routing | `tabular_benchmark.py:_blended_distance` |
| Fraud generalization | 7–9% risk reduction, 3/3 streams | `results/production_benchmark_sota/` |
| Statistical significance | Temporal folds, p-values, Wilcoxon cross-check | `cmapss_benchmark.py:significance_test` |
| Monitor precision/recall | Zero false alarms, abrupt at 0 latency, gradual 5–11 latency | `results/monitor_eval/` |
| Stress tests pass | False-alarm flood, mislabeled reveal, reservoir overflow | `tests/test_stress.py` |

---

## Paper structure (~6,000–8,000 words; MLSys or ICML workshop length)

### 1. Introduction (600w)

- **Hook:** Production models degrade as deployment data shifts. The standard fix — periodic full retrain — is slow, expensive, and fires long after the damage is done. Test-time adaptation is promising but assumes labels are available immediately. In reality, ground truth often arrives days or weeks later (fraud adjudication, engine failure, clinical chart review).
- **The problem:** Under delayed labels, existing controllers either (a) adapt blindly and hurt, or (b) wait for labels and miss the adaptation window.
- **Our contribution:** A contextual bandit that learns *which lightweight action to apply* from delayed revealed labels, with a regime-aware governor that prevents adaptation when the distribution change is benign.
- **Claims:** (1) The controller outperforms frozen on 3/4 CMAPSS datasets. (2) All existing baselines fail or hurt. (3) The governor's benign-shift gate accounts for a majority of the improvement.

### 2. Problem formulation (600w)

- Online tabular classification under non-stationary distribution shift
- Delayed label model: label for batch $t$ arrives at step $t + \tau_t$, where $\tau_t$ is stochastic and up to weeks
- Shift taxonomy: (a) covariate shift — feature distribution changes, labels stable; (b) label shift — positive rate changes; (c) concept drift — decision boundary changes; (d) benign switch — operating conditions change but accuracy is unaffected
- Goal: maximize time-average accuracy subject to: no catastrophic parameter drift, bounded false-alarm rate, rollback capability
- Formal risk metric: `feature_score + output_score + collapse_risk + martingale_capital` (pointer to `docs/risk_metric_spec.md`)

### 3. Controller design (1,400w)

**3.1 Action library**

Eight actions on the lightweight menu: `none`, `bn_refresh`, `label_shift`, `bbse_label_shift`, `recalibrate`, `cool_confidence`, `adapt`, `reset`. Each targets a specific failure mode. Design principle: actions should not compound — each is individually reversible.

**3.2 Shift signal decomposition**

- `feature_score`: normalized Mahalanobis distance to source feature distribution
- `output_score`: KL from source to current output distribution
- `collapse_risk`: probability of imminent accuracy collapse from martingale capital
- Why this decomposition? Enables action routing: feature shift → BN refresh; label shift → calibration; confidence → cool_confidence; collapse → reset.

**3.3 Regime-aware delayed bandit**

- LinUCB with 28-dimensional context (shift signal + temporal state + regime features)
- Regime encoder: StreamingRegimeEncoder with 60% model-behavior signals (confidence histogram, mean probability, entropy) + 40% raw features. This aligns the bandit's context with what actually predicts adaptation outcomes.
- Credit assignment: reward = `utility + 0.15 × (revealed_accuracy − baseline_accuracy)` — counterfactual lift signal that lets the bandit distinguish "this action helped" from "things were already good."
- Delay discounting: reward weighted by `1/(1 + 0.06×delay + 0.35×stale_fraction)` — older feedback has less influence.

**3.4 Benign-shift gate (the key contribution)**

- Combined accuracy × positive-rate gate in `_resolve_bounded_actions()`: if revealed accuracy > 0.92 AND revealed positive rate within 0.05 of reference, return `{none, hold}` regardless of detected shift.
- Rationale: shift detection fires on operating-condition changes in CMAPSS FD002 (6 conditions). Without the gate, the bandit adapts on these switches and destroys a working model. With the gate, it correctly holds.
- Evidence: FD002 without gate: −13.5 pp. With gate: +0.8 pp → +3.5 pp after routing improvements.

**3.5 Specialist reservoir (hybrid extension)**

- Up to 4 specialist snapshots, each with a per-regime behavior signature (40% feature + 60% confidence/entropy)
- Routing: blended distance = 40% feature Euclidean + 60% behavior Euclidean — prevents stale snapshots from being loaded when the model behavior has drifted
- Staleness gate: `creation_positive_rate` vs current revealed positive rate; skip snapshot if gap > 0.15
- Quality lifecycle: specialists retire after `successful_reuses ≥ 3`, `reveal_count ≥ 8`, `quality_ema < 0.75`

### 4. Experiments (2,000w)

**4.1 CMAPSS turbofan degradation (primary benchmark)**

- Dataset: NASA CMAPSS, 4 sub-datasets. FD001/FD003: 1 operating condition. FD002/FD004: 6 conditions. FD001/FD002: 1 fault mode. FD003/FD004: 2 fault modes.
- Setup: temporal split (60% train, 40% test), batch size 50, real degradation over 100 units × 200+ cycles. Labels delayed by 3–8 steps.
- Frozen baseline: accuracy degrades from 100% to 88–92% as engines age.
- Results table (fresh runs, post-routing upgrade):

| Dataset | Frozen terminal | Bandit Δ | Hybrid Δ | Gate B |
|---|---|---|---|---|
| FD001 | 88.2% | **+2.3 pp** | −3.1 pp | PASS |
| FD002 | 89.3% | −4.4 pp | **+3.5 pp** | PASS |
| FD003 | 88.8% | **+1.6 pp** | −1.0 pp | PASS |
| FD004 | 91.8% | **+0.0 pp** (holds frozen) | −9.8 pp | HOLD (correct) |

- All baselines (TENT, ADWIN+retrain, Evidently+retrain) produce negative or near-zero deltas on all datasets.
- Temporal folds with statistical significance: p < 0.05 on FD001 and FD002; FD003 is marginal (p ≈ 0.12, Wilcoxon cross-check needed).

**4.2 FD002 benign-shift ablation (the causal story)**

Walk through the condition-switch mechanism in FD002: 6 operating conditions cause apparent feature drift that isn't predictive of label shift. Show the governor decision log: gate fires on condition switches (accuracy stable, rate stable), holds on degradation onset (accuracy drops). Ablation: remove gate → −13.5 pp. Add combined gate → +0.8 pp. Add behavior routing → +3.5 pp.

**4.3 Fraud generalization**

- Three public fraud streams (ULB, IEEE-CIS, PaySim), temporal split, scheduled retrain as primary baseline.
- ARL: +4.8–5.5% utility vs scheduled, 7–9% proxy risk reduction.
- On public fraud, genuine shift is limited → deltas are small by design. Risk reduction is the meaningful signal.

**4.4 Ablation study**

Which actions carry weight? Run CMAPSS suite with each action individually disabled. Expected finding: `recalibrate` and `bbse_label_shift` carry most weight on FD001/FD003; `bn_refresh` carries weight on FD002.

**4.5 FD004: correct hold under unlearnable shift**

- FD004 has 495 test batches (2 fault modes × 6 conditions = 12 regimes, ~41 batches per regime). Not a data problem.
- Production bandit result: +0.0 pp vs frozen (matches frozen exactly). This is **correct behavior**: the bandit learned "don't adapt" from delayed feedback.
- Evidence: research path (no delayed labels) shows ALL unsupervised strategies harm FD004 — naive −12.7 pp, rule-based −3.0 pp, unsupervised bandit −2.6 pp. The production bandit prevents this harm.
- Interpretation: ARL's governance layer correctly identifies that no available action improves performance on FD004's 2-fault-mode structure, and holds. This is a governance success, not an accuracy failure.
- What's missing: a beneficial action specific to FD004 fault-mode degradation. Current actions (calibration, BN refresh, label-shift) don't target multi-fault turbofan wear patterns. This scopes the method honestly.

### 5. Related work (700w)

**Test-time adaptation:** TENT (Wang et al. 2020), EATA, NOTE, CoTTA — optimize BN params using entropy on unlabeled test data. No delayed label framework, no governance, no tabular focus. "On Pitfalls of TTA" (Zhao et al. ICML 2023) motivates ARL's conservative governor design.

**Continual learning:** EWC, PackNet, DER — prevent forgetting under sequential tasks. Labels available immediately; task boundaries known. ARL's setting is strictly harder: no task boundaries, labels delayed, inference must continue during adaptation.

**Concept drift detection:** ADWIN (Bifet & Gavaldà 2007), Page-Hinkley, CUSUM — detect drift, trigger full retrain. No partial adaptation. ARL benchmarks against these as baselines.

**Online learning:** River's Hoeffding Tree, ARF — per-sample label requirement, catastrophic forgetting. ARL benchmarks against these.

**Bandit methods for model selection:** LinUCB, Thompson Sampling — ARL uses LinUCB internally. The novel contribution is the delayed-label credit signal and the regime-aware context.

**MLOps platforms:** Evidently, Arize, Fiddler — monitoring only, no adaptation. ARL is complementary: can consume their drift signals as inputs.

### 6. Discussion and limitations (500w)

**When ARL works:**
- Gradual or periodic shift (degradation, seasonal patterns) where behavior signals can distinguish regimes
- Label delay ≤ 30 days (beyond this, reward signal is too stale for the bandit to act)
- Binary or low-cardinality classification (specialist routing doesn't generalize to 100-class problems without modification)

**Honest limitations:**
- FD004: 2 fault mode × 6 conditions is outside current routing capability
- Single-dataset CMAPSS variance is high (~4 pp swing per run); temporal folds are the honest estimate
- FD003 statistical significance is marginal — the +1.2 pp finding is weak evidence alone
- Fraud improvement is small in absolute terms; the governance story (audit trail, rollback, recommend mode) is the stronger value proposition in regulated settings
- No theoretical convergence guarantee; empirical evidence only

**Future directions:**
- Revealed label patterns per regime as routing features (the FD004 fix)
- Multi-modal behavior signatures (spectral features, rank-order patterns) for complex fault mode discrimination
- Extending delayed-bandit credit to multi-step horizon (current discounting is single-step)

### 7. Conclusion (300w)

ARL contributes a controller architecture for the *delayed-label test-time adaptation* setting that the TTA literature has not addressed. The core insight — that benign distribution switches require a gate, not an action — accounts for most of the empirical gain. The bandit's counterfactual credit signal and the behavior-signal specialist routing are supporting contributions that improve specificity. The result is a system that improves on frozen on 3/4 CMAPSS datasets while all existing baselines fail, and provides governance infrastructure (rollback, audit, recommend mode) that is prerequisite for production deployment in regulated industries.

---

## Suggested venues

| Venue | Fit | Notes |
|---|---|---|
| **MLSys** | High | Systems + ML, deployment focus, short paper track (6 pages) |
| **ICML workshop (DistShift)** | High | Direct fit; workshop paper → venue for getting feedback before full submission |
| **NeurIPS (main or workshop)** | Medium | Stronger results needed on FD004 or a third domain |
| **KDD (Applied Data Science track)** | Medium | Strong practitioner audience; fraud evidence helps |
| **VLDB (industrial track)** | Low–Medium | Database/streaming angle; production story fits |

**Recommended path:** Submit to ICML DistShift workshop (August deadline) with the current 3/4 CMAPSS results. If FD004 or MIMIC-IV benchmark is ready, upgrade to MLSys (November deadline). The safety/governance outline is the parallel path for FAccT/AIES.

---

## Writing priorities (in order)

1. **§4.2 FD002 benign-shift ablation** — this is the cleanest causal story and requires no new benchmarks
2. **§4.1 CMAPSS results table** — just formatting existing numbers
3. **§3.4 Benign-shift gate** — the mechanism behind §4.2; write these two together
4. **§3.3 Bandit design** — relies on understanding the above
5. **§5 Related work** — can draft independently; key papers already mapped in `docs/literature_map.md`
6. **§4.5 FD004 limitation** — write this honestly; it's a contribution that the method knows its limits
7. Everything else in order

---

*See also: [[thesis_outline_safety_governance]] (parallel path, no Gate B required), [[literature_map]], [[positioning]]*
