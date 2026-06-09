# Thesis Outline: Safety and Governance Path

*Does not require Gate B. Stand-alone claim.*

**Working title:** "Governed Test-Time Adaptation: Preventing Harmful Autonomous Updates in Production ML"

---

## Core claim

A naive test-time adaptation controller that updates on every detected shift makes the model *more* likely to degrade — specifically, it adapts on benign distribution switches (e.g. operating-condition changes in CMAPSS, time-of-day patterns in fraud) and corrupts a model that was already performing well.  ARL's governor prevents this by:

1. Gating adaptation behind a multi-signal threshold (accuracy × positive-rate)
2. Applying safety budgets (per-window action caps, rollback snapshots)
3. Exposing operating modes (shadow, recommend, bounded_auto) so humans remain in the loop

---

## Evidence already in hand (no Gate B needed)

| Finding | Evidence location |
|---|---|
| Naive controller hurts on all 4 CMAPSS datasets (−7 to −23 pp) | `results/cmapss/` |
| Scheduled retrain baseline also hurts without combined gate (FD002: −13.5 pp → +0.8 pp after gate) | Backlog Gate B section |
| Governor decision log shows gate fires correctly on benign switches | `InterventionGovernor.decision_log` |
| Safety budget caps prevent action floods | `test_stress.py::TestFalseAlarmFlood` |
| Operating mode downgrades (bounded_auto → recommend) work correctly | `tests/test_commercial_runtime.py` |
| Rollback restores pre-intervention state deterministically | `tests/test_serving_phase_b.py` |

---

## Paper structure (4,000–6,000 words)

**1. Introduction** (500w)
- The naive adaptation failure mode: why "adapt when you see drift" is dangerous
- The gap: TTA papers optimize for accuracy improvement; governance papers don't cover adaptation

**2. Problem formulation** (500w)
- Online binary classification under distribution shift with delayed labels
- Benign vs. harmful shift: when to adapt, when to hold
- Governance requirements: auditability, rollback, human-in-the-loop

**3. Controller design** (1,200w)
- Shift signal decomposition: feature_score, output_score, collapse_risk
- Martingale risk capital: sequential test on proxy risk
- Action library: why lightweight actions (calibration, BN refresh) beat full adapt
- Safety budget: per-window caps, downgrade-to-recommend
- Operating modes: shadow, recommend, bounded_auto

**4. The benign-shift gating result** (1,000w)
- CMAPSS FD002 case study: 6 operating conditions, 1 fault mode
- Without gate: bandit −13.5 pp (adapts on condition switches)
- With combined accuracy × positive-rate gate: +0.8 pp
- Contrast with naive, scheduled retrain: both hurt without gate

**5. Safety properties** (800w)
- Formal: bounded parameter drift, reversibility via snapshots
- Empirical: false-alarm flood stress test, mislabeled label handling
- Operating mode progression from shadow to autonomous

**6. Related work** (600w)
- TENT/EATA: BN-only, no governance
- Safe RL: reward shaping, action constraints — analogous but for batch inference
- MLOps (Evidently, Arize): monitoring only, no adaptation
- Continual learning (EWC, PackNet): no delayed labels, no governance

**7. Discussion and limitations** (400w)
- ARL is not SOTA accuracy; it's ops-under-shift
- Single-run CMAPSS variance: temporal folds needed for solid claims
- Governance is application-specific: the framework is general, thresholds are not

---

## Differentiation from Gate B path

| | Gate B path | Safety/governance path |
|---|---|---|
| Primary claim | Controller improves accuracy vs frozen | Controller prevents harmful adaptation |
| Key evidence | CMAPSS +4.4 pp, fraud 7–9% risk reduction | FD002 gate study, stress tests, audit trail |
| Gate requirement | Positive delta on 3/4 CMAPSS datasets | FD002 gating story alone suffices |
| Likely venue | ML systems (MLSys, ICML workshop) | Responsible AI / ML safety (FAccT, AIES) |

---

*Next step: write §4 (FD002 case study) first — it's the cleanest empirical story and doesn't require new benchmarks.*
