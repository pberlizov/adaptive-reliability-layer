# ARL Positioning — How It Differs from Alternatives

*How ARL differs from model observability dashboards, online learners, and drift-triggered retrain pipelines.*

---

## The problem ARL solves

Production ML models trained on historical data degrade when the deployment distribution shifts. The standard response is periodic full retrain. ARL is a **bounded reliability controller**: it sits between your inference pipeline and your monitoring layer, applies lightweight targeted adaptations (calibration, batch-norm refresh, label-shift correction), and falls back gracefully when adaptation would hurt more than help. It works with delayed labels — you don't need ground truth immediately.

---

## Comparison table

| | **ARL** | **River** | **Evidently + retrain** | **TENT / TTT** | **ADWIN + retrain** |
|---|---|---|---|---|---|
| **Category** | Bounded reliability controller | Online learner | Drift monitor + full retrain | Test-time adaptation | Drift detector + full retrain |
| **Labels needed at inference?** | No (adapts unsupervised; uses delayed labels when available) | Yes (each sample) | No (monitor only) | No | No (detector); Yes (retrain) |
| **Label delay support** | Yes — core design. Bandit learns from delayed revealed labels | No | Not applicable | No | No |
| **Adaptation granularity** | Lightweight: calibration, BN refresh, label-shift correction, latent recenter | Full model update per sample | None (defers to retrain pipeline) | BN parameters only | None (resets to full retrain) |
| **Rollback / governance** | Yes — snapshot store, audit DB, recommend mode, human approval workflow | No | Depends on MLOps platform | No | No |
| **Multi-action menu** | Yes — 8 actions gated by shift type, risk capital, controller profile | No | No | No | No |
| **Safety budget** | Yes — per-window action cap, downgrade to recommend mode | No | No | No | No |
| **Catastrophic forgetting protection** | Yes — anchor regularization, parameter drift cap, specialist snapshots | No (continual update) | N/A (reset each time) | No (BN only) | N/A (reset each time) |
| **Deployment surface** | FastAPI sidecar, Kafka ingest, Redis policy state | Library only | Library + platform | Library only | Library only |
| **Primary target domain** | Fraud / risk / predictive maintenance | General streaming ML | Any tabular | Computer vision (ImageNet) | General streaming |

---

## Head-to-head on CMAPSS (predictive maintenance)

| Strategy | final_acc delta vs frozen | Notes |
|---|---|---|
| Frozen | 0.0 pp (baseline) | Model degrades ~11–14 pp as engines approach failure |
| BN-only adaptation | −1.2 pp | BN-only adaptation hurts on tabular |
| Naive (always adapt) | −9.1 pp | Unsupervised adaptation without gating |
| **ARL `delayed_bandit`** | **+2.3 pp** | Delayed labels guide targeted adaptation |

---

## Head-to-head on public fraud streams

| Strategy | ULB | IEEE-CIS | PaySim | BAF |
|---|---|---|---|---|
| Frozen | baseline | baseline | baseline | baseline |
| Scheduled retrain | −0.3 pp | +0.1 pp | −0.1 pp | — |
| ADWIN + retrain | −0.2 pp | −0.1 pp | +0.2 pp | — |
| **ARL controller** | **+0.7 pp** | **+0.8 pp** | **+0.7 pp** | — |
| Risk reduction | 7.2% | 8.7% | 7.9% | — |

*(Fraud deltas are small because public fraud datasets show 94–99% accuracy and genuine distributional shift is limited. The risk-reduction metric captures governance cost: fewer false alerts, lower parameter drift, fewer resets.)*

---

## When to use ARL vs. alternatives

**Use ARL when:**
- Labels arrive hours or days after inference (fraud, claims, medical)
- You need an audit trail of every model change (regulated industries)
- You want targeted adaptation (calibration, label-shift) not full retrain
- You need a rollback path if adaptation hurts

**Use River when:**
- Labels arrive immediately with each sample
- You want a fully online model (no source anchor)
- Catastrophic forgetting is acceptable

**Use Evidently + retrain when:**
- You only want monitoring, not automatic adaptation
- You already have a fast retrain pipeline
- Governance is handled elsewhere

**Use TENT / TTT when:**
- Your model is a neural network with BatchNorm layers
- You have image or NLP data (not tabular)
- You don't need governance or rollback

---

## What ARL does not claim

- **Not SOTA accuracy** on any individual benchmark — it's an ops-under-shift story
- **Not a replacement for retraining** — `retrain_recommended` signals when adaptation is insufficient
- **Not a real-time online learner** — it batches adaptation actions with safety budgets

---

*Last updated: 2026-06-05*
