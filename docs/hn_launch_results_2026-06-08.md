# HN launch benchmark results (2026-06-08)

Recorded from `arl-hn-launch --skip-export` on the current repo state using the local exported public-data workspace.

**Artifacts:** [comparison_table.md](../results/hn_launch/comparison_table.md) · [hn_launch_summary.json](../results/hn_launch/hn_launch_summary.json) · [production/suite_report.md](../results/hn_launch/production/suite_report.md) · [discrimination/discrimination_report.md](../results/hn_launch/discrimination/discrimination_report.md)

## Run metadata

| Field | Value |
|-------|-------|
| Generated | 2026-06-08T23:12:54Z |
| Controller (production) | `delayed_hybrid` |
| Controller (discrimination) | `regime_aware_delayed_bandit` |
| Sidecar health | **OK** |
| Config | `configs/hn_launch_production.yaml`, `configs/hn_launch_discrimination.yaml` |

## Production claim suite

**Suite passed:** yes (**3/3 core**, need ≥2) · `require_beat_baselines: true`

Utility = replay accuracy minus operational penalties. **Fair headline: beats scheduled retrain on utility**, not raw accuracy.

### Core sources (claim these)

| Source | Pass | Utility Δ vs frozen | vs scheduled | vs naive | Risk ↓ | Steering |
|--------|------|---------------------|--------------|----------|--------|----------|
| ULB torch | **PASS** | +0.724 | **+0.541** | +0.257 | **7.2%** | **100% correction** |
| IEEE-CIS torch | **PASS** | +0.647 | **+0.508** | +0.043 | **8.7%** | **84% correction** |
| PaySim torch | **PASS** | +0.681 | **+0.515** | -0.002 | **6.0%** | **90% correction** |

All three beat **scheduled retrain** on utility. All three also clear the current `beat_baselines_min_delta` bar versus naive. On these flagship fraud streams, the win comes primarily from **narrow controller steering** (probability / threshold correction), with explicit mutate-the-model actions used rarely.

### Extended sources (honest limits)

| Source | Pass | Why / notes |
|--------|------|-------------|
| Elliptic torch | **FAIL** | Utility +0.275 vs frozen, risk ↓39.5%, beats scheduled (+0.219) — **loses to naive (−0.109)**. Not a core pass. |
| BAF torch | PASS | Utility +0.715 vs frozen/scheduled; **0% risk reduction** (utility-only win). |

## Hard-slice discrimination

**Rankable sources:** 1/5.

| Source | Headroom | Rankable metrics | Notable spread |
|--------|----------|------------------|----------------|
| IEEE hard | yes | 7 | Small but real metric separation |
| ULB hard | limited | 2 | Mostly tied |
| PaySim hard | limited | 3 | Retrain-rate spread more than accuracy spread |
| Elliptic hard | limited | 8 | Largest spread; tradeoffs across acc / bal_acc / recall |
| BAF hard | limited | 1 | All 100% accuracy; effectively no ranking headroom |

**HN framing:** discrimination is still an **ops / utility / tails** story, not a fraud detection leaderboard story.

## What to claim on Show HN

**Safe:**

- 3/3 **core** fraud streams: beats **scheduled retrain** on utility under delayed labels
- **7.2% / 8.7% / 6.0% proxy risk reduction** on ULB / IEEE / PaySim
- The flagship fraud win is primarily **correction-first controller steering**, not frequent explicit mutate-the-model actions
- Sidecar smoke passes in the launch script

**Honest limits:**

- Not SOTA fraud detection accuracy
- Elliptic production is mixed: beats frozen / scheduled, not naive
- BAF is a utility win with zero measured risk delta on this replay
- Discrimination still shows limited ranking headroom on most hard fraud slices
