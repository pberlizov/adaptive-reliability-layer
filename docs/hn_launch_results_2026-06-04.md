# HN launch benchmark results (2026-06-04)

> Historical snapshot. For the current canonical HN run, see [hn_launch_results_2026-06-08.md](hn_launch_results_2026-06-08.md).

Recorded from `arl-hn-launch --skip-export` on a full git checkout with **real Elliptic/BAF** mirrors (`pyg_mirror`, `huggingface_mirror`).

**Artifacts:** [comparison_table.md](../results/hn_launch/comparison_table.md) · [hn_launch_summary.json](../results/hn_launch/hn_launch_summary.json) · [production/suite_report.md](../results/hn_launch/production/suite_report.md) · [discrimination/discrimination_report.md](../results/hn_launch/discrimination/discrimination_report.md)

**Run metadata**

| Field | Value |
|-------|-------|
| Generated | 2026-06-04T17:59:18Z |
| Controller (production) | `delayed_hybrid` |
| Controller (discrimination) | `regime_aware_delayed_bandit` |
| Sidecar health | **OK** |
| Config | `configs/hn_launch_production.yaml`, `configs/hn_launch_discrimination.yaml` |
| Real data | Elliptic 46,564 rows, BAF 250,000 rows (manifest verified) |

## Production claim suite

**Suite passed:** yes (**3/3 core**, need ≥2) · `require_beat_baselines: true`

Utility = KPI-weighted score (accuracy minus ops penalties). **Fair headline: beats scheduled retrain on utility**, not raw accuracy.

### Core sources (claim these)

| Source | Pass | Utility Δ vs frozen | vs scheduled | vs naive | Risk ↓ |
|--------|------|---------------------|--------------|----------|--------|
| ULB torch | **PASS** | +0.733 | **+0.550** | +0.183 | 2.7% |
| IEEE-CIS torch | **PASS** | +0.659 | **+0.525** | +0.002 | **13.6%** |
| PaySim torch | **PASS** | +0.637 | **+0.477** | +0.027 | **10.8%** |

All three beat **scheduled retrain** and **naive** on utility. IEEE and PaySim clear the **5% risk-reduction** floor; ULB passes on utility alone (2.7% risk).

### Extended sources (honest limits)

| Source | Pass | Why / notes |
|--------|------|-------------|
| Elliptic torch | **FAIL** | Utility +0.27 vs frozen, risk ↓39.5%, beats scheduled (+0.22) — **loses to naive (−0.11)**. Stretch bar met; not a core pass. |
| BAF torch | PASS | Utility +0.72 vs frozen/scheduled; **0% risk reduction** (utility-only win). |

## Hard-slice discrimination

**Rankable sources:** 0/5 (methods mostly tied on accuracy-heavy fraud tails).

| Source | Headroom | Rankable metrics | Notable spread |
|--------|----------|------------------|----------------|
| IEEE hard | yes | 1 | Mostly flat |
| ULB hard | limited | 2 | Mostly flat |
| PaySim hard | limited | 2 | Retrain rec ↓ with bandit/hybrid |
| Elliptic hard | limited | 8 | Largest spread; acc/bal_acc tradeoffs |
| BAF hard | limited | 1 | All 100% acc; no fraud in slice |

**HN framing:** discrimination shows **retrain-rate** and **Elliptic hard** headroom; do not claim fraud **accuracy** wins on IEEE/BAF hard slices.

## What to claim on Show HN

**Safe:**

- 3/3 **core** fraud streams: beats **scheduled retrain** on utility under delayed labels
- **~11–14% proxy risk reduction** on IEEE + PaySim (ULB +2.7% — cite utility there)
- Real public data path (no Kaggle API) for Elliptic + BAF
- Sidecar smoke passes in launch script

**Honest limits:**

- Not SOTA fraud accuracy; hard-slice bal_acc still weak on IEEE
- Elliptic production: mixed (beats frozen/scheduled, not naive)
- BAF: utility win, zero measured risk delta on this replay
- Discrimination: 0/5 rankable — ops/utility story, not detection leaderboard

## Re-run before post (optional)

```bash
source .venv/bin/activate
pip install -e ".[torch,serving]"
arl-hn-launch --skip-export
```

Compare new `hn_launch_summary.json` to this file. Prior run failed once on Elliptic BatchNorm batch-size-1 (fixed in `torch_model.py`).

## Related docs

- [SHOW_HN_READY.md](SHOW_HN_READY.md) — pre-flight checklist
- [HN_POST_DRAFT.md](HN_POST_DRAFT.md) — post copy (update numbers from here)
- [FRESH_INSTALL_VERIFY.md](FRESH_INSTALL_VERIFY.md) — PyPI vs clone paths
