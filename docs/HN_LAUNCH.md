# Hacker News launch checklist (Thu → Sun/Mon)

Everything a reader needs in **one install + one command**, plus honest numbers.

**Pre-flight:** [SHOW_HN_READY.md](SHOW_HN_READY.md) · **Recorded run:** [hn_launch_results_2026-06-08.md](hn_launch_results_2026-06-08.md)

## Before you post

- [ ] PyPI **0.3.4** published (`arl-hn-launch` on pip, with launch-sync artifacts)
- [ ] Dry-run on clean venv ([FRESH_INSTALL_VERIFY.md](FRESH_INSTALL_VERIFY.md))
- [x] Demo completes: `arl-hn-launch --skip-export` (2026-06-08)
- [x] Comparison table: `results/hn_launch/comparison_table.md`
- [x] Sidecar health passes (included in launch script)
- [x] Real Elliptic/BAF via public mirrors (not synthetic_fallback)
- [ ] README + [HN_POST_DRAFT.md](HN_POST_DRAFT.md) match recorded numbers
- [ ] GitHub URL filled in post draft

## One-command demo

```bash
pip install "adaptive-reliability-layer[torch,serving]"   # 0.3.4+ for launch-synced arl-hn-launch
arl-hn-launch
```

If you want the repo tip instead of PyPI, use a **git clone** + editable install:

```bash
git clone <repo-url> && cd adaptive-reliability-layer
pip install -e ".[torch,serving]"
arl-hn-launch
```

**Fast export-only** (no training, ~1 min):

```bash
arl-hn-launch --export-only
```

**Full benchmarks** (~30–90 min depending on CPU; runs 5 production sources + 5 hard discrimination sources):

```bash
arl-hn-launch
```

Artifacts land in `results/hn_launch/`:

| File | Contents |
|------|----------|
| `comparison_table.md` | **The table for the post** — production utility/risk + hard-slice bal_acc/PR-AUC |
| `hn_launch_summary.json` | Machine-readable pass/fail |
| `production/suite_report.md` | Claim suite detail |
| `discrimination/discrimination_report.md` | Hard-slice detail |

## Real Elliptic + BAF (optional, recommended)

Bundled **synthetic fallbacks** work out of the box. For **real** graph/tabular fraud data **without a Kaggle account**:

```bash
# From a git checkout (scripts not on PyPI 0.3.1)
cd adaptive-reliability-layer
pip install -e ".[torch,serving]"

# ~700 MB Elliptic (PyG mirror) + ~213 MB BAF (HuggingFace mirror)
python3 scripts/fetch_public_fraud_raw.py

python3 scripts/export_open_datasets.py
# Or Elliptic+BAF only: see export_elliptic_baf_fraud_data.py

arl-hn-launch --skip-export
```

Confirm `data/open_datasets_manifest.json` shows `"source": "pyg_mirror"` (elliptic) and `"huggingface_mirror"` (baf). BAF labels: **`fraud_bool`**, not `is_fraud`.

**Kaggle zips** (if you already have them): `scripts/ingest_elliptic_kaggle_zip.py` / `scripts/ingest_baf_kaggle_zip.py`.

## HTTP API (sidecar)

```bash
pip install "adaptive-reliability-layer[torch,serving]"
arl-serve --config configs/serving_pilot_fraud_torch.yaml --force-shadow
curl -s http://127.0.0.1:8080/v1/health
```

Full curl sequence: [sidecar_demo.md](sidecar_demo.md)

## What to claim (honest — from 2026-06-08 run)

**Lead with:**

- Fraud models under **delayed labels** (chargebacks arrive late)
- **Accuracy saturates** on public streams (~95–99%) — we measure **utility** and **proxy risk**
- Beats **scheduled retrain** on utility on **3/3 core** sources (ULB, IEEE-CIS, PaySim)
- **7.2% / 8.7% / 6.0% proxy risk reduction** on the 3 core streams
- The flagship fraud win is primarily **narrow controller steering** (correction / threshold path), with explicit mutations used sparingly

**Do not lead with:**

- SOTA fraud detection accuracy
- Elliptic as a clean win (extended tier: loses to naive)
- Customer ROI without their replay

## Post draft

See [HN_POST_DRAFT.md](HN_POST_DRAFT.md) — title, body, and Show HN comment template.

## Doc map (everything else)

Research notes and pilot docs are indexed in [INDEX.md](INDEX.md). For HN you only need this file, [SHOW_HN_READY.md](SHOW_HN_READY.md), the README, and `comparison_table.md`.

## Timeline

| Day | Task |
|-----|------|
| **Thu–Fri** | Tighten docs, publish 0.3.4, optional re-run |
| **Sat** | Dry-run post, screenshot comparison table, clean venv test |
| **Sun/Mon** | Post — link GitHub + PyPI + comparison table |
