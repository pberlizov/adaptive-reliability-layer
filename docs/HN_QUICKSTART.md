# Show HN quickstart

Three paths depending on how much time you have. All work from **any directory** after `pip install "adaptive-reliability-layer[torch,serving]>=0.3.4"`.

## Tier 0 — 30 seconds (no data, no torch training)

Synthetic fraud stream through the replay engine:

```bash
arl-offline-replay --synthetic --config default.yaml --output-dir results/toy_replay
cat results/toy_replay/replay_report.txt
```

Shows frozen vs controller vs bandit on a built-in synthetic stream.

## Tier 1 — ~2–5 minutes (toy open dataset, recommended first click)

PaySim synthetic CSV + shortened production + discrimination benchmarks:

```bash
mkdir arl-demo && cd arl-demo
pip install "adaptive-reliability-layer[torch,serving]>=0.3.4"
arl-hn-launch --quick
# or: arl-demo
cat results/hn_launch/comparison_table_quick.md
```

- **No network** for data (PaySim generated locally)
- Sidecar health smoke included
- Same metrics as full run, one source only — uses **toy evidence thresholds** (not the full 3/3 core bar; see Tier 2 for that)

## Tier 2 — ~30–90 minutes (full public fraud suite)

All five open fraud streams (ULB, IEEE-CIS, PaySim, Elliptic, BAF):

```bash
arl-hn-launch
# or skip re-export if data/ already populated:
arl-hn-launch --skip-export
```

First run downloads/builds CSVs (~5–15 min export depending on network). See [FRESH_INSTALL_VERIFY.md](FRESH_INSTALL_VERIFY.md).

## Tier 3 — your CSV (design partner path)

```bash
arl-customer-replay \
  --input your_export.csv \
  --config customer_shadow.yaml \
  --customer acme \
  --output-dir results/acme_shadow
```

Required columns: `timestamp`, `label`, `feature_*`. See [customer_replay.md](customer_replay.md).

## HTTP sidecar (after Tier 1 export)

```bash
arl-serve --config serving_pilot_fraud_torch.yaml --force-shadow
curl -s http://127.0.0.1:8080/v1/health
```

Full curl flow: [sidecar_demo.md](sidecar_demo.md)

## What to cite on Show HN

| Claim | Tier |
|-------|------|
| One command, reproducible table | Tier 1 (`--quick`) or Tier 2 (full) |
| Beats scheduled retrain on utility, 3/3 core | Tier 2 only — [hn_launch_results_2026-06-08.md](hn_launch_results_2026-06-08.md) |
| Not SOTA fraud accuracy | Always honest |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: fastapi` | Use `[torch,serving]` extra, not `[torch]` alone |
| `Config not found` | Upgrade to **0.3.4+** (launch-sync package) |
| Full run slow | Use `--quick` first; `--skip-export` on re-runs |
| IEEE/Elliptic missing | Normal without Kaggle raw; export uses mirrors or synthetic fallback |
