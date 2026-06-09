# Fresh install verification

End-to-end check for a **new user** with no Kaggle API credentials.

## Pip-only path (0.3.4+, recommended for Show HN)

```bash
rm -rf /tmp/arl-verify && mkdir /tmp/arl-verify && cd /tmp/arl-verify
python3 -m venv .venv && source .venv/bin/activate
pip install "adaptive-reliability-layer[torch,serving]>=0.3.4"
```

| Check | Command | Expected |
|-------|---------|----------|
| Import | `python3 -c "import adaptive_reliability_layer"` | PASS |
| HN CLI | `arl-hn-launch --help` and `arl-demo --help` | PASS |
| Quick demo | `arl-demo` | Creates `./data/fraud/paysim.csv` + `results/hn_launch/comparison_table_quick.md` (~2–5 min) |
| Export | `arl-hn-launch --export-only` | Creates `./data/fraud/*.csv` + manifest |
| Configs | `python3 -c "from adaptive_reliability_layer.workspace import resolve_config_path; print(resolve_config_path('hn_launch_production.yaml'))"` | Path under `bundled_configs/` |
| Sidecar | `arl-serve --config serving_pilot_fraud_torch.yaml --help` | PASS (needs export + torch for full run) |

**Full benchmark:** `arl-hn-launch` (~30–90 min CPU). Writes `results/hn_launch/comparison_table.md`.

**Workspace:** Data and results default to **current working directory**. Override with `ARL_WORKSPACE=/path`.

## Git clone path (developers)

```bash
git clone https://github.com/adaptive-reliability-layer/adaptive-reliability-layer
cd adaptive-reliability-layer
pip install -e ".[torch,serving]"
arl-hn-launch --skip-export   # uses existing data/ if present
```

Repo `configs/` overrides bundled configs when present.

## Public datasets

| Dataset | Export | Real mirror |
|---------|--------|-------------|
| PaySim | Synthetic (instant) | built-in |
| ULB | Zenodo/OpenML download | auto |
| IEEE-CIS | Kaggle raw if present, else synthetic | optional `data/fraud/raw/train_transaction.csv` |
| Elliptic | PyG mirror auto-fetch | `fetch` in export |
| BAF | HuggingFace auto-fetch | `fraud_bool` label |

Manual fetch (clone or pip, from any cwd):

```bash
arl-export-datasets
# or: python3 scripts/fetch_public_fraud_raw.py  # from clone only
```

## PyPI 0.3.1 / 0.3.3 (legacy for Show HN)

`0.3.1` is missing `arl-hn-launch` and bundled configs. `0.3.3` installs the CLI, but its quick-launch artifacts still predate the current HN doc sync. Upgrade to **0.3.4+**.
