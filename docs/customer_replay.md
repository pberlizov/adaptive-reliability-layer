# Customer historical replay

Productized path for design-partner pilots: replay a customer CSV or JSONL through ARL and emit buyer/operator artifacts without custom scripts.

## Install (PyPI-ready layout)

```bash
pip install "adaptive-reliability-layer[torch,serving]"
# or editable from a clone:
pip install -e ".[torch,serving]"
```

- **Core** (`numpy`, `pandas`, `pyyaml`, `scikit-learn`) — ingest + sklearn adapters.
- **`[torch]`** — fraud/tabular torch pilots and production bundles.
- **`[serving]`** — HTTP sidecar.
- **`[research]`** — optional WILDS benchmarks (not required for customer replay).

Publish: see [pypi_publish.md](pypi_publish.md).

## Input contract

Columns (CSV) or JSONL fields:

| Field | Required | Notes |
|-------|----------|-------|
| `timestamp` | Recommended | Chronological ordering |
| `label` | For KPIs | May be omitted at scoring time when using label delay |
| `feature_*` | Yes | Or any numeric columns except `label` / `event_id` |
| `regime_id` | Optional | Improves delayed bandit |
| `event_id` | Optional | Defaults to row index |

## Shadow-first replay (default)

```bash
arl-customer-replay \
  --input /path/to/customer_export.csv \
  --config configs/customer_shadow.yaml \
  --customer acme_fraud \
  --label-delay-steps 12 \
  --output-dir results/acme_shadow
```

Artifacts:

- `customer_manifest.json` — run metadata
- `replay_schema.md` — column contract
- `dual_metric_report.md` — shadow vs bounded_auto comparison
- `buyer_report.md` — buyer-facing summary (when dual-mode)

## Bounded-auto replay (after shadow sign-off)

```bash
arl-customer-replay \
  --input /path/to/customer_export.csv \
  --config configs/customer_bounded_auto.yaml \
  --customer acme_fraud \
  --output-dir results/acme_bounded
```

## Risk reduction metrics

Reports now compare controller vs **frozen baseline** on:

- Mean risk capital (mitigation decay when controller holds escalations)
- Risk alert rate (frozen uses naive alert→retrain rule)
- Retrain recommendation rate

See `controller_vs_frozen_risk_reduction` in `dual_metric_report.md` / `technical_report.md`.

## Next steps for production claims

1. Agree KPI weights in `kpi:` block with the buyer.
2. Map `label_delay_steps` to business label latency.
3. Wire BYOM model + reference batches for live sidecar (`docs/sidecar_production.md`).
4. Pass `docs/production_evidence_bar.md` on customer replay before external ROI claims.
