# Latency Budget

*Per-batch processing time targets for the ARL sidecar. Last updated: 2026-06-05*

---

## Targets

| Operating mode | P50 target | P99 target | Notes |
|---|---|---|---|
| Shadow | < 5 ms | < 20 ms | Monitor + predict, no adaptation |
| Bounded auto (no action) | < 8 ms | < 30 ms | Monitor + policy decision, no weight update |
| Bounded auto (bn_refresh) | < 15 ms | < 50 ms | One BN forward pass |
| Bounded auto (recalibrate / cool_confidence) | < 10 ms | < 35 ms | Temperature scalar update |
| Bounded auto (adapt) | < 50 ms | < 150 ms | Gradient step on confident samples |
| Bounded auto (reset) | < 5 ms | < 20 ms | State dict copy |

All targets assume CPU-only inference (no GPU), batch size 16–64 rows, 24–30 features.

---

## Current measurements

*Not yet benchmarked systematically. Run `scripts/run_latency_benchmark.py` (not yet written)
to generate measurements and compare to targets above.*

Rough estimates from ad-hoc profiling:

| Component | Estimated time (batch=48, features=30) |
|---|---|
| `TabularShiftMonitor.evaluate` | ~0.3 ms |
| `MartingaleRiskMonitor.update` | ~0.05 ms |
| `TorchTabularAdapterModel.predict_proba` | ~1.5 ms (CPU) |
| `SotaRuntimeExtensions.observe_batch` | ~0.2 ms |
| `GovernanceService.record_intervention` (SQLite) | ~1.0 ms |
| `DelayedCorrectionEngine.enqueue` | ~0.1 ms |
| **Total (shadow, no action)** | **~3–5 ms** |
| `model.adapt` (gradient step, 2 passes) | ~25–40 ms |

---

## Bottlenecks

1. **SQLite audit write** (~1 ms per batch) — the largest non-ML component. For high-frequency
   inference (> 1000 batches/sec), consider async audit writes or batching.

2. **`adapt` action** (~25–40 ms) — the most expensive action. The safety budget caps how often
   it fires; in production typically 0–2 times per 24-step window.

3. **Snapshot serialization** (~5–15 ms) — saving a PyTorch state dict to JSON. Disabled in
   shadow mode by default; fires only after mutations.

---

## CI regression test (TODO)

Add `tests/test_latency.py` that:
1. Runs 100 batches through a shadow-mode layer
2. Asserts median latency < 10 ms
3. Asserts P99 latency < 50 ms

This ensures regressions are caught before release.

---

## Deployment notes

- **CPU-only is the default** and expected deployment target. The PyTorch model uses
  `torch.set_num_threads(1)` to prevent thread-pool contention under concurrent requests.
- **FastAPI + uvicorn** adds ~0.5–1 ms HTTP overhead per request.
- **Redis policy state** adds ~0.5–2 ms per save (network round-trip). Disable for
  latency-sensitive deployments and use file-based persistence instead.
- **Prometheus metrics** collection adds < 0.1 ms per batch when enabled.
