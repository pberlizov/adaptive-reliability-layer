#!/usr/bin/env python3
"""PaySim sklearn fraud pilot (faster): dual-metric + delayed labels + policy state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.pilot import DEFAULT_PILOT, run_pilot_case_study
from adaptive_reliability_layer.replay.real_data import load_paysim_fraud_bundle
from adaptive_reliability_layer.runtime.config import load_runtime_config


def main() -> None:
    config = load_runtime_config("configs/pilot_fraud_sklearn.yaml")
    stream_cycles = 2
    bundle = load_paysim_fraud_bundle(
        steps=config.replay.max_steps or 48,
        batch_size=config.replay.batch_size,
        stream_cycles=stream_cycles,
    )
    pilot = replace(
        DEFAULT_PILOT,
        name="paysim_sklearn_dual_pilot",
        description="PaySim sklearn stream with regime-aware delayed bandit (fast milestone path).",
        use_synthetic_stream=False,
        label_delay_steps=config.replay.label_delay_steps,
        operating_mode="bounded_auto",
        strategies=("frozen", "regime_aware_delayed_bandit"),
        controller_name="regime_aware_delayed_bandit",
        dual_mode=True,
    )
    summary = run_pilot_case_study(
        pilot,
        runtime_config=config,
        output_dir="results/pilot_sklearn",
        layer_builder=bundle.build_layer,
        stream=bundle.stream,
    )
    print("Pilot complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
