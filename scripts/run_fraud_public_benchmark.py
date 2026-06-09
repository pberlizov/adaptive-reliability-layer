#!/usr/bin/env python3
"""Public fraud benchmark: PaySim + IEEE-CIS + German Credit, long horizon + bounded_auto."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.fraud_public_benchmark import run_fraud_public_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public fraud benchmark suite.")
    parser.add_argument("--config", default="configs/fraud_public_benchmark.yaml")
    parser.add_argument("--output-dir", default="results/fraud_public_benchmark")
    parser.add_argument("--stream-cycles", type=int, default=6)
    parser.add_argument("--skip-torch-full", action="store_true")
    args = parser.parse_args()
    run_fraud_public_benchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        stream_cycles=args.stream_cycles,
        skip_torch_full=args.skip_torch_full,
    )


if __name__ == "__main__":
    main()
