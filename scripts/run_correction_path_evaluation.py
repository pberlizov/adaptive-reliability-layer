#!/usr/bin/env python3
"""Evaluate correction-centric parallel paths against the fraud SOTA suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.correction_path import (
    render_correction_path_evaluation,
    run_correction_path_evaluation,
    write_correction_path_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run correction-centric parallel-path evaluation.")
    parser.add_argument(
        "--config",
        default="configs/production_benchmark_sota_suite.yaml",
        help="Production suite config to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default="results/correction_path_evaluation",
        help="Output directory",
    )
    parser.add_argument("--source", action="append", dest="sources", help="Run only these source ids")
    args = parser.parse_args()

    report = run_correction_path_evaluation(
        config_path=ROOT / args.config,
        source_ids=tuple(args.sources) if args.sources else None,
    )
    output = write_correction_path_evaluation(report, ROOT / args.output_dir)
    print(render_correction_path_evaluation(report))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
