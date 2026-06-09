#!/usr/bin/env python3
"""Run focused production SOTA failure analysis on fraud sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.failure_analysis import (
    run_production_failure_analysis,
    write_production_failure_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARL production failure analysis.")
    parser.add_argument(
        "--config",
        default="configs/production_benchmark_sota_suite.yaml",
        help="Benchmark config to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default="results/production_failure_analysis",
        help="Output directory",
    )
    parser.add_argument("--source", action="append", dest="sources", help="Run only these source ids")
    args = parser.parse_args()

    report = run_production_failure_analysis(
        config_path=ROOT / args.config,
        source_ids=tuple(args.sources) if args.sources else None,
    )
    output = write_production_failure_analysis(report, ROOT / args.output_dir)
    print(output.read_text(encoding="utf-8"))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
