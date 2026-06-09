#!/usr/bin/env python3
"""Run discrimination-focused benchmarks with imbalance-aware metrics and harder temporal slices."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_reliability_layer.replay.discrimination_benchmark import (
    run_discrimination_benchmark,
    write_discrimination_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/discrimination_benchmark_suite.yaml",
        help="Discrimination benchmark YAML config",
    )
    parser.add_argument(
        "--output",
        default="results/discrimination_benchmark",
        help="Directory for markdown/json reports",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Optional source id filter (repeatable)",
    )
    args = parser.parse_args()

    report = run_discrimination_benchmark(
        config_path=args.config,
        source_ids=tuple(args.sources) if args.sources else None,
    )
    output = write_discrimination_artifacts(report, output_dir=args.output)
    print(f"Discrimination benchmark complete: {output / 'discrimination_report.md'}")
    print(f"Rankable sources: {report.rankable_sources}/{len(report.sources)}")


if __name__ == "__main__":
    main()
