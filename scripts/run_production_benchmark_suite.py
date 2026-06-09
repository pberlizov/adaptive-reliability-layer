#!/usr/bin/env python3
"""Run production-grade open-dataset benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.production_benchmark import (
    load_production_benchmark_spec,
    render_production_benchmark_report,
    run_production_benchmark_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARL production evidence benchmark suite.")
    parser.add_argument(
        "--config",
        default="configs/production_benchmark_suite.yaml",
        help="Suite config (runtime + source list + thresholds)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/production_benchmark",
        help="Directory for per-source dual-metric reports and suite summary",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Run only these source ids (repeatable). Default: all in config.",
    )
    args = parser.parse_args()

    runtime_config, spec = load_production_benchmark_spec(ROOT / args.config)
    if args.sources:
        allowed = set(args.sources)
        filtered = tuple(source for source in spec.sources if source.id in allowed)
        if not filtered:
            raise SystemExit(f"No matching sources in config for: {sorted(allowed)}")
        from dataclasses import replace

        spec = replace(spec, sources=filtered)

    report = run_production_benchmark_suite(
        runtime_config=runtime_config,
        spec=spec,
        output_dir=ROOT / args.output_dir,
    )
    print(render_production_benchmark_report(report))
    print(f"\nWrote {ROOT / args.output_dir / 'suite_report.md'}")
    if report.suite_passed:
        raise SystemExit(0)
    # Single-source runs use min_core_sources_passing from config; treat lone pass as success for CI smoke.
    if args.sources and len(report.sources) == 1 and report.sources[0].passed:
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
