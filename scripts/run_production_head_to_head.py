#!/usr/bin/env python3
"""Run delayed_hybrid vs regime_aware_delayed_bandit on identical SOTA torch config."""

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
    render_head_to_head_report,
    render_production_benchmark_report,
    run_production_benchmark_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controller head-to-head production benchmark.")
    parser.add_argument(
        "--config",
        default="configs/production_benchmark_head_to_head.yaml",
        help="Head-to-head suite config",
    )
    parser.add_argument(
        "--output-dir",
        default="results/production_benchmark_head_to_head",
        help="Output directory",
    )
    parser.add_argument("--source", action="append", dest="sources", help="Run only these source ids")
    args = parser.parse_args()

    runtime_config, spec = load_production_benchmark_spec(ROOT / args.config)
    if not spec.comparison_controller:
        raise SystemExit("Config must set production_benchmark.comparison_controller")

    if args.sources:
        from dataclasses import replace

        allowed = set(args.sources)
        filtered = tuple(source for source in spec.sources if source.id in allowed)
        if not filtered:
            raise SystemExit(f"No matching sources in config for: {sorted(allowed)}")
        spec = replace(spec, sources=filtered)

    output = ROOT / args.output_dir
    report = run_production_benchmark_suite(
        runtime_config=runtime_config,
        spec=spec,
        output_dir=output,
    )
    print(render_head_to_head_report(report))
    print()
    print(render_production_benchmark_report(report))
    print(f"\nWrote {output / 'head_to_head_report.md'}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
