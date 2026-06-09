#!/usr/bin/env python3
"""Run the monitor quality evaluation benchmark.

Tests TabularShiftMonitor independently of any controller by running it
against synthetic streams with known ground-truth drift onset times.

Outputs:
  results/monitor_eval/monitor_eval_report.md   — human-readable markdown
  results/monitor_eval/monitor_eval.json        — machine-readable JSON

Usage:
    python3 scripts/run_monitor_eval.py
    python3 scripts/run_monitor_eval.py --n-trials 20 --n-stable 100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.monitor_eval import (
    MonitorEvalConfig,
    render_monitor_eval_report,
    run_monitor_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-stable",
        type=int,
        default=50,
        help="Number of stable batches per stream (default: 50)",
    )
    parser.add_argument(
        "--n-drift",
        type=int,
        default=50,
        help="Number of drift batches per stream (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Samples per batch (default: 200)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=10,
        help="Feature dimensionality (default: 10)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Trials per (shift_type, magnitude) combination (default: 10)",
    )
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0],
        help="Drift magnitudes to sweep (default: 1.5 2.0 3.0)",
    )
    parser.add_argument(
        "--shift-types",
        nargs="+",
        default=["abrupt", "gradual", "recurring"],
        choices=["abrupt", "gradual", "recurring"],
        help="Shift types to evaluate (default: all three)",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=1.1,
        help="TabularShiftMonitor alert threshold (default: 1.1)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/monitor_eval",
        help="Output directory for reports (default: results/monitor_eval)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MonitorEvalConfig(
        n_stable_batches=args.n_stable,
        n_drift_batches=args.n_drift,
        batch_size=args.batch_size,
        n_features=args.n_features,
        drift_magnitudes=list(args.magnitudes),
        n_trials=args.n_trials,
        shift_types=list(args.shift_types),
        alert_threshold=args.alert_threshold,
    )

    print(
        f"Running monitor eval: {len(config.shift_types)} shift types × "
        f"{len(config.drift_magnitudes)} magnitudes × "
        f"{config.n_trials} trials = "
        f"{len(config.shift_types) * len(config.drift_magnitudes) * config.n_trials} total trials"
    )

    results = run_monitor_eval(config)

    report = render_monitor_eval_report(results)
    print(report)

    # Save outputs
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "monitor_eval_report.md"
    md_path.write_text(report, encoding="utf-8")

    json_path = output_dir / "monitor_eval.json"
    json_payload = {
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved markdown report : {md_path}")
    print(f"Saved JSON results    : {json_path}")


if __name__ == "__main__":
    main()
