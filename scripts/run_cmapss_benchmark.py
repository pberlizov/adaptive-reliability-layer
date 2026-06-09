#!/usr/bin/env python3
"""Run the CMAPSS turbofan degradation benchmark (Gate B).

Measures whether a frozen model's accuracy degrades as turbofan engines
approach failure over time, and compares adaptive controllers.

Usage
-----
  python scripts/run_cmapss_benchmark.py                # FD001, default settings
  python scripts/run_cmapss_benchmark.py --dataset FD003 --batch-size 40
  python scripts/run_cmapss_benchmark.py --all-datasets

Requires data in data/cmapss/.  Run scripts/export_cmapss.py first if the
CSVs are not yet present (it will auto-generate a synthetic fallback if the
NASA download fails).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.cmapss_benchmark import (
    CMAPSSBenchmarkResult,
    CMAPSSConfig,
    DATASET_IDS,
    run_cmapss_benchmark,
    run_cmapss_production_benchmark,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 30, *, fill: str = "#", empty: str = "-") -> str:
    """Render a simple ASCII progress bar."""
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return fill * filled + empty * (width - filled)


def _degradation_curve(
    per_batch: tuple[float, ...],
    *,
    window: int = 5,
    label: str = "",
    width: int = 50,
) -> list[str]:
    """Render a smoothed accuracy curve as ASCII art."""
    if not per_batch:
        return []
    accs = list(per_batch)
    smoothed: list[float] = []
    for i in range(len(accs)):
        lo = max(0, i - window // 2)
        hi = min(len(accs), i + window // 2 + 1)
        smoothed.append(float(sum(accs[lo:hi]) / (hi - lo)))

    lines = [f"  {label} degradation curve (smoothed accuracy per batch):"]
    for i, acc in enumerate(smoothed):
        bar = _bar(acc, width=width)
        lines.append(f"  batch {i + 1:>3}: [{bar}] {acc:.3f}")
    return lines


def _render_summary_table(results: dict[str, CMAPSSBenchmarkResult]) -> list[str]:
    """Render a comparison table of strategy results."""
    lines = [
        "",
        f"{'strategy':<16}{'mean_acc':>10}{'early_acc':>11}{'final_acc':>11}"
        f"{'frozen_delta':>14}{'ctrl_delta':>12}{'risk_reduct':>13}",
        "-" * 87,
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<16}{r.mean_accuracy:>10.3f}{r.early_accuracy:>11.3f}"
            f"{r.final_accuracy:>11.3f}{r.frozen_accuracy_delta:>14.3f}"
            f"{r.controller_accuracy_delta:>12.3f}{r.risk_reduction:>13.3f}"
        )
    return lines


def _render_report(
    config: CMAPSSConfig,
    results: dict[str, CMAPSSBenchmarkResult],
    *,
    include_curve: bool = True,
) -> str:
    """Render a human-readable Markdown report."""
    frozen = results.get("frozen")
    best_controller = max(
        (r for r in results.values() if r.strategy_name != "frozen"),
        key=lambda r: r.mean_accuracy,
        default=None,
    )

    lines: list[str] = [
        "# CMAPSS Turbofan Degradation Benchmark",
        "",
        f"**Dataset:** {config.dataset_id}  "
        f"**Batch size:** {config.batch_size}  "
        f"**Train fraction:** {config.train_fraction}",
        "",
    ]

    if frozen is not None:
        degradation_pct = frozen.frozen_accuracy_delta * 100.0
        lines += [
            "## Key Finding",
            "",
            f"Frozen model accuracy: **{frozen.early_accuracy:.1%}** "
            f"(early) → **{frozen.final_accuracy:.1%}** (late/terminal batches)",
            f"Degradation: **{degradation_pct:+.1f} pp** as fleet ages toward failure",
            "",
        ]
        if best_controller is not None:
            lift_pct = (best_controller.final_accuracy - frozen.final_accuracy) * 100.0
            lines += [
                f"Best controller ({best_controller.strategy_name}): "
                f"**{best_controller.final_accuracy:.1%}** at terminal "
                f"(+{lift_pct:+.1f} pp vs frozen)",
                "",
            ]

    lines += ["## Strategy Comparison", ""]
    lines += _render_summary_table(results)
    lines += [
        "",
        "**Columns:** mean_acc = overall accuracy; early_acc = first-10-batch average;",
        "final_acc = last-10-batch average; frozen_delta = early−final (degradation);",
        "ctrl_delta = controller final − frozen final; risk_reduct = relative lift.",
        "",
    ]

    if include_curve and frozen is not None and frozen.per_batch_accuracies:
        lines.append("## Frozen Model Degradation Curve")
        lines.append("")
        lines += _degradation_curve(
            frozen.per_batch_accuracies,
            label="frozen",
            window=3,
        )
        lines.append("")

    if best_controller is not None and best_controller.per_batch_accuracies:
        lines.append(f"## Best Controller ({best_controller.strategy_name}) Curve")
        lines.append("")
        lines += _degradation_curve(
            best_controller.per_batch_accuracies,
            label=best_controller.strategy_name,
            window=3,
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ensure_data(config: CMAPSSConfig) -> None:
    """Run export_cmapss.py if the CSV is not already present."""
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    csv = data_dir / f"cmapss_{config.dataset_id}.csv"
    if not csv.exists():
        print(f"Data not found at {csv} — running export_cmapss.py…")
        export_script = ROOT / "scripts" / "export_cmapss.py"
        subprocess.run([sys.executable, str(export_script)], check=True, cwd=ROOT)


def _run_single(
    config: CMAPSSConfig,
    *,
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, CMAPSSBenchmarkResult]:
    """Run and report a single dataset benchmark."""
    _ensure_data(config)

    print(f"\n{'=' * 70}")
    print(f"CMAPSS Benchmark: {config.dataset_id}")
    print(f"  batch_size={config.batch_size}  train_fraction={config.train_fraction}")
    print(f"{'=' * 70}\n")

    # --- Research path (unsupervised) ----------------------------------------
    results = run_cmapss_benchmark(config)

    frozen = results.get("frozen")
    if frozen is not None:
        degradation_pct = frozen.frozen_accuracy_delta * 100.0
        print(
            f"  Frozen model degraded from {frozen.early_accuracy:.1%} to "
            f"{frozen.final_accuracy:.1%} accuracy ({degradation_pct:+.1f} pp)"
        )

    # --- Production path (delayed label reveals) ----------------------------
    print("\n  Running production path (ReliabilityLayer + reveal_labels)…")
    prod_results = run_cmapss_production_benchmark(config)
    prod_frozen = prod_results.get("frozen")

    print("\n  [Research path — unsupervised]")
    if verbose:
        for line in _render_summary_table(results):
            print(line)

    print("\n  [Production path — delayed label reveals]")
    if verbose:
        for line in _render_summary_table(prod_results):
            print(line)

    if prod_frozen is not None:
        best_prod = max(
            (r for r in prod_results.values() if r.strategy_name != "frozen"),
            key=lambda r: r.final_accuracy,
            default=None,
        )
        if best_prod is not None:
            lift = (best_prod.final_accuracy - prod_frozen.final_accuracy) * 100.0
            gate_b_status = "PASS" if lift > 0.5 else "HOLD" if lift >= 0.0 else "FAIL"
            print(
                f"\n  *** Gate B [{gate_b_status}]: "
                f"best controller ({best_prod.strategy_name}) "
                f"{'+' if lift >= 0 else ''}{lift:.1f} pp vs frozen at terminal ***"
            )

    # Print comparison table
    if verbose:
        for line in _render_summary_table(results):
            print(line)
        print()

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    def _serialise(d: dict) -> dict:
        return {
            name: {k: v if not isinstance(v, tuple) else list(v) for k, v in asdict(r).items()}
            for name, r in d.items()
        }

    # Research path outputs
    json_path = output_dir / f"cmapss_{config.dataset_id}_results.json"
    json_path.write_text(json.dumps(_serialise(results), indent=2), encoding="utf-8")
    report_text = _render_report(config, results)
    md_path = output_dir / f"cmapss_{config.dataset_id}_report.md"
    md_path.write_text(report_text, encoding="utf-8")

    # Production path outputs
    prod_json_path = output_dir / f"cmapss_{config.dataset_id}_production_results.json"
    prod_json_path.write_text(json.dumps(_serialise(prod_results), indent=2), encoding="utf-8")
    prod_report_text = _render_report(config, prod_results)
    prod_md_path = output_dir / f"cmapss_{config.dataset_id}_production_report.md"
    prod_md_path.write_text(prod_report_text, encoding="utf-8")

    print(f"\n  Saved research JSON  → {json_path}")
    print(f"  Saved production JSON → {prod_json_path}")
    print(f"  Saved production MD  → {prod_md_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CMAPSS turbofan degradation benchmark."
    )
    parser.add_argument(
        "--dataset",
        default="FD001",
        choices=DATASET_IDS,
        help="CMAPSS sub-dataset to benchmark (default: FD001)",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run benchmark on all 4 CMAPSS sub-datasets",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for replay (default: 50)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="Fraction of time-cycles used for training (default: 0.6)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cmapss",
        help="Directory for output files (default: results/cmapss)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    datasets = DATASET_IDS if args.all_datasets else [args.dataset]

    all_results: dict[str, dict[str, CMAPSSBenchmarkResult]] = {}
    for dataset_id in datasets:
        config = CMAPSSConfig(
            dataset_id=dataset_id,
            batch_size=args.batch_size,
            train_fraction=args.train_fraction,
        )
        results = _run_single(config, output_dir=output_dir)
        all_results[dataset_id] = results

    # If multiple datasets, write a combined summary
    if len(datasets) > 1:
        combined_lines = [
            "# CMAPSS Benchmark — Combined Summary",
            "",
            f"Batch size: {args.batch_size}  |  "
            f"Train fraction: {args.train_fraction}",
            "",
            f"{'dataset':<10}{'strategy':<16}{'mean_acc':>10}{'early_acc':>11}"
            f"{'final_acc':>11}{'frozen_delta':>14}",
            "-" * 72,
        ]
        for did, results in all_results.items():
            for name, r in results.items():
                combined_lines.append(
                    f"{did:<10}{name:<16}{r.mean_accuracy:>10.3f}"
                    f"{r.early_accuracy:>11.3f}{r.final_accuracy:>11.3f}"
                    f"{r.frozen_accuracy_delta:>14.3f}"
                )
            combined_lines.append("")

        combined_json = {
            did: {
                name: {
                    k: v if not isinstance(v, tuple) else list(v)
                    for k, v in asdict(r).items()
                }
                for name, r in results.items()
            }
            for did, results in all_results.items()
        }

        (output_dir / "cmapss_results.json").write_text(
            json.dumps(combined_json, indent=2), encoding="utf-8"
        )
        (output_dir / "cmapss_report.md").write_text(
            "\n".join(combined_lines), encoding="utf-8"
        )
        print(f"\n  Combined report → {output_dir / 'cmapss_report.md'}")
        print(f"  Combined JSON   → {output_dir / 'cmapss_results.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
