#!/usr/bin/env python3
"""One-command Hacker News launch demo: export public data, run benchmarks, write comparison table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.hn_launch import (
    export_public_datasets,
    render_hn_comparison_table,
    run_hn_discrimination_benchmark,
    run_hn_production_benchmark,
    verify_sidecar_health,
    write_hn_launch_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HN launch pack: export open fraud data, run public benchmarks, emit comparison table.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hn_launch",
        help="Artifacts directory",
    )
    parser.add_argument(
        "--production-config",
        default="configs/hn_launch_production.yaml",
        help="Production claim suite config",
    )
    parser.add_argument(
        "--discrimination-config",
        default="configs/hn_launch_discrimination.yaml",
        help="Hard-slice discrimination config",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip open dataset export")
    parser.add_argument("--skip-production", action="store_true", help="Skip production benchmark")
    parser.add_argument("--skip-discrimination", action="store_true", help="Skip discrimination benchmark")
    parser.add_argument("--skip-sidecar", action="store_true", help="Skip HTTP sidecar health check")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export public datasets (fast sanity check)",
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    manifest_path = None

    if not args.skip_export:
        print("==> Exporting public fraud datasets (ULB, IEEE, PaySim, Elliptic, BAF)...")
        manifest_path = export_public_datasets(root=ROOT)
        print(f"    Manifest: {manifest_path}")
    else:
        candidate = ROOT / "data" / "open_datasets_manifest.json"
        if candidate.exists():
            manifest_path = candidate

    if args.export_only:
        print("Export-only complete.")
        return

    production_report = None
    discrimination_report = None

    if not args.skip_production:
        print("==> Running production claim benchmark (5 public fraud sources)...")
        production_report = run_hn_production_benchmark(
            config_path=ROOT / args.production_config,
            output_dir=output_dir / "production",
        )
        print(
            f"    Suite passed: {production_report.suite_passed} "
            f"({production_report.core_sources_passing} core sources)"
        )

    if not args.skip_discrimination:
        print("==> Running hard-slice discrimination benchmark...")
        discrimination_report = run_hn_discrimination_benchmark(
            config_path=ROOT / args.discrimination_config,
            output_dir=output_dir / "discrimination",
        )
        print(
            f"    Rankable sources: {discrimination_report.rankable_sources}/"
            f"{len(discrimination_report.sources)}"
        )

    sidecar_ok = None
    if not args.skip_sidecar:
        print("==> Verifying HTTP sidecar /v1/health...")
        sidecar_ok = verify_sidecar_health()
        print(f"    Sidecar OK: {sidecar_ok}")

    paths = write_hn_launch_artifacts(
        output_dir=output_dir,
        production=production_report,
        discrimination=discrimination_report,
        manifest_path=manifest_path,
        sidecar_ok=sidecar_ok,
    )
    table = render_hn_comparison_table(
        production=production_report,
        discrimination=discrimination_report,
    )
    print("\n" + table)
    print(f"\nWrote {paths['comparison_table']}")
    print(f"Wrote {paths['summary']}")
    print("\nNext: read docs/HN_LAUNCH.md and docs/HN_POST_DRAFT.md")


if __name__ == "__main__":
    main()
