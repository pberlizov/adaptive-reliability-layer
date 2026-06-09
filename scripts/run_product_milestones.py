#!/usr/bin/env python3
"""Run product milestones M1–M5 (data export, pilots, verification, sidecar check)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    py = sys.executable
    results_dir = ROOT / "results" / "product_milestones"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Export open datasets (PaySim, ULB credit card, IEEE-CIS) ===")
    _run([py, "scripts/export_open_datasets.py"])
    _run([py, "scripts/export_bundled_real_data.py"])

    print("\n=== M1/M2/M5 — Sklearn PaySim pilot (fast) ===")
    _run([py, "scripts/run_pilot_sklearn.py"])

    print("\n=== M1/M2/M5 — Torch PaySim pilot ===")
    _run([py, "scripts/run_pilot_torch.py"])

    print("\n=== M4 — Real-data verification (fraud-focused subset) ===")
    _run(
        [
            py,
            "scripts/run_real_data_verification.py",
            "--config",
            "configs/real_data_verification.yaml",
            "--sources",
            "breast_cancer,paysim_fraud,ieee_cis_fraud,openml_credit_g",
            "--output-dir",
            "results/real_data_verification",
        ]
    )

    print("\n=== M3 — HTTP sidecar smoke (pytest) ===")
    try:
        _run([py, "-m", "pytest", "tests/test_tier12_product.py::test_fastapi_health_endpoint", "-q"])
        sidecar_ok = True
    except subprocess.CalledProcessError:
        sidecar_ok = False
        print("Sidecar test skipped or failed (install .[serving] for full M3)")

    sklearn_status = ROOT / "results/pilot_sklearn/milestone_status.json"
    torch_status = ROOT / "results/pilot_torch/milestone_status.json"
    summary = {
        "sklearn_pilot": json.loads(sklearn_status.read_text()) if sklearn_status.exists() else None,
        "torch_pilot": json.loads(torch_status.read_text()) if torch_status.exists() else None,
        "verification_md": str(ROOT / "results/real_data_verification/verification_suite.md"),
        "sidecar_pytest": sidecar_ok,
    }
    out = results_dir / "run_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("Review:")
    print("  results/pilot_sklearn/dual_metric_report.md")
    print("  results/pilot_torch/dual_metric_report.md")
    print("  results/real_data_verification/verification_suite.md")
    print("  docs/sidecar_demo.md")


if __name__ == "__main__":
    main()
