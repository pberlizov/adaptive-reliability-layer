"""Hacker News launch orchestration — export data, benchmarks, comparison table."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .discrimination_benchmark import (
    DiscriminationBenchmarkReport,
    run_discrimination_benchmark,
    write_discrimination_artifacts,
)
from .production_benchmark import (
    ProductionBenchmarkReport,
    load_production_benchmark_spec,
    run_production_benchmark_suite,
)


from ..data_export.open_datasets import export_minimal_datasets, export_open_datasets
from ..workspace import data_dir, resolve_config_path, resolve_workspace_root


def export_public_datasets(*, root: Path | None = None, minimal: bool = False) -> Path:
    """Export fraud CSVs into workspace data/ (pip-installable)."""

    workspace = resolve_workspace_root(root)
    if minimal:
        return export_minimal_datasets(root=workspace)
    return export_open_datasets(root=workspace)


def run_hn_production_benchmark(
    *,
    config_path: Path,
    output_dir: Path,
) -> ProductionBenchmarkReport:
    runtime_config, spec = load_production_benchmark_spec(config_path)
    return run_production_benchmark_suite(
        runtime_config=runtime_config,
        spec=spec,
        output_dir=output_dir,
    )


def run_hn_discrimination_benchmark(
    *,
    config_path: Path,
    output_dir: Path,
) -> DiscriminationBenchmarkReport:
    report = run_discrimination_benchmark(config_path=config_path)
    write_discrimination_artifacts(report, output_dir=output_dir)
    return report


def render_hn_comparison_table(
    *,
    production: ProductionBenchmarkReport | None,
    discrimination: DiscriminationBenchmarkReport | None,
    quick: bool = False,
) -> str:
    quick_banner = (
        [
            "> **Quick demo run (PaySim toy only).** "
            "These numbers are from the 2–5 min synthetic demo and cover 1 source. "
            "For the full 3-source claim (ULB + IEEE-CIS + PaySim), run `arl-hn-launch` (~30–90 min).",
            "",
        ]
        if quick
        else []
    )
    lines = [
        "# Adaptive Reliability Layer — public benchmark comparison",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        *quick_banner,
        "Two suites answer different questions:",
        "",
        "1. **Production claim** — temporal fraud replay with delayed labels: does ARL beat frozen / scheduled retrain on **utility** and **proxy risk**?",
        "2. **Hard-slice discrimination** — harder tails + imbalance-aware metrics: where is there headroom when accuracy saturates?",
        "",
    ]

    if production is not None:
        controller = production.spec.controller_name
        lines.extend(
            [
                "## Production claim suite",
                "",
                f"Controller: `{controller}` · Suite passed: **{'yes' if production.suite_passed else 'no'}** "
                f"({production.core_sources_passing} core sources)",
                "",
                "| Source | Pass | Utility Δ vs frozen | vs scheduled | vs naive | Risk ↓ | Steering | Temporal |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source in production.sources:
            scheduled = dict(source.baseline_utility_deltas).get("scheduled_retrain")
            naive = dict(source.baseline_utility_deltas).get("naive")
            utility = f"{source.utility_delta:+.3f}" if source.utility_delta is not None else "n/a"
            sched = f"{scheduled:+.3f}" if scheduled is not None else "n/a"
            na = f"{naive:+.3f}" if naive is not None else "n/a"
            risk = f"{source.risk_reduction_pct:.1f}%" if source.risk_reduction_pct is not None else "n/a"
            steering = f"{source.correction_applied_rate:.0%} corr"
            lines.append(
                f"| `{source.source_id}` | {'PASS' if source.passed else 'FAIL'} | {utility} | {sched} | {na} | {risk} | "
                f"{steering} | {'yes' if source.temporal_split else 'no'} |"
            )
        lines.append("")

    if discrimination is not None:
        lines.extend(
            [
                "## Hard-slice discrimination (fraud + Elliptic + BAF)",
                "",
                f"Controller reference: `{discrimination.controller_name}` · "
                f"Sources with rankable metrics: **{discrimination.rankable_sources}/{len(discrimination.sources)}**",
                "",
            ]
        )
        for source in discrimination.sources:
            lines.extend(
                [
                    f"### {source.source_id}",
                    source.description,
                    f"- headroom: **{'yes' if source.has_headroom else 'limited'}** · rankable metrics: **{source.rankable_metric_count}**",
                    "",
                    "| Strategy | acc | bal_acc | PR-AUC | recall | cost err | retrain rec |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for item in source.strategy_metrics:
                metrics = item.stream_metrics
                pr = "n/a" if metrics.pr_auc is None else f"{metrics.pr_auc:.3f}"
                lines.append(
                    f"| `{item.name}` | {metrics.accuracy:.3f} | {metrics.balanced_accuracy:.3f} | "
                    f"{pr} | {metrics.recall:.3f} | {metrics.cost_weighted_error:.3f} | "
                    f"{item.mean_retrain_recommendation_rate:.3f} |"
                )
            lines.append("")

    if quick:
        lines.extend(
            [
                "## How to reproduce",
                "",
                "```bash",
                "pip install \"adaptive-reliability-layer[torch,serving]>=0.3.4\"",
                "arl-demo",
                "```",
                "",
                "Full five-dataset suite (numbers for Show HN post): `arl-hn-launch` (~30–90 min).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## How to reproduce",
                "",
                "```bash",
                "pip install \"adaptive-reliability-layer[torch,serving]>=0.3.4\"",
                "arl-hn-launch",
                "```",
                "",
                "Quick toy demo (~2–5 min): `arl-demo`",
                "",
                "Optional real Elliptic/BAF without Kaggle:",
                "",
                "```bash",
                "arl-export-datasets",
                "arl-hn-launch --skip-export",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_hn_launch_artifacts(
    *,
    output_dir: Path,
    production: ProductionBenchmarkReport | None,
    discrimination: DiscriminationBenchmarkReport | None,
    manifest_path: Path | None,
    sidecar_ok: bool | None = None,
    quick: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = render_hn_comparison_table(production=production, discrimination=discrimination, quick=quick)
    table_filename = "comparison_table_quick.md" if quick else "comparison_table.md"
    table_path = output_dir / table_filename
    table_path.write_text(table, encoding="utf-8")

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sidecar_health_ok": sidecar_ok,
        "manifest": str(manifest_path) if manifest_path else None,
    }
    if production is not None:
        payload["production"] = production.to_dict()
    if discrimination is not None:
        payload["discrimination"] = {
            "controller_name": discrimination.controller_name,
            "rankable_sources": discrimination.rankable_sources,
            "source_count": len(discrimination.sources),
        }
    summary_path = output_dir / "hn_launch_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"comparison_table": table_path, "summary": summary_path}


def verify_sidecar_health() -> bool:
    try:
        from fastapi.testclient import TestClient

        from ..runtime.config import load_runtime_config
        from ..serving.app import create_app
        from ..serving.config import ServingConfig
        from ..serving.loader import build_layer_for_serving

        config = load_runtime_config(resolve_config_path("default.yaml"))
        serving = ServingConfig(model_bundle="paysim_fraud_torch")
        layer = build_layer_for_serving(config, serving)
        client = TestClient(create_app(layer=layer, serving=serving, runtime_config=config))
        return client.get("/v1/health").status_code == 200
    except Exception:
        return False
