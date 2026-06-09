from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from ..runtime.config import RuntimeConfig
from ..runtime.kpi import KpiConfig, summarize_business_kpis
from ..runtime.types import OperatingMode
from .buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from .engine import run_offline_replay_comparison, run_replay_on_stream
from .loader import ReplayStream
from .report import ReplayComparisonResult, render_replay_report


def run_dual_mode_replay(
    stream: ReplayStream,
    *,
    runtime_config: RuntimeConfig,
    layer_builder: Callable[[RuntimeConfig], object],
    strategies: tuple[str, ...] = ("frozen", "bandit"),
    controller_name: str = "bandit",
) -> dict:
    """Run shadow (monitoring) and bounded_auto (intervention) on the same stream."""

    results: dict = {"strategies": strategies, "modes": {}}
    for mode_name, mode in (
        ("shadow", OperatingMode.SHADOW),
        ("bounded_auto", OperatingMode.BOUNDED_AUTO),
    ):
        mode_config = replace(runtime_config, operating_mode=mode)
        replay = run_offline_replay_comparison(
            stream,
            runtime_config=mode_config,
            strategies=strategies,
            layer_builder=layer_builder,
            controller_name=controller_name,
        )
        layer = layer_builder(
            replace(
                mode_config,
                policy=replace(mode_config.policy, name=controller_name),
                log_json=False,
            )
        )
        run = run_replay_on_stream(
            layer,
            stream,
            config=mode_config.replay,
            name=controller_name,
        )
        kpi_config = KpiConfig.from_mapping(asdict(mode_config.kpi))
        frozen = next((item for item in replay.summaries if item.name == "frozen"), None)
        controller = next((item for item in replay.summaries if item.name == controller_name), None)
        baseline_alert = (
            (frozen.risk_alert_count / max(1, frozen.steps)) if frozen and frozen.steps else None
        )
        business = summarize_business_kpis(
            run.surfaces,
            config=kpi_config,
            baseline_alert_rate=baseline_alert if mode_name == "bounded_auto" else None,
        )
        buyer = compute_buyer_kpis(replay, controller_name=controller_name)
        results["modes"][mode_name] = {
            "replay": replay,
            "buyer_kpis": buyer,
            "business_kpis": business,
            "surfaces": run.surfaces,
        }
    return results


def render_dual_metric_report(
    payload: dict,
    *,
    source_label: str = "dual replay",
    stream_records: int | None = None,
    label_delay_steps: int | None = None,
) -> str:
    lines = [
        f"# Dual-Metric Report — {source_label}",
        "",
        "Compare **shadow** (monitoring-only, no scored mutations) vs **bounded_auto** (low-risk interventions applied).",
        "",
    ]
    if stream_records is not None:
        lines.append(f"- Stream size: **{stream_records}** labeled events (rows in replay stream)")
    if label_delay_steps is not None:
        lines.append(f"- Label delay: **{label_delay_steps}** batch steps (delayed supervision via `reveal_labels`)")
    if stream_records is not None or label_delay_steps is not None:
        lines.append("")
    for mode_name, mode_payload in payload["modes"].items():
        replay: ReplayComparisonResult = mode_payload["replay"]
        lines.append(f"## {mode_name}")
        buyer = mode_payload.get("buyer_kpis")
        if buyer:
            lines.append(buyer.headline)
            lines.append("")
        business = mode_payload.get("business_kpis") or {}
        lines.append(
            f"- Business score (mean): {business.get('mean_business_score', 0):.3f} | "
            f"risk alert rate: {business.get('risk_alert_rate', 0):.1%} | "
            f"intervention rate: {business.get('intervention_rate', 0):.1%}"
        )
        if business.get("harmful_alert_reduction_pct") is not None:
            lines.append(f"- Harmful alert reduction vs frozen shadow: {business['harmful_alert_reduction_pct']:.0f}%")
        lines.append("")
        lines.append(render_buyer_replay_report(replay, source_label=f"{source_label} / {mode_name}"))
        lines.append("")
        lines.append("### Technical table")
        lines.append(render_replay_report(replay))
        lines.append("")
    return "\n".join(lines)


def write_dual_metric_artifacts(
    payload: dict,
    output_dir: str | Path,
    *,
    source_label: str,
    stream_records: int | None = None,
    label_delay_steps: int | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = render_dual_metric_report(
        payload,
        source_label=source_label,
        stream_records=stream_records,
        label_delay_steps=label_delay_steps,
    )
    (output / "dual_metric_report.md").write_text(report, encoding="utf-8")

    serializable = {
        "source_label": source_label,
        "modes": {},
    }
    for mode_name, mode_payload in payload["modes"].items():
        replay: ReplayComparisonResult = mode_payload["replay"]
        buyer = mode_payload.get("buyer_kpis")
        serializable["modes"][mode_name] = {
            "buyer_kpis": asdict(buyer) if buyer else None,
            "business_kpis": mode_payload.get("business_kpis"),
            "summaries": [summary.__dict__ for summary in replay.summaries],
            "utility_delta": replay.controller_vs_frozen_utility_delta,
            "risk_reduction": replay.controller_vs_frozen_risk_reduction,
        }
    (output / "dual_metric_report.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output / "dual_metric_report.md"
