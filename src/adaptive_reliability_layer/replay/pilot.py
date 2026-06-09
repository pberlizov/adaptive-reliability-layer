from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..runtime.config import RuntimeConfig
from ..runtime.types import OperatingMode
from .dual_metric import run_dual_mode_replay, write_dual_metric_artifacts
from .engine import (
    build_synthetic_fraud_like_stream,
    export_stream_to_csv,
    run_offline_replay_comparison,
)
from .report import render_replay_report


@dataclass(frozen=True)
class PilotCaseStudy:
    name: str
    wedge: str
    description: str
    primary_kpi: str
    dataset_path: str | None
    use_synthetic_stream: bool
    label_delay_steps: int
    operating_mode: str
    strategies: tuple[str, ...]
    dual_mode: bool = True
    controller_name: str = "regime_aware_delayed_bandit"


DEFAULT_PILOT = PilotCaseStudy(
    name="fraud_risk_tabular_replay",
    wedge="fraud_risk",
    description=(
        "Offline replay of a delayed-label fraud/risk-style tabular stream. "
        "Demonstrates shift detection, bounded interventions, and utility gains vs frozen inference."
    ),
    primary_kpi="utility_under_delayed_labels",
    dataset_path=None,
    use_synthetic_stream=True,
    label_delay_steps=4,
    operating_mode="shadow",
    strategies=("frozen", "naive", "controller", "bandit"),
    dual_mode=True,
    controller_name="regime_aware_delayed_bandit",
)


def _load_stream(pilot: PilotCaseStudy, runtime_config: RuntimeConfig):
    if pilot.use_synthetic_stream:
        stream = build_synthetic_fraud_like_stream(
            steps=runtime_config.replay.max_steps or 90,
            batch_size=runtime_config.replay.batch_size,
        )
        return stream, None
    if pilot.dataset_path:
        from .loader import load_replay_csv

        return load_replay_csv(pilot.dataset_path, runtime_config.replay), Path(pilot.dataset_path)
    raise ValueError("pilot requires either use_synthetic_stream or dataset_path")


def run_pilot_case_study(
    pilot: PilotCaseStudy,
    *,
    runtime_config: RuntimeConfig,
    output_dir: str | Path,
    layer_builder=None,
    stream: ReplayStream | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    csv_path: Path | None
    if stream is not None:
        csv_path = Path(pilot.dataset_path) if pilot.dataset_path else output / "pilot_stream.csv"
        if csv_path == output / "pilot_stream.csv" or not csv_path.exists():
            export_stream_to_csv(stream, csv_path)
    else:
        stream, loaded_csv = _load_stream(pilot, runtime_config)
        csv_path = loaded_csv
        if pilot.use_synthetic_stream:
            csv_path = output / "synthetic_stream.csv"
            export_stream_to_csv(stream, csv_path)

    replay_config = runtime_config.replay.__class__(
        timestamp_column=runtime_config.replay.timestamp_column,
        label_column=runtime_config.replay.label_column,
        feature_prefix=runtime_config.replay.feature_prefix,
        batch_size=runtime_config.replay.batch_size,
        label_delay_steps=pilot.label_delay_steps,
        max_steps=runtime_config.replay.max_steps,
    )
    config = replace(
        runtime_config,
        operating_mode=OperatingMode(pilot.operating_mode),
        replay=replay_config,
        policy=replace(runtime_config.policy, name=pilot.controller_name),
    )

    summary: dict = {
        "report_md": str(output / "pilot_report.md"),
        "report_json": str(output / "pilot_report.json"),
        "dataset_csv": str(csv_path) if csv_path else "",
        "primary_kpi": pilot.primary_kpi,
        "stream_records": len(stream.records),
        "label_delay_steps": pilot.label_delay_steps,
        "controller_name": pilot.controller_name,
    }

    if pilot.dual_mode and layer_builder is not None:
        dual = run_dual_mode_replay(
            stream,
            runtime_config=config,
            layer_builder=layer_builder,
            strategies=pilot.strategies,
            controller_name=pilot.controller_name,
        )
        write_dual_metric_artifacts(
            dual,
            output,
            source_label=pilot.name,
            stream_records=len(stream.records),
            label_delay_steps=pilot.label_delay_steps,
        )
        from .milestone_status import evaluate_pilot_milestones, write_milestone_status

        dual_json_path = output / "dual_metric_report.json"
        dual_payload = json.loads(dual_json_path.read_text(encoding="utf-8")) if dual_json_path.exists() else {}
        policy_path = Path(runtime_config.policy_state_save_path) if runtime_config.policy_state_save_path else None
        milestone_report = evaluate_pilot_milestones(
            dual_metric_json=dual_payload,
            stream_records=len(stream.records),
            label_delay_steps=pilot.label_delay_steps,
            policy_state_path=policy_path,
        )
        write_milestone_status(milestone_report, output)
        summary["milestone_status"] = str(output / "milestone_status.json")
        summary["milestones_passed"] = milestone_report.passed
        bounded = dual["modes"]["bounded_auto"]["replay"]
        summary.update(
            {
                "dual_metric_md": str(output / "dual_metric_report.md"),
                "utility_delta": bounded.controller_vs_frozen_utility_delta,
                "risk_reduction": bounded.controller_vs_frozen_risk_reduction,
            }
        )
    else:
        result = run_offline_replay_comparison(
            stream,
            runtime_config=config,
            strategies=pilot.strategies,
            layer_builder=layer_builder,
        )
        (output / "pilot_report.md").write_text(render_replay_report(result), encoding="utf-8")
        (output / "pilot_report.json").write_text(
            json.dumps(
                {
                    "pilot": asdict(pilot),
                    "summaries": [summary.__dict__ for summary in result.summaries],
                    "controller_vs_frozen_utility_delta": result.controller_vs_frozen_utility_delta,
                    "controller_vs_frozen_risk_reduction": result.controller_vs_frozen_risk_reduction,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary["utility_delta"] = result.controller_vs_frozen_utility_delta
        summary["risk_reduction"] = result.controller_vs_frozen_risk_reduction

    return summary
