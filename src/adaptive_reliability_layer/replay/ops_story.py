from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from ..runtime.config import RuntimeConfig
from ..runtime.types import OperatingMode
from .buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from .engine import run_offline_replay_comparison
from .real_data import load_real_data_bundle
from .report import render_operator_replay_report, render_replay_report


def run_public_ops_story(
    *,
    source_id: str,
    runtime_config: RuntimeConfig,
    output_dir: str | Path,
    controller_name: str = "bandit",
    operating_modes: tuple[OperatingMode, ...] = (
        OperatingMode.SHADOW,
        OperatingMode.RECOMMEND,
        OperatingMode.BOUNDED_AUTO,
    ),
    steps: int | None = None,
    batch_size: int | None = None,
    stream_cycles: int = 1,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    effective_steps = steps or runtime_config.replay.max_steps or 24
    effective_batch_size = batch_size or runtime_config.replay.batch_size
    bundle = load_real_data_bundle(
        source_id,
        steps=effective_steps,
        batch_size=effective_batch_size,
        stream_cycles=stream_cycles,
    )

    summary_payload: dict[str, object] = {
        "source_id": source_id,
        "wedge": bundle.wedge,
        "description": bundle.description,
        "validation_accuracy": bundle.validation_accuracy,
        "controller_name": controller_name,
        "stream_cycles": stream_cycles,
        "modes": {},
    }

    for mode in operating_modes:
        mode_output = output / mode.value
        mode_output.mkdir(parents=True, exist_ok=True)
        mode_config = replace(
            runtime_config,
            operating_mode=mode,
            governance=replace(
                runtime_config.governance,
                audit_db_path=str(mode_output / "audit.db"),
                snapshot_dir=str(mode_output / "snapshots"),
            ),
        )
        comparison = run_offline_replay_comparison(
            bundle.stream,
            runtime_config=mode_config,
            strategies=("frozen", controller_name),
            layer_builder=bundle.build_layer,
        )
        buyer_kpis = compute_buyer_kpis(comparison, controller_name=controller_name)
        (mode_output / "technical_report.md").write_text(render_replay_report(comparison), encoding="utf-8")
        (mode_output / "operator_report.md").write_text(
            render_operator_replay_report(comparison, controller_name=controller_name),
            encoding="utf-8",
        )
        (mode_output / "buyer_report.md").write_text(
            render_buyer_replay_report(
                comparison,
                source_label=f"{source_id} / {mode.value}",
                wedge=bundle.wedge,
                controller_name=controller_name,
            ),
            encoding="utf-8",
        )
        (mode_output / "summary.json").write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "mode": mode.value,
                    "buyer_kpis": asdict(buyer_kpis) if buyer_kpis is not None else None,
                    "comparison": {
                        "controller_vs_frozen_utility_delta": comparison.controller_vs_frozen_utility_delta,
                        "controller_vs_frozen_risk_reduction": comparison.controller_vs_frozen_risk_reduction,
                        "controller_vs_frozen_harmful_events_avoided": comparison.controller_vs_frozen_harmful_events_avoided,
                        "controller_vs_frozen_retrain_deferral_steps": comparison.controller_vs_frozen_retrain_deferral_steps,
                        "summaries": [asdict(summary) for summary in comparison.summaries],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_payload["modes"][mode.value] = {
            "buyer_kpis": asdict(buyer_kpis) if buyer_kpis is not None else None,
            "controller_vs_frozen_utility_delta": comparison.controller_vs_frozen_utility_delta,
            "controller_vs_frozen_risk_reduction": comparison.controller_vs_frozen_risk_reduction,
            "controller_vs_frozen_harmful_events_avoided": comparison.controller_vs_frozen_harmful_events_avoided,
            "controller_vs_frozen_retrain_deferral_steps": comparison.controller_vs_frozen_retrain_deferral_steps,
        }

    root_summary = output / "ops_story_summary.json"
    root_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return root_summary
