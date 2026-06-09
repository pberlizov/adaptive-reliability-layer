#!/usr/bin/env python3
"""Compare shadow vs bounded_auto on torch tabular shift + full-intervention research benchmark."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from adaptive_reliability_layer.replay.engine import run_offline_replay_comparison
from adaptive_reliability_layer.replay.real_data import load_real_data_bundle
from adaptive_reliability_layer.replay.tta_comparison import render_tta_comparison_report, run_tta_tabular_comparison
from adaptive_reliability_layer.runtime.config import load_runtime_config
from adaptive_reliability_layer.runtime.types import OperatingMode


def _action_histogram(surfaces) -> dict[str, int]:
    counts: dict[str, int] = {}
    for surface in surfaces:
        key = surface.action_taken
        counts[key] = counts.get(key, 0) + 1
    return counts


def _recommended_histogram(surfaces) -> dict[str, int]:
    counts: dict[str, int] = {}
    for surface in surfaces:
        key = surface.recommended_action
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    output = Path("results/bounded_auto_demo")
    output.mkdir(parents=True, exist_ok=True)

    steps = 90
    batch_size = 48
    bundle = load_real_data_bundle(
        "tabular_breast_cancer_shift",
        steps=steps,
        batch_size=batch_size,
        seed=7,
    )

    base_config = load_runtime_config("configs/bounded_auto_demo.yaml")
    base_config = replace(
        base_config,
        replay=replace(base_config.replay, max_steps=steps, batch_size=batch_size, label_delay_steps=0),
        governance=replace(
            base_config.governance,
            audit_db_path=str(output / "audit_bounded.db"),
            snapshot_dir=str(output / "snapshots_bounded"),
        ),
    )

    modes = (
        ("shadow", OperatingMode.SHADOW),
        ("bounded_auto", OperatingMode.BOUNDED_AUTO),
    )
    sections: list[str] = [
        "# Bounded Auto Demo — Torch Tabular Shift Stream",
        "",
        "Compares **shadow** (no scored mutations) vs **bounded_auto** (low-risk actions applied).",
        "Labels aligned (`label_delay_steps=0`) so accuracy reflects decisions on the current batch.",
        "",
    ]

    mode_results = {}
    for mode_name, operating_mode in modes:
        mode_config = replace(
            base_config,
            operating_mode=operating_mode,
            governance=replace(
                base_config.governance,
                audit_db_path=str(output / f"audit_{mode_name}.db"),
                snapshot_dir=str(output / f"snapshots_{mode_name}"),
            ),
        )
        replay = run_offline_replay_comparison(
            bundle.stream,
            runtime_config=mode_config,
            strategies=("frozen", "bandit"),
            layer_builder=bundle.build_layer,
        )
        from adaptive_reliability_layer.replay.engine import run_replay_on_stream

        bandit_layer = bundle.build_layer(
            replace(mode_config, policy=replace(mode_config.policy, name="bandit"), log_json=False)
        )
        bandit_run = run_replay_on_stream(
            bandit_layer,
            bundle.stream,
            config=mode_config.replay,
            name="bandit",
        )
        mode_results[mode_name] = {
            "replay": replay,
            "surfaces": bandit_run.surfaces,
        }
        kpis = compute_buyer_kpis(replay, controller_name="bandit")
        sections.extend(
            [
                f"## {mode_name}",
                "",
                render_buyer_replay_report(
                    replay,
                    source_label=f"tabular_breast_cancer_shift / {mode_name}",
                    wedge="general_tabular",
                ),
                "",
                f"Bandit actions taken: `{json.dumps(_action_histogram(bandit_run.surfaces))}`",
                f"Bandit actions recommended: `{json.dumps(_recommended_histogram(bandit_run.surfaces))}`",
                "",
            ]
        )

    # Full-intervention research benchmark (no shadow gating)
    tta_result = run_tta_tabular_comparison(steps=steps, batch_size=batch_size, seed=7)
    sections.extend(
        [
            "## Research benchmark (full interventions — no shadow)",
            "",
            "Policies mutate the model before scoring; this is the fair accuracy head-to-head.",
            "",
            render_tta_comparison_report(tta_result),
            "",
        ]
    )

    report_path = output / "bounded_auto_demo.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")

    shadow_replay = mode_results["shadow"]["replay"]
    bounded_replay = mode_results["bounded_auto"]["replay"]
    shadow_bandit = next(item for item in shadow_replay.summaries if item.name == "bandit")
    bounded_bandit = next(item for item in bounded_replay.summaries if item.name == "bandit")
    shadow_frozen = next(item for item in shadow_replay.summaries if item.name == "frozen")
    bounded_frozen = next(item for item in bounded_replay.summaries if item.name == "frozen")

    summary = {
        "shadow": {
            "frozen_accuracy": shadow_frozen.mean_accuracy,
            "bandit_accuracy": shadow_bandit.mean_accuracy,
            "bandit_intervention_rate": shadow_bandit.intervention_rate,
            "bandit_actions_taken": _action_histogram(mode_results["shadow"]["surfaces"]),
        },
        "bounded_auto": {
            "frozen_accuracy": bounded_frozen.mean_accuracy,
            "bandit_accuracy": bounded_bandit.mean_accuracy,
            "bandit_intervention_rate": bounded_bandit.intervention_rate,
            "bandit_actions_taken": _action_histogram(mode_results["bounded_auto"]["surfaces"]),
            "bandit_actions_recommended": _recommended_histogram(mode_results["bounded_auto"]["surfaces"]),
        },
        "research_tabular": {
            strategy.name: {
                "accuracy": strategy.overall_accuracy,
                "utility": strategy.mean_utility,
                "risk_capital": strategy.mean_risk_capital,
                "adaptations": strategy.adaptations,
                "resets": strategy.resets,
            }
            for strategy in tta_result.strategies
            if strategy.name in {"frozen", "bandit", "tent"}
        },
    }
    (output / "bounded_auto_demo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {report_path.resolve()}")
    print()
    print("=== Shadow (monitoring only) ===")
    print(f"  frozen acc={shadow_frozen.mean_accuracy:.3f}  bandit acc={shadow_bandit.mean_accuracy:.3f}")
    print(f"  bandit interventions={shadow_bandit.intervention_rate:.1%}  actions={summary['shadow']['bandit_actions_taken']}")
    print()
    print("=== Bounded auto (low-risk actions applied) ===")
    print(f"  frozen acc={bounded_frozen.mean_accuracy:.3f}  bandit acc={bounded_bandit.mean_accuracy:.3f}")
    print(
        f"  bandit interventions={bounded_bandit.intervention_rate:.1%}  "
        f"actions taken={summary['bounded_auto']['bandit_actions_taken']}"
    )
    print(f"  recommended={summary['bounded_auto']['bandit_actions_recommended']}")
    print()
    print("=== Research tabular (full interventions) ===")
    for name, row in summary["research_tabular"].items():
        print(f"  {name}: acc={row['accuracy']:.3f} utility={row['utility']:.3f} risk_cap={row['risk_capital']:.1f}")


if __name__ == "__main__":
    main()
