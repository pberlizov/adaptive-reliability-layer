#!/usr/bin/env python3
"""Compare regime_aware_delayed_bandit vs delayed_hybrid on production replay paths."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.engine import run_offline_replay_comparison
from adaptive_reliability_layer.replay.real_data import load_paysim_fraud_torch_bundle
from adaptive_reliability_layer.replay.report import render_replay_report
from adaptive_reliability_layer.runtime.config import load_runtime_config
from adaptive_reliability_layer.runtime.types import OperatingMode
from adaptive_reliability_layer.tabular_benchmark import (
    DelayedHybridBanditSpecialistPolicy,
    FrozenTabularPolicy,
    RegimeAwareDelayedBanditTabularPolicy,
)
from adaptive_reliability_layer.temporal_fashion_mnist_benchmark import (
    render_temporal_fashion_mnist_report,
    run_temporal_fashion_mnist_benchmark,
)


STRATEGIES = ("frozen", "regime_aware_delayed_bandit", "delayed_hybrid")


def _summary_row(name: str, result) -> dict:
    summary = next(item for item in result.summaries if item.name == name)
    return {
        "strategy": name,
        "mean_accuracy": summary.mean_accuracy,
        "mean_utility": summary.mean_utility,
        "mean_risk_capital": summary.mean_risk_capital,
        "intervention_rate": summary.intervention_rate,
        "risk_alert_count": summary.risk_alert_count,
        "utility_delta_vs_frozen": (
            summary.mean_utility - next(s.mean_utility for s in result.summaries if s.name == "frozen")
        ),
    }


def _run_sklearn_paysim() -> dict | None:
    # Specialist routing requires torch-style export_state/load_state snapshots.
    return None


def _run_torch_paysim() -> dict:
    config = load_runtime_config("configs/pilot_fraud_torch.yaml")
    bundle = load_paysim_fraud_torch_bundle(
        steps=config.replay.max_steps or 48,
        batch_size=config.replay.batch_size,
        stream_cycles=2,
    )
    runtime = replace(
        config,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        policy=replace(config.policy, distance_threshold=0.55),
    )
    result = run_offline_replay_comparison(
        bundle.stream,
        runtime_config=runtime,
        strategies=STRATEGIES,
        layer_builder=bundle.build_layer,
    )
    return {
        "benchmark": "paysim_torch_bounded_auto",
        "label_delay_steps": config.replay.label_delay_steps,
        "rows": [_summary_row(name, result) for name in STRATEGIES],
        "report": render_replay_report(result),
    }


def _run_temporal_fashion_mnist() -> dict:
    factories = [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        (
            "regime_aware_delayed_bandit",
            lambda reference: RegimeAwareDelayedBanditTabularPolicy(reference),
        ),
        (
            "delayed_hybrid",
            lambda reference: DelayedHybridBanditSpecialistPolicy(reference, distance_threshold=0.55),
        ),
    ]
    result = run_temporal_fashion_mnist_benchmark(
        steps=36,
        batch_size=64,
        reveal_delay_steps=12,
        policy_factories=factories,
    )
    report = render_temporal_fashion_mnist_report(result)
    rows = []
    frozen_utility = next(
        strategy.base.mean_utility for strategy in result.strategies if strategy.base.name == "frozen"
    )
    for strategy in result.strategies:
        base = strategy.base
        rows.append(
            {
                "strategy": base.name,
                "accuracy": base.overall_accuracy,
                "mean_utility": base.mean_utility,
                "revealed_accuracy": strategy.revealed_accuracy,
                "utility_delta_vs_frozen": base.mean_utility - frozen_utility,
            }
        )
    return {
        "benchmark": "temporal_fashion_mnist",
        "label_delay_steps": 12,
        "rows": rows,
        "report": report,
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Integrated policy comparison",
        "",
        "Compares `regime_aware_delayed_bandit` (production default) against `delayed_hybrid`",
        "(recurrence gate + exchangeability + residual correction + outstanding-feedback control).",
        "",
        "Note: `delayed_hybrid` requires torch tabular adapters (specialist snapshots use export_state/load_state).",
        "",
    ]
    for section in payload["benchmarks"]:
        lines.append(f"## {section['benchmark']}")
        lines.append(f"- label_delay_steps: {section.get('label_delay_steps')}")
        lines.append("")
        lines.append("| strategy | mean_acc / accuracy | mean_utility | Δ utility vs frozen |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in section["rows"]:
            accuracy = row.get("mean_accuracy", row.get("accuracy"))
            acc_text = "n/a" if accuracy is None else f"{accuracy:.3f}"
            utility = row["mean_utility"]
            lines.append(
                f"| {row['strategy']} | {acc_text} | {utility:.3f} | {row['utility_delta_vs_frozen']:+.3f} |"
            )
        hybrid = next((row for row in section["rows"] if row["strategy"] == "delayed_hybrid"), None)
        baseline = next(
            (row for row in section["rows"] if row["strategy"] == "regime_aware_delayed_bandit"),
            None,
        )
        if hybrid and baseline:
            hybrid_u = hybrid["mean_utility"]
            baseline_u = baseline["mean_utility"]
            lines.append("")
            lines.append(
                f"**delayed_hybrid vs regime_aware_delayed_bandit utility delta:** {hybrid_u - baseline_u:+.3f}"
            )
        lines.append("")
        lines.append("<details><summary>Full replay report</summary>")
        lines.append("")
        lines.append("```")
        lines.append(section["report"].rstrip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    benchmarks = [
        section
        for section in (
            _run_torch_paysim(),
            _run_sklearn_paysim(),
            _run_temporal_fashion_mnist(),
        )
        if section is not None
    ]
    payload = {"benchmarks": benchmarks}
    out_dir = ROOT / "results" / "integrated_policy_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    md_path = out_dir / "comparison.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(_render_markdown(payload))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
