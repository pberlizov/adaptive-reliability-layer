"""Public fraud benchmark runner (PaySim, IEEE-CIS, German Credit)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from ..replay.buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from ..replay.engine import run_offline_replay_comparison, run_replay_on_stream
from ..replay.real_data import load_real_data_bundle
from ..runtime.config import load_runtime_config
from ..runtime.model_adapter import TorchTabularModelAdapter
from ..runtime.types import OperatingMode
from ..tabular_benchmark import (
    BanditTabularPolicy,
    FrozenTabularPolicy,
    MultiActionTabularPolicy,
    TabularBatch,
    _build_reference_profile,
    _evaluate_strategy,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_fraud_csvs() -> None:
    root = _repo_root()
    paysim = root / "data" / "fraud" / "paysim.csv"
    ieee_full = root / "data" / "fraud" / "ieee_cis_full.csv"
    ieee_sample = root / "data" / "fraud" / "ieee_cis_sample.csv"
    if paysim.exists() and (ieee_full.exists() or ieee_sample.exists()):
        return
    export = root / "scripts" / "export_open_datasets.py"
    if not export.exists():
        export = root / "scripts" / "export_bundled_fraud_data.py"
    subprocess.run([sys.executable, str(export)], check=True, cwd=root)


def _action_histogram(surfaces) -> dict[str, int]:
    counts: dict[str, int] = {}
    for surface in surfaces:
        counts[surface.action_taken] = counts.get(surface.action_taken, 0) + 1
    return counts


def _recommended_histogram(surfaces) -> dict[str, int]:
    counts: dict[str, int] = {}
    for surface in surfaces:
        counts[surface.recommended_action] = counts.get(surface.recommended_action, 0) + 1
    return counts


def _stream_to_tabular_batches(stream, batch_size: int, max_steps: int | None) -> list[TabularBatch]:
    import numpy as np

    from .loader import iter_replay_batches

    batches: list[TabularBatch] = []
    for step, batch, _ in iter_replay_batches(stream, batch_size=batch_size, max_steps=max_steps):
        if batch.labels is None:
            continue
        batches.append(
            TabularBatch(
                features=np.asarray(batch.features, dtype=np.float32),
                labels=np.asarray(batch.labels, dtype=np.int64),
                regime=str(batch.regime or f"step_{step}"),
            )
        )
    return batches


def _run_torch_full_intervention(*, steps: int, batch_size: int, stream_cycles: int, seed: int = 7) -> dict:
    bundle = load_real_data_bundle(
        "paysim_fraud_torch",
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
    )
    config = load_runtime_config(_repo_root() / "configs" / "default.yaml")
    layer = bundle.build_layer(replace(config, log_json=False))
    if not isinstance(layer._adapter, TorchTabularModelAdapter):
        raise TypeError("expected torch paysim bundle")
    model = layer._adapter.inner.clone()

    reference, reference_scores = _build_reference_profile(model, bundle.reference_batches)
    batches = _stream_to_tabular_batches(bundle.stream, batch_size=batch_size, max_steps=steps * stream_cycles)
    strategies = []
    for name, policy in (
        ("frozen", FrozenTabularPolicy()),
        ("multi_action", MultiActionTabularPolicy(reference)),
        ("bandit", BanditTabularPolicy(reference)),
    ):
        result = _evaluate_strategy(name, model.clone(), policy, batches, reference, reference_scores)
        strategies.append(
            {
                "name": result.name,
                "accuracy": result.overall_accuracy,
                "utility": result.mean_utility,
                "risk_capital": result.mean_risk_capital,
                "adaptations": result.adaptations,
                "resets": result.resets,
                "risk_alerts": result.risk_alerts,
            }
        )
    return {"source_id": "paysim_fraud_torch", "strategies": strategies}


def _run_source_modes(
    source_id: str,
    *,
    base_config,
    steps: int,
    batch_size: int,
    stream_cycles: int,
    output_dir: Path,
) -> dict:
    effective_steps = steps * stream_cycles
    bundle = load_real_data_bundle(
        source_id,
        steps=steps,
        batch_size=batch_size,
        stream_cycles=stream_cycles,
    )
    controller_name = "multi_action" if base_config.policy.name == "multi_action" else "bandit"
    results: dict = {"source_id": source_id, "stream_rows": bundle.stream_size, "modes": {}}

    for mode_name, mode in (("shadow", OperatingMode.SHADOW), ("bounded_auto", OperatingMode.BOUNDED_AUTO)):
        mode_config = replace(
            base_config,
            operating_mode=mode,
            replay=replace(base_config.replay, max_steps=effective_steps, label_delay_steps=0),
            governance=replace(
                base_config.governance,
                audit_db_path=str(output_dir / source_id / f"audit_{mode_name}.db"),
                snapshot_dir=str(output_dir / source_id / f"snapshots_{mode_name}"),
            ),
        )
        replay = run_offline_replay_comparison(
            bundle.stream,
            runtime_config=mode_config,
            strategies=("frozen", controller_name),
            layer_builder=bundle.build_layer,
        )
        layer = bundle.build_layer(
            replace(mode_config, policy=replace(mode_config.policy, name=controller_name), log_json=False)
        )
        run = run_replay_on_stream(layer, bundle.stream, config=mode_config.replay, name=controller_name)
        kpis = compute_buyer_kpis(replay, controller_name=controller_name)
        source_dir = output_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / f"{mode_name}_buyer_report.md").write_text(
            render_buyer_replay_report(
                replay,
                source_label=f"{source_id} / {mode_name} / {stream_cycles} cycles",
                wedge="fraud_risk",
            ),
            encoding="utf-8",
        )
        results["modes"][mode_name] = {
            "buyer_kpis": asdict(kpis) if kpis else None,
            "summaries": [summary.__dict__ for summary in replay.summaries],
            "actions_taken": _action_histogram(run.surfaces),
            "actions_recommended": _recommended_histogram(run.surfaces),
        }
    return results


def run_fraud_public_benchmark(
    *,
    config_path: str | Path = "configs/fraud_public_benchmark.yaml",
    output_dir: str | Path = "results/fraud_public_benchmark",
    stream_cycles: int = 6,
    skip_torch_full: bool = False,
) -> Path:
    _ensure_fraud_csvs()
    root = _repo_root()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    base_config = load_runtime_config(config_file)
    steps = base_config.replay.max_steps or 24
    batch_size = base_config.replay.batch_size
    bounded_config = replace(
        base_config,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        policy=replace(base_config.policy, name="multi_action"),
    )

    all_results: dict = {
        "stream_cycles": stream_cycles,
        "effective_steps": steps * stream_cycles,
        "sources": {},
    }

    for source_id in ("openml_credit_g", "paysim_fraud", "ieee_cis_fraud"):
        print(f"=== {source_id} ===")
        all_results["sources"][source_id] = _run_source_modes(
            source_id,
            base_config=bounded_config,
            steps=steps,
            batch_size=batch_size,
            stream_cycles=stream_cycles,
            output_dir=output,
        )
        shadow = all_results["sources"][source_id]["modes"]["shadow"]
        bounded = all_results["sources"][source_id]["modes"]["bounded_auto"]
        if shadow.get("buyer_kpis") and bounded.get("buyer_kpis"):
            print(
                f"  shadow risk↓ {shadow['buyer_kpis']['risk_exposure_reduction_pct']:.0f}% | "
                f"bounded risk↓ {bounded['buyer_kpis']['risk_exposure_reduction_pct']:.0f}% | "
                f"bounded actions {bounded['actions_taken']}"
            )

    if not skip_torch_full:
        print("=== paysim_fraud_torch (full interventions) ===")
        all_results["torch_full_intervention"] = _run_torch_full_intervention(
            steps=steps,
            batch_size=batch_size,
            stream_cycles=stream_cycles,
        )
        for row in all_results["torch_full_intervention"]["strategies"]:
            print(
                f"  {row['name']}: acc={row['accuracy']:.3f} utility={row['utility']:.3f} "
                f"risk={row['risk_capital']:.1f} adapt={row['adaptations']} reset={row['resets']}"
            )

    summary_lines = [
        "# Public Fraud Benchmark",
        "",
        f"Steps per cycle: {steps} | Cycles: {stream_cycles} | Effective steps: {steps * stream_cycles}",
        "",
        "Sources: German Credit, PaySim (synthetic), IEEE-CIS (sample or synthetic), chronological streams.",
        "",
    ]
    for source_id, payload in all_results["sources"].items():
        summary_lines.append(f"## {source_id}")
        for mode_name, mode_payload in payload["modes"].items():
            kpis = mode_payload.get("buyer_kpis") or {}
            summary_lines.append(f"- **{mode_name}**: {kpis.get('headline', 'n/a')}")
            summary_lines.append(f"  - actions taken: `{mode_payload['actions_taken']}`")
        summary_lines.append("")

    if "torch_full_intervention" in all_results:
        summary_lines.append("## PaySim torch — full interventions (no shadow)")
        for row in all_results["torch_full_intervention"]["strategies"]:
            summary_lines.append(
                f"- {row['name']}: acc={row['accuracy']:.3f}, utility={row['utility']:.3f}, "
                f"risk_capital={row['risk_capital']:.1f}"
            )

    (output / "fraud_public_benchmark.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output / "fraud_public_benchmark.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {output / 'fraud_public_benchmark.md'}")
    return output
