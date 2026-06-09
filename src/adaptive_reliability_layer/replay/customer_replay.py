"""Productized customer CSV/JSONL replay — shadow-first pilot deliverables."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..ingest.contract import events_to_replay_stream, load_events_csv, load_events_jsonl
from ..runtime.config import RuntimeConfig, load_runtime_config
from ..runtime.layer import ReliabilityLayer
from ..runtime.layer import build_reliability_layer_from_reference_batches
from ..runtime.model_adapter import TorchTabularModelAdapter
from ..tabular_benchmark import _build_reference_batches
from ..torch_model import TorchTabularAdapterModel
from .buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from .dual_metric import run_dual_mode_replay, write_dual_metric_artifacts
from .engine import build_layer_for_tabular_replay
from .loader import render_replay_schema_markdown
from .loader import ReplayStream
from .report import render_operator_replay_report, render_replay_report


@dataclass(frozen=True)
class CustomerReplaySpec:
    """Inputs for a design-partner historical replay."""

    input_path: Path
    config_path: Path
    output_dir: Path
    customer_label: str = "customer"
    wedge: str = "fraud_risk"
    dual_mode: bool = True
    batch_size: int | None = None
    label_delay_steps: int | None = None
    layer_builder: Callable[[RuntimeConfig], ReliabilityLayer] | None = None


@dataclass(frozen=True)
class CustomerReplayResult:
    customer_label: str
    input_path: str
    output_dir: str
    dual_mode: bool
    artifact_paths: tuple[str, ...]
    buyer_kpis: dict[str, Any] | None
    controller_vs_frozen_utility_delta: float | None
    controller_vs_frozen_risk_reduction: float | None


def _load_customer_events(path: Path) -> tuple[Any, ...]:
    if path.suffix == ".jsonl":
        return tuple(load_events_jsonl(path))
    return tuple(load_events_csv(path))


def _apply_overrides(
    config: RuntimeConfig,
    *,
    batch_size: int | None,
    label_delay_steps: int | None,
) -> RuntimeConfig:
    replay = config.replay
    if batch_size is not None:
        replay = replace(replay, batch_size=batch_size)
    if label_delay_steps is not None:
        replay = replace(replay, label_delay_steps=label_delay_steps)
    if batch_size is not None or label_delay_steps is not None:
        config = replace(config, replay=replay)
    return config


def _customer_stream_layer_builder(stream: ReplayStream) -> Callable[[RuntimeConfig], ReliabilityLayer]:
    features = np.stack([record.features for record in stream.records], axis=0).astype(np.float32)
    labels = np.asarray([record.label for record in stream.records], dtype=np.int64)

    def _builder(config: RuntimeConfig) -> ReliabilityLayer:
        if len(features) < 8:
            raise ValueError("customer replay needs at least 8 records to build a reference/model")
        split_train = max(4, int(0.5 * len(features)))
        split_valid = max(split_train + 2, int(0.75 * len(features)))
        split_valid = min(split_valid, len(features) - 1)
        x_train = features[:split_train]
        y_train = labels[:split_train]
        x_valid = features[split_train:split_valid]
        y_valid = labels[split_train:split_valid]
        if len(x_valid) < 2:
            x_valid = features[-max(2, len(features) // 4):]
            y_valid = labels[-max(2, len(features) // 4):]

        model = TorchTabularAdapterModel(input_dim=features.shape[1], seed=7)
        model.fit_source(x_train, y_train, x_valid, y_valid, epochs=12)
        adapter = TorchTabularModelAdapter(model, model_version=config.model_version)
        reference_batches = _build_reference_batches(
            x_valid,
            y_valid,
            batch_size=config.replay.batch_size,
            seed=7,
        )
        return build_reliability_layer_from_reference_batches(adapter, reference_batches, config=config)

    return _builder


def run_customer_replay(spec: CustomerReplaySpec) -> CustomerReplayResult:
    """Replay a customer ingest file and write pilot-grade artifacts."""

    events = _load_customer_events(spec.input_path)
    if not events:
        raise ValueError(f"No events loaded from {spec.input_path}")
    stream = events_to_replay_stream(events)

    config = load_runtime_config(spec.config_path)
    config = _apply_overrides(
        config,
        batch_size=spec.batch_size,
        label_delay_steps=spec.label_delay_steps,
    )
    builder = spec.layer_builder or _customer_stream_layer_builder(stream)
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "replay_schema.md").write_text(
        render_replay_schema_markdown(config.replay.feature_prefix),
        encoding="utf-8",
    )
    (output / "customer_manifest.json").write_text(
        json.dumps(
            {
                "customer_label": spec.customer_label,
                "input_path": str(spec.input_path),
                "config_path": str(spec.config_path),
                "stream_records": len(stream.records),
                "feature_columns": list(stream.feature_columns),
                "operating_mode": config.operating_mode.value,
                "policy": config.policy.name,
                "label_delay_steps": config.replay.label_delay_steps,
                "batch_size": config.replay.batch_size,
                "dual_mode": spec.dual_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifact_paths: list[str] = [
        str(output / "replay_schema.md"),
        str(output / "customer_manifest.json"),
    ]

    utility_delta = None
    risk_reduction = None
    buyer_dict: dict[str, Any] | None = None

    if spec.dual_mode:
        payload = run_dual_mode_replay(
            stream,
            runtime_config=config,
            layer_builder=builder,
        )
        artifacts = write_dual_metric_artifacts(
            payload,
            output,
            source_label=f"{spec.customer_label}:{spec.input_path.name}",
        )
        artifact_paths.extend(str(path) for path in artifacts.values())
        bounded = payload["modes"]["bounded_auto"]
        replay = bounded["replay"]
        utility_delta = replay.controller_vs_frozen_utility_delta
        risk_reduction = replay.controller_vs_frozen_risk_reduction
        buyer = compute_buyer_kpis(replay)
        if buyer is not None:
            buyer_dict = buyer.__dict__
            buyer_path = output / "buyer_report.md"
            buyer_path.write_text(
                render_buyer_replay_report(
                    replay,
                    source_label=spec.customer_label,
                    wedge=spec.wedge,
                ),
                encoding="utf-8",
            )
            artifact_paths.append(str(buyer_path))
    else:
        from .engine import run_offline_replay_comparison

        result = run_offline_replay_comparison(
            stream,
            runtime_config=config,
            layer_builder=builder,
            controller_name=config.policy.name,
        )
        (output / "technical_report.md").write_text(render_replay_report(result), encoding="utf-8")
        (output / "operator_report.md").write_text(
            render_operator_replay_report(result, controller_name=config.policy.name),
            encoding="utf-8",
        )
        buyer_path = output / "buyer_report.md"
        buyer_path.write_text(
            render_buyer_replay_report(
                result,
                source_label=spec.customer_label,
                wedge=spec.wedge,
            ),
            encoding="utf-8",
        )
        artifact_paths.extend(
            [
                str(output / "technical_report.md"),
                str(output / "operator_report.md"),
                str(buyer_path),
            ]
        )
        utility_delta = result.controller_vs_frozen_utility_delta
        risk_reduction = result.controller_vs_frozen_risk_reduction
        buyer = compute_buyer_kpis(result, controller_name=config.policy.name)
        if buyer is not None:
            buyer_dict = buyer.__dict__

    return CustomerReplayResult(
        customer_label=spec.customer_label,
        input_path=str(spec.input_path),
        output_dir=str(output),
        dual_mode=spec.dual_mode,
        artifact_paths=tuple(artifact_paths),
        buyer_kpis=buyer_dict,
        controller_vs_frozen_utility_delta=utility_delta,
        controller_vs_frozen_risk_reduction=risk_reduction,
    )
