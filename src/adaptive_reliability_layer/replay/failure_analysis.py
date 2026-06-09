from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ..runtime.config import RuntimeConfig
from ..runtime.types import OperatingMode
from .engine import run_replay_on_stream
from .production_benchmark import ProductionBenchmarkSpec, ProductionSourceSpec, load_production_benchmark_spec
from .real_data import REAL_DATA_LOADERS, RealDataBundle, load_real_data_bundle
from .report import adaptation_safety_rate, summarize_replay_runs
from .types import ReplayRunState


@dataclass(frozen=True)
class DriverBucketSummary:
    count: int
    revealed_batches: int
    mean_revealed_accuracy: float | None
    mean_revealed_utility: float | None
    mean_correction_delta: float
    mean_flips: float


@dataclass(frozen=True)
class VariantAnalysis:
    name: str
    mean_accuracy: float | None
    mean_utility: float
    recommendation_rate: float
    recommendation_execution_rate: float
    intervention_rate: float
    explicit_action_rate: float
    correction_applied_rate: float
    correction_only_rate: float
    mean_correction_delta: float
    mean_correction_flips: float
    revealed_alert_rate: float | None
    revealed_burden: float | None
    revealed_event_avoids_vs_frozen: int | None
    revealed_event_introductions_vs_frozen: int | None
    mean_revealed_accuracy_delta_vs_frozen: float | None
    adaptation_safety_rate: float | None
    buckets: dict[str, DriverBucketSummary]


@dataclass(frozen=True)
class SourceFailureAnalysis:
    source_id: str
    description: str
    dataset_path: str | None
    validation_accuracy: float
    revealed_accuracy_alert_threshold: float
    variants: dict[str, VariantAnalysis]
    utility_driver: str
    full_vs_no_correction_utility_delta: float
    full_vs_no_explicit_actions_utility_delta: float


@dataclass(frozen=True)
class ProductionFailureAnalysisReport:
    config_path: str
    controller_name: str
    baseline_strategies: tuple[str, ...]
    sources: tuple[SourceFailureAnalysis, ...]


def _resolved_loader(source_id: str) -> Callable[..., RealDataBundle]:
    return REAL_DATA_LOADERS[source_id]


def _bundle_kwargs(loader: Callable[..., RealDataBundle], source: ProductionSourceSpec) -> dict[str, Any]:
    candidate: dict[str, Any] = {"stream_cycles": source.stream_cycles}
    if source.steps is not None:
        candidate["steps"] = source.steps
    if source.batch_size is not None:
        candidate["batch_size"] = source.batch_size
    if source.apply_synthetic_shift is not None:
        candidate["apply_synthetic_shift"] = source.apply_synthetic_shift
    if source.temporal_split is not None:
        candidate["temporal_split"] = source.temporal_split
    sig = inspect.signature(loader)
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values())
    return {key: value for key, value in candidate.items() if accepts_var_kwargs or key in sig.parameters}


def _summary_for_run(run: ReplayRunState):
    result = summarize_replay_runs([run], controller_name=run.name)
    return result.summaries[0], result.timelines[0]


def _disable_probability_correction(layer: object) -> None:
    policy = getattr(layer, "_policy", None)
    if policy is None or not hasattr(policy, "correct_probabilities"):
        return

    def identity(probabilities, **_: object):
        return list(probabilities)

    setattr(policy, "correct_probabilities", identity)


def _force_passive_actions(layer: object) -> None:
    policy = getattr(layer, "_policy", None)
    if policy is None or not hasattr(policy, "apply"):
        return

    original_apply = policy.apply

    def passive_apply(*args: object, **kwargs: object):
        decision = original_apply(*args, **kwargs)
        return replace(
            decision,
            action="none",
            reason=f"{decision.reason};correction_only_passive",
            selected_fraction=0.0,
        )

    setattr(policy, "apply", passive_apply)


def _run_variant(
    *,
    bundle: RealDataBundle,
    runtime_config: RuntimeConfig,
    policy_name: str,
    label: str,
    bounded_actions: frozenset[str] | None = None,
    layer_mutator: Callable[[object], None] | None = None,
) -> ReplayRunState:
    config = replace(
        runtime_config,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        policy=replace(runtime_config.policy, name=policy_name),
        bounded_auto_actions=bounded_actions or runtime_config.bounded_auto_actions,
        log_json=False,
    )
    layer = bundle.build_layer(config)
    if layer_mutator is not None:
        layer_mutator(layer)
    return run_replay_on_stream(layer, bundle.stream, config=config.replay, name=label)


def _revealed_metric_map(run: ReplayRunState) -> dict[int, dict[str, float | str | None]]:
    out: dict[int, dict[str, float | str | None]] = {}
    for item in run.revealed_metrics:
        step = item.get("step")
        if step is None:
            continue
        out[int(float(step))] = item
    return out


def _revealed_alert_metrics(
    run: ReplayRunState,
    *,
    threshold: float,
    baseline: ReplayRunState | None = None,
) -> tuple[float | None, float | None, int | None, int | None, float | None]:
    revealed = _revealed_metric_map(run)
    if not revealed:
        return None, None, None, None, None
    accuracies = [float(item["batch_accuracy"]) for item in revealed.values() if item.get("batch_accuracy") is not None]
    if not accuracies:
        return None, None, None, None, None
    alert_rate = sum(accuracy < threshold for accuracy in accuracies) / max(1, len(accuracies))
    burden = sum(max(0.0, threshold - accuracy) for accuracy in accuracies) / max(1, len(accuracies))

    if baseline is None:
        return alert_rate, burden, None, None, None

    baseline_revealed = _revealed_metric_map(baseline)
    common_steps = sorted(set(revealed) & set(baseline_revealed))
    if not common_steps:
        return alert_rate, burden, None, None, None
    avoided = 0
    introduced = 0
    deltas: list[float] = []
    for step in common_steps:
        baseline_acc = float(baseline_revealed[step]["batch_accuracy"])
        current_acc = float(revealed[step]["batch_accuracy"])
        deltas.append(current_acc - baseline_acc)
        baseline_bad = baseline_acc < threshold
        current_bad = current_acc < threshold
        if baseline_bad and not current_bad:
            avoided += 1
        if current_bad and not baseline_bad:
            introduced += 1
    mean_delta = sum(deltas) / max(1, len(deltas))
    return alert_rate, burden, avoided, introduced, mean_delta


def _bucket_label(surface) -> str:
    if surface.explicit_action_executed:
        return "explicit_action"
    if surface.correction_applied:
        return "correction_only"
    return "passive"


def _bucket_summaries(run: ReplayRunState) -> dict[str, DriverBucketSummary]:
    revealed = _revealed_metric_map(run)
    buckets: dict[str, dict[str, list[float] | int]] = {}
    for surface in run.surfaces:
        label = _bucket_label(surface)
        payload = buckets.setdefault(
            label,
            {
                "count": 0,
                "revealed_batches": 0,
                "revealed_accuracy": [],
                "revealed_utility": [],
                "correction_delta": [],
                "correction_flips": [],
            },
        )
        payload["count"] += 1
        payload["correction_delta"].append(surface.correction_mean_abs_delta)
        payload["correction_flips"].append(float(surface.correction_flipped_predictions))
        metric = revealed.get(surface.step)
        if metric is not None and metric.get("batch_accuracy") is not None:
            payload["revealed_batches"] += 1
            payload["revealed_accuracy"].append(float(metric["batch_accuracy"]))
            if metric.get("utility") is not None:
                payload["revealed_utility"].append(float(metric["utility"]))

    out: dict[str, DriverBucketSummary] = {}
    for label, payload in buckets.items():
        acc = payload["revealed_accuracy"]
        util = payload["revealed_utility"]
        corr = payload["correction_delta"]
        flips = payload["correction_flips"]
        out[label] = DriverBucketSummary(
            count=int(payload["count"]),
            revealed_batches=int(payload["revealed_batches"]),
            mean_revealed_accuracy=(sum(acc) / len(acc)) if acc else None,
            mean_revealed_utility=(sum(util) / len(util)) if util else None,
            mean_correction_delta=(sum(corr) / len(corr)) if corr else 0.0,
            mean_flips=(sum(flips) / len(flips)) if flips else 0.0,
        )
    return out


def _variant_analysis(
    *,
    run: ReplayRunState,
    frozen_run: ReplayRunState,
    threshold: float,
) -> VariantAnalysis:
    summary, timeline = _summary_for_run(run)
    surfaces = run.surfaces
    explicit_action_rate = sum(surface.explicit_action_executed for surface in surfaces) / max(1, len(surfaces))
    correction_applied_rate = sum(surface.correction_applied for surface in surfaces) / max(1, len(surfaces))
    correction_only_rate = sum(
        1 for surface in surfaces if surface.correction_applied and not surface.explicit_action_executed
    ) / max(1, len(surfaces))
    mean_correction_delta = sum(surface.correction_mean_abs_delta for surface in surfaces) / max(1, len(surfaces))
    mean_correction_flips = sum(surface.correction_flipped_predictions for surface in surfaces) / max(1, len(surfaces))
    alert_rate, burden, avoided, introduced, mean_delta = _revealed_alert_metrics(
        run,
        threshold=threshold,
        baseline=frozen_run,
    )
    return VariantAnalysis(
        name=run.name,
        mean_accuracy=summary.mean_accuracy,
        mean_utility=summary.mean_utility,
        recommendation_rate=summary.recommendation_rate,
        recommendation_execution_rate=summary.recommendation_execution_rate,
        intervention_rate=summary.intervention_rate,
        explicit_action_rate=explicit_action_rate,
        correction_applied_rate=correction_applied_rate,
        correction_only_rate=correction_only_rate,
        mean_correction_delta=mean_correction_delta,
        mean_correction_flips=mean_correction_flips,
        revealed_alert_rate=alert_rate,
        revealed_burden=burden,
        revealed_event_avoids_vs_frozen=avoided,
        revealed_event_introductions_vs_frozen=introduced,
        mean_revealed_accuracy_delta_vs_frozen=mean_delta,
        adaptation_safety_rate=adaptation_safety_rate(timeline),
        buckets=_bucket_summaries(run),
    )


def _driver_label(full: VariantAnalysis, no_correction: VariantAnalysis, no_actions: VariantAnalysis) -> tuple[str, float, float]:
    corr_delta = full.mean_utility - no_correction.mean_utility
    action_delta = full.mean_utility - no_actions.mean_utility
    if corr_delta >= 0.005 and corr_delta >= action_delta * 1.25:
        return "correction_dominant", corr_delta, action_delta
    if action_delta >= 0.005 and action_delta >= corr_delta * 1.25:
        return "action_dominant", corr_delta, action_delta
    if corr_delta >= 0.003 and action_delta >= 0.003:
        return "mixed", corr_delta, action_delta
    return "weak_or_unclear", corr_delta, action_delta


def analyze_production_source(
    *,
    source: ProductionSourceSpec,
    runtime_config: RuntimeConfig,
    controller_name: str,
) -> SourceFailureAnalysis:
    loader = _resolved_loader(source.id)
    bundle = loader(**_bundle_kwargs(loader, source))
    threshold = max(0.50, float(bundle.validation_accuracy) - 0.08)

    frozen_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name="frozen",
        label="frozen",
    )
    scheduled_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name="scheduled_retrain",
        label="scheduled_retrain",
    )
    naive_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name="naive",
        label="naive",
    )
    full_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name=controller_name,
        label=controller_name,
    )
    no_correction_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name=controller_name,
        label=f"{controller_name}_no_correction",
        layer_mutator=_disable_probability_correction,
    )
    no_actions_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name=controller_name,
        label=f"{controller_name}_no_explicit_actions",
        bounded_actions=frozenset({"none", "hold"}),
    )
    correction_only_run = _run_variant(
        bundle=bundle,
        runtime_config=runtime_config,
        policy_name=controller_name,
        label=f"{controller_name}_correction_only",
        layer_mutator=_force_passive_actions,
    )

    variants = {
        "frozen": _variant_analysis(run=frozen_run, frozen_run=frozen_run, threshold=threshold),
        "scheduled_retrain": _variant_analysis(run=scheduled_run, frozen_run=frozen_run, threshold=threshold),
        "naive": _variant_analysis(run=naive_run, frozen_run=frozen_run, threshold=threshold),
        "full": _variant_analysis(run=full_run, frozen_run=frozen_run, threshold=threshold),
        "no_correction": _variant_analysis(run=no_correction_run, frozen_run=frozen_run, threshold=threshold),
        "no_explicit_actions": _variant_analysis(run=no_actions_run, frozen_run=frozen_run, threshold=threshold),
        "correction_only": _variant_analysis(run=correction_only_run, frozen_run=frozen_run, threshold=threshold),
    }
    driver, corr_delta, action_delta = _driver_label(
        variants["full"],
        variants["no_correction"],
        variants["no_explicit_actions"],
    )
    return SourceFailureAnalysis(
        source_id=source.id,
        description=source.description or bundle.description,
        dataset_path=bundle.dataset_path,
        validation_accuracy=float(bundle.validation_accuracy),
        revealed_accuracy_alert_threshold=threshold,
        variants=variants,
        utility_driver=driver,
        full_vs_no_correction_utility_delta=corr_delta,
        full_vs_no_explicit_actions_utility_delta=action_delta,
    )


def run_production_failure_analysis(
    *,
    config_path: str | Path = "configs/production_benchmark_sota_suite.yaml",
    source_ids: tuple[str, ...] | None = None,
) -> ProductionFailureAnalysisReport:
    runtime_config, spec = load_production_benchmark_spec(config_path)
    selected_sources = spec.sources
    if source_ids:
        wanted = set(source_ids)
        selected_sources = tuple(source for source in spec.sources if source.id in wanted)
    sources = tuple(
        analyze_production_source(
            source=source,
            runtime_config=runtime_config,
            controller_name=spec.controller_name,
        )
        for source in selected_sources
    )
    return ProductionFailureAnalysisReport(
        config_path=str(config_path),
        controller_name=spec.controller_name,
        baseline_strategies=spec.baseline_strategies,
        sources=sources,
    )


def production_failure_analysis_to_dict(report: ProductionFailureAnalysisReport) -> dict[str, Any]:
    return {
        "config_path": report.config_path,
        "controller_name": report.controller_name,
        "baseline_strategies": list(report.baseline_strategies),
        "sources": [asdict(source) for source in report.sources],
    }


def render_production_failure_analysis(report: ProductionFailureAnalysisReport) -> str:
    lines = [
        "# Production SOTA Failure Analysis",
        "",
        f"Controller: `{report.controller_name}`",
        f"Baselines: {', '.join(f'`{name}`' for name in report.baseline_strategies)}",
        "",
    ]
    for source in report.sources:
        lines.extend(
            [
                f"## {source.source_id}",
                source.description,
                f"- validation accuracy: `{source.validation_accuracy:.3f}`",
                f"- revealed accuracy alert threshold: `{source.revealed_accuracy_alert_threshold:.3f}`",
                f"- dataset: `{source.dataset_path or 'n/a'}`",
                f"- utility driver: `{source.utility_driver}`",
                f"- full vs no-correction utility delta: `{source.full_vs_no_correction_utility_delta:+.3f}`",
                f"- full vs no-explicit-actions utility delta: `{source.full_vs_no_explicit_actions_utility_delta:+.3f}`",
                "",
                "| variant | utility | acc | rec rate | exec rate | explicit | correction | correction-only | revealed alert | revealed burden | avoids | introduces | revealed Δacc | safety |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in ("frozen", "scheduled_retrain", "naive", "full", "correction_only", "no_correction", "no_explicit_actions"):
            variant = source.variants[key]
            acc = "n/a" if variant.mean_accuracy is None else f"{variant.mean_accuracy:.3f}"
            alert = "n/a" if variant.revealed_alert_rate is None else f"{variant.revealed_alert_rate:.3f}"
            burden = "n/a" if variant.revealed_burden is None else f"{variant.revealed_burden:.3f}"
            avoids = "n/a" if variant.revealed_event_avoids_vs_frozen is None else str(variant.revealed_event_avoids_vs_frozen)
            introduces = "n/a" if variant.revealed_event_introductions_vs_frozen is None else str(variant.revealed_event_introductions_vs_frozen)
            delta = "n/a" if variant.mean_revealed_accuracy_delta_vs_frozen is None else f"{variant.mean_revealed_accuracy_delta_vs_frozen:+.3f}"
            safety = "n/a" if variant.adaptation_safety_rate is None else f"{variant.adaptation_safety_rate:.0%}"
            lines.append(
                f"| {key} | {variant.mean_utility:.3f} | {acc} | {variant.recommendation_rate:.3f} | "
                f"{variant.recommendation_execution_rate:.3f} | {variant.explicit_action_rate:.3f} | "
                f"{variant.correction_applied_rate:.3f} | {variant.correction_only_rate:.3f} | {alert} | {burden} | "
                f"{avoids} | {introduces} | {delta} | {safety} |"
            )
        lines.extend(["", "### Bucket attribution", "", "| bucket | count | revealed | acc | utility | corrΔ | flips |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        full = source.variants["full"]
        for bucket_name, bucket in full.buckets.items():
            acc = "n/a" if bucket.mean_revealed_accuracy is None else f"{bucket.mean_revealed_accuracy:.3f}"
            util = "n/a" if bucket.mean_revealed_utility is None else f"{bucket.mean_revealed_utility:.3f}"
            lines.append(
                f"| {bucket_name} | {bucket.count} | {bucket.revealed_batches} | {acc} | {util} | "
                f"{bucket.mean_correction_delta:.4f} | {bucket.mean_flips:.2f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_production_failure_analysis(
    report: ProductionFailureAnalysisReport,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure_analysis.md").write_text(render_production_failure_analysis(report), encoding="utf-8")
    (output / "failure_analysis.json").write_text(
        json.dumps(production_failure_analysis_to_dict(report), indent=2),
        encoding="utf-8",
    )
    return output / "failure_analysis.md"
