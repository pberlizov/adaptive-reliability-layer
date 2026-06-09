from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from ..runtime.config import RuntimeConfig, load_runtime_config
from ..runtime.types import OperatingMode
from .discrimination_metrics import (
    FraudCostConfig,
    MetricSpread,
    StrategyDiscriminationMetrics,
    benchmark_has_headroom,
    compute_metric_spreads,
    summarize_strategy_discrimination,
)
from .engine import run_replay_on_stream
from .failure_analysis import _force_passive_actions
from .real_data import REAL_DATA_LOADERS, RealDataBundle, load_real_data_bundle


@dataclass(frozen=True)
class DiscriminationSourceSpec:
    id: str
    tier: str = "discrimination"
    description: str = ""
    steps: int | None = None
    batch_size: int | None = None
    stream_cycles: int = 1
    label_delay_steps: int | None = None
    label_delay_jitter_steps: int | None = None
    apply_synthetic_shift: bool | None = None
    temporal_split: bool | None = None
    test_fraction: float | None = None
    stream_tail_fraction: float | None = None


@dataclass(frozen=True)
class DiscriminationBenchmarkSpec:
    controller_name: str = "regime_aware_delayed_bandit"
    strategies: tuple[str, ...] = (
        "frozen",
        "scheduled_retrain",
        "naive",
        "regime_aware_delayed_bandit",
    )
    min_rankable_metrics: int = 2
    min_metric_spread: float = 0.005
    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0
    sources: tuple[DiscriminationSourceSpec, ...] = ()


@dataclass(frozen=True)
class DiscriminationSourceResult:
    source_id: str
    tier: str
    description: str
    stream_records: int
    label_delay_steps: int
    has_headroom: bool
    headroom_detail: str
    rankable_metric_count: int
    strategy_metrics: tuple[StrategyDiscriminationMetrics, ...]
    metric_spreads: tuple[MetricSpread, ...]


@dataclass(frozen=True)
class DiscriminationBenchmarkReport:
    config_path: str
    controller_name: str
    sources: tuple[DiscriminationSourceResult, ...]
    rankable_sources: int


def load_discrimination_benchmark_spec(
    config_path: str | Path,
) -> tuple[RuntimeConfig, DiscriminationBenchmarkSpec]:
    runtime = load_runtime_config(str(config_path))
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    benchmark = payload.get("discrimination_benchmark") or {}
    sources = tuple(
        DiscriminationSourceSpec(
            id=item["id"],
            tier=item.get("tier", "discrimination"),
            description=item.get("description", ""),
            steps=item.get("steps"),
            batch_size=item.get("batch_size"),
            stream_cycles=int(item.get("stream_cycles", 1)),
            label_delay_steps=item.get("label_delay_steps"),
            label_delay_jitter_steps=item.get("label_delay_jitter_steps"),
            apply_synthetic_shift=item.get("apply_synthetic_shift"),
            temporal_split=item.get("temporal_split"),
            test_fraction=item.get("test_fraction"),
            stream_tail_fraction=item.get("stream_tail_fraction"),
        )
        for item in benchmark.get("sources", [])
    )
    spec = DiscriminationBenchmarkSpec(
        controller_name=benchmark.get("controller_name", runtime.policy.name),
        strategies=tuple(benchmark.get("strategies", ("frozen", runtime.policy.name))),
        min_rankable_metrics=int(benchmark.get("min_rankable_metrics", 2)),
        min_metric_spread=float(benchmark.get("min_metric_spread", 0.005)),
        false_negative_cost=float(benchmark.get("false_negative_cost", 10.0)),
        false_positive_cost=float(benchmark.get("false_positive_cost", 1.0)),
        sources=sources,
    )
    return runtime, spec


def _resolved_loader(source_id: str) -> Callable[..., RealDataBundle]:
    return REAL_DATA_LOADERS[source_id]


def _bundle_kwargs(loader: Callable[..., RealDataBundle], source: DiscriminationSourceSpec) -> dict[str, Any]:
    candidate: dict[str, Any] = {"stream_cycles": source.stream_cycles}
    for key, value in (
        ("steps", source.steps),
        ("batch_size", source.batch_size),
        ("apply_synthetic_shift", source.apply_synthetic_shift),
        ("temporal_split", source.temporal_split),
        ("test_fraction", source.test_fraction),
        ("stream_tail_fraction", source.stream_tail_fraction),
    ):
        if value is not None:
            candidate[key] = value
    sig = inspect.signature(loader)
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values())
    return {key: value for key, value in candidate.items() if accepts_var_kwargs or key in sig.parameters}


_CORRECTION_PATH_STRATEGIES: dict[str, dict[str, Any]] = {
    "correction_only": {
        "layer_mutator": _force_passive_actions,
        "bounded_actions": None,
    },
    "correction_plus_governor": {
        "layer_mutator": None,
        "bounded_actions": frozenset({"none", "hold"}),
    },
}


def _resolve_strategy(
    strategy_name: str,
    *,
    controller_name: str,
) -> tuple[str, str, Callable[[object], None] | None, frozenset[str] | None]:
    if strategy_name in _CORRECTION_PATH_STRATEGIES:
        variant = _CORRECTION_PATH_STRATEGIES[strategy_name]
        return controller_name, strategy_name, variant["layer_mutator"], variant["bounded_actions"]
    return strategy_name, strategy_name, None, None


def _run_strategy(
    *,
    bundle: RealDataBundle,
    runtime_config: RuntimeConfig,
    strategy_name: str,
    controller_name: str,
    label_delay_steps: int,
    label_delay_jitter_steps: int,
    batch_size: int,
    max_steps: int | None,
) -> Any:
    policy_name, run_name, layer_mutator, bounded_actions = _resolve_strategy(
        strategy_name,
        controller_name=controller_name,
    )
    config = replace(
        runtime_config,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        policy=replace(runtime_config.policy, name=policy_name),
        bounded_auto_actions=bounded_actions or runtime_config.bounded_auto_actions,
        replay=replace(
            runtime_config.replay,
            batch_size=batch_size,
            max_steps=max_steps,
            label_delay_steps=label_delay_steps,
            label_delay_jitter_steps=label_delay_jitter_steps,
        ),
        log_json=False,
    )
    layer = bundle.build_layer(config)
    if layer_mutator is not None:
        layer_mutator(layer)
    return run_replay_on_stream(layer, bundle.stream, config=config.replay, name=run_name)


def evaluate_discrimination_source(
    *,
    source: DiscriminationSourceSpec,
    runtime_config: RuntimeConfig,
    spec: DiscriminationBenchmarkSpec,
) -> DiscriminationSourceResult:
    loader = _resolved_loader(source.id)
    bundle = loader(**_bundle_kwargs(loader, source))
    label_delay = source.label_delay_steps or runtime_config.replay.label_delay_steps
    label_jitter = (
        source.label_delay_jitter_steps
        if source.label_delay_jitter_steps is not None
        else runtime_config.replay.label_delay_jitter_steps
    )
    batch_size = source.batch_size or runtime_config.replay.batch_size
    max_steps = source.steps or runtime_config.replay.max_steps
    cost = FraudCostConfig(
        false_positive_cost=spec.false_positive_cost,
        false_negative_cost=spec.false_negative_cost,
    )
    strategy_metrics: list[StrategyDiscriminationMetrics] = []
    for strategy_name in spec.strategies:
        run = _run_strategy(
            bundle=bundle,
            runtime_config=runtime_config,
            strategy_name=strategy_name,
            controller_name=spec.controller_name,
            label_delay_steps=label_delay,
            label_delay_jitter_steps=label_jitter,
            batch_size=batch_size,
            max_steps=max_steps,
        )
        strategy_metrics.append(
            summarize_strategy_discrimination(
                strategy_name,
                bundle.stream,
                run,
                batch_size=batch_size,
                max_steps=max_steps,
                cost=cost,
            )
        )
    spreads = compute_metric_spreads(
        tuple(strategy_metrics),
        min_spread=spec.min_metric_spread,
    )
    frozen = next(item for item in strategy_metrics if item.name == "frozen")
    has_headroom, headroom_detail = benchmark_has_headroom(frozen)
    rankable_count = sum(1 for spread in spreads if spread.rankable)
    return DiscriminationSourceResult(
        source_id=source.id,
        tier=source.tier,
        description=source.description or bundle.description,
        stream_records=bundle.stream_size,
        label_delay_steps=label_delay,
        has_headroom=has_headroom,
        headroom_detail=headroom_detail,
        rankable_metric_count=rankable_count,
        strategy_metrics=tuple(strategy_metrics),
        metric_spreads=spreads,
    )


def run_discrimination_benchmark(
    *,
    config_path: str | Path = "configs/discrimination_benchmark_suite.yaml",
    source_ids: tuple[str, ...] | None = None,
    skip_missing_data: bool = True,
) -> DiscriminationBenchmarkReport:
    runtime_config, spec = load_discrimination_benchmark_spec(config_path)
    selected = spec.sources
    if source_ids:
        wanted = set(source_ids)
        selected = tuple(source for source in spec.sources if source.id in wanted)
    sources: list[DiscriminationSourceResult] = []
    for source in selected:
        try:
            sources.append(
                evaluate_discrimination_source(
                    source=source,
                    runtime_config=runtime_config,
                    spec=spec,
                )
            )
        except FileNotFoundError:
            if not skip_missing_data:
                raise
    rankable_sources = sum(
        1
        for source in sources
        if source.has_headroom and source.rankable_metric_count >= spec.min_rankable_metrics
    )
    return DiscriminationBenchmarkReport(
        config_path=str(config_path),
        controller_name=spec.controller_name,
        sources=tuple(sources),
        rankable_sources=rankable_sources,
    )


def discrimination_report_to_dict(report: DiscriminationBenchmarkReport) -> dict[str, Any]:
    return {
        "config_path": report.config_path,
        "controller_name": report.controller_name,
        "rankable_sources": report.rankable_sources,
        "sources": [
            {
                **asdict(source),
                "strategy_metrics": [asdict(item) for item in source.strategy_metrics],
                "metric_spreads": [asdict(item) for item in source.metric_spreads],
            }
            for source in report.sources
        ],
    }


def render_discrimination_report(report: DiscriminationBenchmarkReport) -> str:
    lines = [
        "# Discrimination benchmark",
        "",
        "Benchmarks chosen to **separate methods** when raw accuracy saturates.",
        "Lead metrics: balanced accuracy, PR-AUC, recall@precision≥0.80, cost-weighted error, late-stream recall.",
        "",
        f"**Controller reference:** `{report.controller_name}`",
        f"**Sources with headroom + rankable metrics:** {report.rankable_sources}/{len(report.sources)}",
        "",
    ]
    for source in report.sources:
        lines.extend(
            [
                f"## {source.source_id}",
                source.description,
                f"- stream records: **{source.stream_records}**",
                f"- label delay: **{source.label_delay_steps}** steps",
                f"- headroom: **{'yes' if source.has_headroom else 'limited'}** ({source.headroom_detail})",
                f"- rankable metrics: **{source.rankable_metric_count}**",
                "",
                "| strategy | acc | bal_acc | PR-AUC | recall | R@P≥0.8 | cost err | late recall | retrain rec |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in source.strategy_metrics:
            metrics = item.stream_metrics
            pr_auc = "n/a" if metrics.pr_auc is None else f"{metrics.pr_auc:.3f}"
            r_at_p = (
                "n/a"
                if metrics.recall_at_precision_80 is None
                else f"{metrics.recall_at_precision_80:.3f}"
            )
            lines.append(
                f"| `{item.name}` | {metrics.accuracy:.3f} | {metrics.balanced_accuracy:.3f} | "
                f"{pr_auc} | {metrics.recall:.3f} | {r_at_p} | {metrics.cost_weighted_error:.3f} | "
                f"{item.temporal_halves.second_half.recall:.3f} | {item.mean_retrain_recommendation_rate:.3f} |"
            )
        lines.append("")
        lines.append("### Metric spread (can we tell methods apart?)")
        lines.append("")
        lines.append("| metric | spread | rankable | values |")
        lines.append("| --- | ---: | ---: | --- |")
        for spread in source.metric_spreads:
            spread_text = "n/a" if spread.spread is None else f"{spread.spread:.4f}"
            values = ", ".join(
                f"{name}={value:.3f}" if value is not None else f"{name}=n/a"
                for name, value in spread.values_by_strategy.items()
            )
            lines.append(
                f"| {spread.metric_name} | {spread_text} | {'yes' if spread.rankable else 'no'} | {values} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_discrimination_artifacts(
    report: DiscriminationBenchmarkReport,
    *,
    output_dir: str | Path = "results/discrimination_benchmark",
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "discrimination_report.md").write_text(render_discrimination_report(report), encoding="utf-8")
    (root / "discrimination_report.json").write_text(
        json.dumps(discrimination_report_to_dict(report), indent=2),
        encoding="utf-8",
    )
    return root
