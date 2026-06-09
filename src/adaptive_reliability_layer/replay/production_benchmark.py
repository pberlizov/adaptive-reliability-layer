from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ..runtime.config import RuntimeConfig
from ..runtime.types import OperatingMode
from .buyer_kpis import compute_buyer_kpis
from .dual_metric import run_dual_mode_replay, write_dual_metric_artifacts
from .real_data import REAL_DATA_LOADERS, RealDataBundle, load_real_data_bundle
from .report import (
    ReplayComparisonResult,
    StrategyReplaySummary,
    adaptation_safety_rate,
    utility_delta_vs_baseline,
)


@dataclass(frozen=True)
class ProductionEvidenceThresholds:
    min_utility_delta: float = 0.005
    min_risk_reduction_pct: float = 5.0
    min_stream_records: int = 20_000
    stretch_utility_delta: float = 0.01
    stretch_risk_reduction_pct: float = 10.0
    min_adaptation_safety_rate: float = 0.85


@dataclass(frozen=True)
class ProductionSourceSpec:
    id: str
    tier: str = "core"
    description: str = ""
    steps: int | None = None
    batch_size: int | None = None
    stream_cycles: int = 1
    label_delay_steps: int | None = None
    label_delay_jitter_steps: int | None = None
    apply_synthetic_shift: bool | None = None
    temporal_split: bool | None = None


@dataclass(frozen=True)
class ProductionBenchmarkSpec:
    controller_name: str = "regime_aware_delayed_bandit"
    comparison_controller: str | None = None
    strategies: tuple[str, ...] = ("frozen", "regime_aware_delayed_bandit")
    baseline_strategies: tuple[str, ...] = ("scheduled_retrain", "naive")
    min_core_sources_passing: int = 2
    require_beat_baselines: bool = False
    beat_baselines_min_delta: float = 0.0
    evidence: ProductionEvidenceThresholds = ProductionEvidenceThresholds()
    sources: tuple[ProductionSourceSpec, ...] = ()


@dataclass(frozen=True)
class ProductionSourceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ProductionSourceResult:
    source_id: str
    tier: str
    description: str
    stream_records: int
    label_delay_steps: int
    passed: bool
    checks: tuple[ProductionSourceCheck, ...]
    utility_delta: float | None
    risk_reduction_pct: float | None
    harmful_events_avoided: int | None
    intervention_rate: float
    recommendation_execution_rate: float
    correction_applied_rate: float
    correction_only_rate: float
    mean_correction_flipped_predictions: float
    mean_abs_threshold_shift: float
    stretch_passed: bool
    dataset_path: str | None = None
    baseline_utility_deltas: tuple[tuple[str, float | None], ...] = ()
    adaptation_safety_rate: float | None = None
    temporal_split: bool = False
    comparison_utility_delta: float | None = None
    strategy_utilities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ProductionBenchmarkReport:
    spec: ProductionBenchmarkSpec
    sources: tuple[ProductionSourceResult, ...]
    core_sources_passing: int
    suite_passed: bool
    stretch_any: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_passed": self.suite_passed,
            "core_sources_passing": self.core_sources_passing,
            "stretch_any": self.stretch_any,
            "min_core_sources_passing": self.spec.min_core_sources_passing,
            "controller_name": self.spec.controller_name,
            "baseline_strategies": list(self.spec.baseline_strategies),
            "sources": [
                {
                    "source_id": source.source_id,
                    "tier": source.tier,
                    "passed": source.passed,
                    "stretch_passed": source.stretch_passed,
                    "utility_delta": source.utility_delta,
                    "risk_reduction_pct": source.risk_reduction_pct,
                    "stream_records": source.stream_records,
                    "label_delay_steps": source.label_delay_steps,
                    "dataset_path": source.dataset_path,
                    "temporal_split": source.temporal_split,
                    "adaptation_safety_rate": source.adaptation_safety_rate,
                    "baseline_utility_deltas": dict(source.baseline_utility_deltas),
                    "correction_applied_rate": source.correction_applied_rate,
                    "correction_only_rate": source.correction_only_rate,
                    "mean_correction_flipped_predictions": source.mean_correction_flipped_predictions,
                    "mean_abs_threshold_shift": source.mean_abs_threshold_shift,
                    "checks": [asdict(check) for check in source.checks],
                }
                for source in self.sources
            ],
        }


def load_production_benchmark_spec(config_path: str | Path) -> tuple[RuntimeConfig, ProductionBenchmarkSpec]:
    import yaml

    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    runtime = RuntimeConfig.from_mapping(payload)
    bench = payload.get("production_benchmark", {})
    evidence_raw = bench.get("evidence", {})
    evidence = ProductionEvidenceThresholds(
        min_utility_delta=float(evidence_raw.get("min_utility_delta", 0.005)),
        min_risk_reduction_pct=float(evidence_raw.get("min_risk_reduction_pct", 5.0)),
        min_stream_records=int(evidence_raw.get("min_stream_records", 20_000)),
        stretch_utility_delta=float(evidence_raw.get("stretch_utility_delta", 0.01)),
        stretch_risk_reduction_pct=float(evidence_raw.get("stretch_risk_reduction_pct", 10.0)),
        min_adaptation_safety_rate=float(evidence_raw.get("min_adaptation_safety_rate", 0.85)),
    )
    sources = tuple(
        ProductionSourceSpec(
            id=str(item["id"]),
            tier=str(item.get("tier", "core")),
            description=str(item.get("description", "")),
            steps=item.get("steps"),
            batch_size=item.get("batch_size"),
            stream_cycles=int(item.get("stream_cycles", 1)),
            label_delay_steps=item.get("label_delay_steps"),
            label_delay_jitter_steps=item.get("label_delay_jitter_steps"),
            apply_synthetic_shift=item.get("apply_synthetic_shift"),
            temporal_split=item.get("temporal_split"),
        )
        for item in payload.get("sources", [])
    )
    comparison = bench.get("comparison_controller")
    spec = ProductionBenchmarkSpec(
        controller_name=str(bench.get("controller_name", "regime_aware_delayed_bandit")),
        comparison_controller=str(comparison) if comparison else None,
        strategies=tuple(bench.get("strategies", ("frozen", "regime_aware_delayed_bandit"))),
        baseline_strategies=tuple(bench.get("baseline_strategies", ("scheduled_retrain", "naive"))),
        min_core_sources_passing=int(bench.get("min_core_sources_passing", 2)),
        require_beat_baselines=bool(bench.get("require_beat_baselines", False)),
        beat_baselines_min_delta=float(bench.get("beat_baselines_min_delta", 0.0)),
        evidence=evidence,
        sources=sources,
    )
    return runtime, spec


def _controller_summary(
    replay: ReplayComparisonResult,
    controller_name: str,
) -> StrategyReplaySummary | None:
    return next((item for item in replay.summaries if item.name == controller_name), None)


def _intervention_rate(replay: ReplayComparisonResult, controller_name: str) -> float:
    controller = _controller_summary(replay, controller_name)
    if controller is None or controller.steps <= 0:
        return 0.0
    return controller.intervention_rate


def _recommendation_execution_rate(replay: ReplayComparisonResult, controller_name: str) -> float:
    controller = _controller_summary(replay, controller_name)
    if controller is None:
        return 0.0
    return controller.recommendation_execution_rate


def _correction_applied_rate(replay: ReplayComparisonResult, controller_name: str) -> float:
    controller = _controller_summary(replay, controller_name)
    if controller is None:
        return 0.0
    return controller.correction_applied_rate


def _correction_only_rate(replay: ReplayComparisonResult, controller_name: str) -> float:
    controller = _controller_summary(replay, controller_name)
    if controller is None:
        return 0.0
    return controller.correction_only_rate


def _mean_correction_flips(replay: ReplayComparisonResult, controller_name: str) -> float:
    controller = _controller_summary(replay, controller_name)
    if controller is None:
        return 0.0
    return controller.mean_correction_flipped_predictions


def _mean_abs_threshold_shift(replay: ReplayComparisonResult, controller_name: str) -> float:
    controller = _controller_summary(replay, controller_name)
    if controller is None:
        return 0.0
    return controller.mean_abs_threshold_shift


def _strategy_mean_utility(replay: ReplayComparisonResult, strategy_name: str) -> float | None:
    summary = next((item for item in replay.summaries if item.name == strategy_name), None)
    if summary is None:
        return None
    return summary.mean_utility


def _collect_head_to_head_metrics(
    replay: ReplayComparisonResult,
    *,
    primary: str,
    comparison: str,
    baseline_strategies: tuple[str, ...],
) -> tuple[float | None, tuple[tuple[str, float], ...]]:
    primary_u = _strategy_mean_utility(replay, primary)
    comparison_u = _strategy_mean_utility(replay, comparison)
    delta = None
    if primary_u is not None and comparison_u is not None:
        delta = primary_u - comparison_u
    names = ("frozen", *baseline_strategies, comparison, primary)
    utilities: list[tuple[str, float]] = []
    for name in names:
        value = _strategy_mean_utility(replay, name)
        if value is not None:
            utilities.append((name, value))
    return delta, tuple(utilities)


def _merged_strategies(spec: ProductionBenchmarkSpec) -> tuple[str, ...]:
    merged: list[str] = []
    for name in (*spec.strategies, *spec.baseline_strategies):
        if name not in merged:
            merged.append(name)
    if spec.controller_name not in merged:
        merged.append(spec.controller_name)
    if spec.comparison_controller and spec.comparison_controller not in merged:
        merged.append(spec.comparison_controller)
    if "frozen" not in merged:
        merged.insert(0, "frozen")
    return tuple(merged)


def evaluate_source_evidence(
    *,
    source: ProductionSourceSpec,
    bundle: RealDataBundle,
    dual_payload: dict,
    thresholds: ProductionEvidenceThresholds,
    controller_name: str,
    baseline_strategies: tuple[str, ...],
    require_beat_baselines: bool,
    beat_baselines_min_delta: float = 0.0,
) -> ProductionSourceResult:
    bounded = dual_payload["modes"]["bounded_auto"]
    replay: ReplayComparisonResult = bounded["replay"]
    utility_delta = replay.controller_vs_frozen_utility_delta
    risk_reduction = replay.controller_vs_frozen_risk_reduction
    risk_reduction_pct = (risk_reduction or 0.0) * 100.0
    harmful_events = replay.controller_vs_frozen_harmful_events_avoided
    label_delay = dual_payload.get("label_delay_steps", 0)

    baseline_deltas = tuple(
        (baseline, utility_delta_vs_baseline(replay, controller_name=controller_name, baseline_name=baseline))
        for baseline in baseline_strategies
    )

    controller_timeline = next(
        (timeline for timeline in replay.timelines if timeline.name == controller_name),
        None,
    )
    safety_rate = adaptation_safety_rate(controller_timeline)
    scheduled_delta = dict(baseline_deltas).get("scheduled_retrain")

    checks: list[ProductionSourceCheck] = []
    checks.append(
        ProductionSourceCheck(
            "stream_size",
            bundle.stream_size >= thresholds.min_stream_records,
            f"{bundle.stream_size} records (need >= {thresholds.min_stream_records})",
        )
    )
    utility_ok = utility_delta is not None and utility_delta >= thresholds.min_utility_delta
    risk_ok = risk_reduction is not None and risk_reduction_pct >= thresholds.min_risk_reduction_pct
    checks.append(
        ProductionSourceCheck(
            "economic_win",
            utility_ok or risk_ok,
            f"utility_delta={utility_delta!r}, risk_reduction={risk_reduction_pct:.1f}% "
            f"(need utility>={thresholds.min_utility_delta} OR risk>={thresholds.min_risk_reduction_pct}%)",
        )
    )
    if require_beat_baselines:
        for baseline_name, baseline_delta in baseline_deltas:
            checks.append(
                ProductionSourceCheck(
                    f"beat_{baseline_name}",
                    baseline_delta is not None and baseline_delta >= beat_baselines_min_delta,
                    f"utility_delta_vs_{baseline_name}={baseline_delta!r} (need >= {beat_baselines_min_delta})",
                )
            )
    if safety_rate is not None:
        safety_waived = (
            scheduled_delta is not None
            and scheduled_delta >= thresholds.min_utility_delta
            and safety_rate >= 0.65
        )
        checks.append(
            ProductionSourceCheck(
                "adaptation_safety",
                safety_rate >= thresholds.min_adaptation_safety_rate or safety_waived,
                f"adaptation_safety_rate={safety_rate:.3f} (need >= {thresholds.min_adaptation_safety_rate}"
                + (
                    f", waived: beats scheduled_retrain by {scheduled_delta:+.3f})"
                    if safety_waived and safety_rate < thresholds.min_adaptation_safety_rate
                    else ")"
                ),
            )
        )
    intervention_rate = _intervention_rate(replay, controller_name)
    exec_rate = _recommendation_execution_rate(replay, controller_name)
    correction_rate = _correction_applied_rate(replay, controller_name)
    correction_only_rate = _correction_only_rate(replay, controller_name)
    mean_correction_flips = _mean_correction_flips(replay, controller_name)
    mean_abs_threshold_shift = _mean_abs_threshold_shift(replay, controller_name)
    controller = _controller_summary(replay, controller_name)
    recommendation_rate = controller.recommendation_rate if controller else 0.0
    if source.tier == "core":
        policy_active = (
            intervention_rate > 0.0
            or exec_rate > 0.0
            or recommendation_rate > 0.05
            or correction_rate > 0.05
            or mean_correction_flips > 0.0
            or mean_abs_threshold_shift > 1e-6
        )
        activity_detail = (
            f"intervention_rate={intervention_rate:.3f}, recommendation_exec_rate={exec_rate:.3f}, "
            f"recommendation_rate={recommendation_rate:.3f}, correction_rate={correction_rate:.3f}, "
            f"correction_only_rate={correction_only_rate:.3f}, mean_correction_flips={mean_correction_flips:.3f}, "
            f"mean_abs_threshold_shift={mean_abs_threshold_shift:.4f}"
        )
        checks.append(
            ProductionSourceCheck(
                "policy_activity",
                policy_active,
                activity_detail,
            )
        )

    stretch_passed = (
        utility_delta is not None
        and utility_delta >= thresholds.stretch_utility_delta
        and risk_reduction is not None
        and risk_reduction_pct >= thresholds.stretch_risk_reduction_pct
    )
    passed = all(check.passed for check in checks)
    return ProductionSourceResult(
        source_id=source.id,
        tier=source.tier,
        description=source.description or bundle.description,
        stream_records=bundle.stream_size,
        label_delay_steps=int(label_delay),
        passed=passed,
        checks=tuple(checks),
        utility_delta=utility_delta,
        risk_reduction_pct=risk_reduction_pct,
        harmful_events_avoided=harmful_events,
        intervention_rate=intervention_rate,
        recommendation_execution_rate=exec_rate,
        correction_applied_rate=correction_rate,
        correction_only_rate=correction_only_rate,
        mean_correction_flipped_predictions=mean_correction_flips,
        mean_abs_threshold_shift=mean_abs_threshold_shift,
        stretch_passed=stretch_passed,
        dataset_path=bundle.dataset_path,
        baseline_utility_deltas=baseline_deltas,
        adaptation_safety_rate=safety_rate,
        temporal_split=bool(dual_payload.get("temporal_split", False)),
    )


def run_production_source_benchmark(
    *,
    source: ProductionSourceSpec,
    runtime_config: RuntimeConfig,
    spec: ProductionBenchmarkSpec,
    bundle_loader: Callable[..., RealDataBundle] | None = None,
) -> tuple[RealDataBundle, dict, ProductionSourceResult]:
    loader = bundle_loader or load_real_data_bundle
    candidate: dict[str, Any] = {"stream_cycles": source.stream_cycles}
    if source.steps is not None:
        candidate["steps"] = source.steps
    if source.batch_size is not None:
        candidate["batch_size"] = source.batch_size
    if source.apply_synthetic_shift is not None:
        candidate["apply_synthetic_shift"] = source.apply_synthetic_shift
    if source.temporal_split is not None:
        candidate["temporal_split"] = source.temporal_split

    resolved_loader = loader
    if loader is load_real_data_bundle:
        resolved_loader = REAL_DATA_LOADERS[source.id]

    sig = inspect.signature(resolved_loader)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    )
    kwargs = {
        key: value
        for key, value in candidate.items()
        if accepts_var_kwargs or key in sig.parameters
    }
    bundle = resolved_loader(**kwargs)

    replay_config = runtime_config.replay
    if source.label_delay_steps is not None:
        replay_config = replace(replay_config, label_delay_steps=source.label_delay_steps)
    if source.label_delay_jitter_steps is not None:
        replay_config = replace(replay_config, label_delay_jitter_steps=source.label_delay_jitter_steps)
    if source.steps is not None:
        replay_config = replace(replay_config, max_steps=source.steps)
    if source.batch_size is not None:
        replay_config = replace(replay_config, batch_size=source.batch_size)

    source_config = replace(
        runtime_config,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        replay=replay_config,
        policy=replace(runtime_config.policy, name=spec.controller_name),
    )

    strategies = _merged_strategies(spec)
    dual = run_dual_mode_replay(
        bundle.stream,
        runtime_config=source_config,
        layer_builder=bundle.build_layer,
        strategies=strategies,
        controller_name=spec.controller_name,
    )
    dual["label_delay_steps"] = replay_config.label_delay_steps
    dual["temporal_split"] = bool(source.temporal_split)
    result = evaluate_source_evidence(
        source=source,
        bundle=bundle,
        dual_payload=dual,
        thresholds=spec.evidence,
        controller_name=spec.controller_name,
        baseline_strategies=spec.baseline_strategies,
        require_beat_baselines=spec.require_beat_baselines,
        beat_baselines_min_delta=spec.beat_baselines_min_delta,
    )
    comparison_delta = None
    strategy_utilities: tuple[tuple[str, float], ...] = ()
    if spec.comparison_controller:
        bounded_replay: ReplayComparisonResult = dual["modes"]["bounded_auto"]["replay"]
        comparison_delta, strategy_utilities = _collect_head_to_head_metrics(
            bounded_replay,
            primary=spec.controller_name,
            comparison=spec.comparison_controller,
            baseline_strategies=spec.baseline_strategies,
        )
        result = replace(
            result,
            comparison_utility_delta=comparison_delta,
            strategy_utilities=strategy_utilities,
        )
    return bundle, dual, result


def run_production_benchmark_suite(
    *,
    runtime_config: RuntimeConfig,
    spec: ProductionBenchmarkSpec,
    output_dir: str | Path,
) -> ProductionBenchmarkReport:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_results: list[ProductionSourceResult] = []

    for source in spec.sources:
        bundle, dual, result = run_production_source_benchmark(
            source=source,
            runtime_config=runtime_config,
            spec=spec,
        )
        source_dir = output / source.id
        write_dual_metric_artifacts(
            dual,
            source_dir,
            source_label=f"{source.id} ({source.tier})",
            stream_records=bundle.stream_size,
            label_delay_steps=dual.get("label_delay_steps"),
        )
        buyer = compute_buyer_kpis(dual["modes"]["bounded_auto"]["replay"], controller_name=spec.controller_name)
        if buyer is not None:
            (source_dir / "buyer_summary.txt").write_text(
                "\n".join([buyer.headline, buyer.risk_sentence, buyer.accuracy_sentence, buyer.operations_sentence]),
                encoding="utf-8",
            )
        baseline_lines = [
            f"{name}: {delta!r}" for name, delta in result.baseline_utility_deltas
        ]
        (source_dir / "baseline_comparison.txt").write_text(
            "\n".join(
                [
                    f"controller: {spec.controller_name}",
                    f"utility_delta_vs_frozen: {result.utility_delta!r}",
                    *baseline_lines,
                    f"adaptation_safety_rate: {result.adaptation_safety_rate!r}",
                    f"temporal_split: {result.temporal_split}",
                ]
            ),
            encoding="utf-8",
        )
        if spec.comparison_controller and result.comparison_utility_delta is not None:
            (source_dir / "head_to_head.txt").write_text(
                "\n".join(
                    [
                        f"primary ({spec.controller_name}) vs {spec.comparison_controller}: "
                        f"{result.comparison_utility_delta:+.4f} utility",
                        "strategy mean_utility:",
                        *[f"  {name}: {utility:.4f}" for name, utility in result.strategy_utilities],
                    ]
                ),
                encoding="utf-8",
            )
        source_results.append(result)

    core_passing = sum(1 for item in source_results if item.tier == "core" and item.passed)
    suite_passed = core_passing >= spec.min_core_sources_passing
    stretch_any = any(item.stretch_passed for item in source_results)
    report = ProductionBenchmarkReport(
        spec=spec,
        sources=tuple(source_results),
        core_sources_passing=core_passing,
        suite_passed=suite_passed,
        stretch_any=stretch_any,
    )
    (output / "suite_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (output / "suite_report.md").write_text(render_production_benchmark_report(report), encoding="utf-8")
    if spec.comparison_controller:
        (output / "head_to_head_report.md").write_text(
            render_head_to_head_report(report),
            encoding="utf-8",
        )
    return report


def render_head_to_head_report(report: ProductionBenchmarkReport) -> str:
    primary = report.spec.controller_name
    comparison = report.spec.comparison_controller
    if comparison is None:
        return "# Head-to-head report\n\nNo comparison_controller configured.\n"

    wins = losses = ties = 0
    lines = [
        "# Controller head-to-head (same torch + temporal split + baselines)",
        "",
        f"**Primary:** `{primary}` | **Comparison:** `{comparison}`",
        "",
        "| source | utility (primary) | utility (comparison) | Δ primary − comparison | winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for source in report.sources:
        utilities = dict(source.strategy_utilities)
        primary_u = utilities.get(primary)
        comparison_u = utilities.get(comparison)
        delta = source.comparison_utility_delta
        if delta is None:
            winner = "n/a"
        elif delta > 0.0005:
            winner = primary
            wins += 1
        elif delta < -0.0005:
            winner = comparison
            losses += 1
        else:
            winner = "tie"
            ties += 1
        primary_text = "n/a" if primary_u is None else f"{primary_u:.3f}"
        comparison_text = "n/a" if comparison_u is None else f"{comparison_u:.3f}"
        delta_text = "n/a" if delta is None else f"{delta:+.3f}"
        lines.append(
            f"| {source.source_id} | {primary_text} | {comparison_text} | {delta_text} | {winner} |"
        )
    lines.extend(
        [
            "",
            f"**Score:** {primary} wins {wins}, {comparison} wins {losses}, ties {ties} "
            f"(of {len(report.sources)} sources)",
            "",
            "All strategies per source (mean utility, bounded_auto):",
        ]
    )
    for source in report.sources:
        lines.append(f"- **{source.source_id}**:")
        for name, utility in source.strategy_utilities:
            lines.append(f"  - `{name}`: {utility:.3f}")
    return "\n".join(lines) + "\n"


def render_production_benchmark_report(report: ProductionBenchmarkReport) -> str:
    lines = [
        "# Production benchmark suite",
        "",
        f"**Suite passed:** {'yes' if report.suite_passed else 'no'} "
        f"({report.core_sources_passing} core sources pass, need {report.spec.min_core_sources_passing})",
        "",
        f"Controller: `{report.spec.controller_name}`",
        f"Baselines: {', '.join(f'`{name}`' for name in report.spec.baseline_strategies)}",
        "",
        "| source | tier | pass | utility Δ frozen | vs scheduled | vs naive | risk ↓ | safety | exec | corr | temporal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in report.sources:
        utility = "n/a" if source.utility_delta is None else f"{source.utility_delta:+.3f}"
        risk = f"{source.risk_reduction_pct:.1f}%"
        safety = "n/a" if source.adaptation_safety_rate is None else f"{source.adaptation_safety_rate:.0%}"
        baseline_map = dict(source.baseline_utility_deltas)
        vs_scheduled = baseline_map.get("scheduled_retrain")
        vs_naive = baseline_map.get("naive")
        scheduled_text = "n/a" if vs_scheduled is None else f"{vs_scheduled:+.3f}"
        naive_text = "n/a" if vs_naive is None else f"{vs_naive:+.3f}"
        lines.append(
            f"| {source.source_id} | {source.tier} | {'PASS' if source.passed else 'FAIL'} | "
            f"{utility} | {scheduled_text} | {naive_text} | {risk} | {safety} | "
            f"{source.recommendation_execution_rate:.1%} | {source.correction_applied_rate:.1%} | "
            f"{'yes' if source.temporal_split else 'no'} |"
        )
        for check in source.checks:
            if not check.passed:
                lines.append(f"| ↳ {check.name} | | | | | | | | | | `{check.detail}` |")
        if source.dataset_path:
            lines.append(f"| ↳ dataset | | | | | | | | | | `{source.dataset_path}` |")
    lines.extend(
        [
            "",
            "## Thresholds",
            f"- min utility delta vs frozen: {report.spec.evidence.min_utility_delta}",
            f"- min risk reduction: {report.spec.evidence.min_risk_reduction_pct}%",
            f"- min stream records: {report.spec.evidence.min_stream_records}",
            f"- min adaptation safety rate: {report.spec.evidence.min_adaptation_safety_rate}",
            f"- require beat baselines: {report.spec.require_beat_baselines}",
            f"- stretch (marketing): utility ≥ {report.spec.evidence.stretch_utility_delta}, "
            f"risk ≥ {report.spec.evidence.stretch_risk_reduction_pct}%",
            "",
            "See [production_evidence_bar.md](../docs/production_evidence_bar.md) for interpretation.",
        ]
    )
    return "\n".join(lines) + "\n"
