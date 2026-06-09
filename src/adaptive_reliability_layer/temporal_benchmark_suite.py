from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np

from .benchmark_suite import AggregateStat
from .temporal_fashion_mnist_benchmark import (
    TemporalFashionBenchmarkResult,
    run_temporal_fashion_mnist_benchmark,
)
from .tabular_benchmark import (
    BanditTabularPolicy,
    ControllerTabularPolicy,
    DelayedHybridBanditSpecialistPolicy,
    DelayedBanditTabularPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    PolicyFactory,
    RegimeAwareDelayedBanditTabularPolicy,
    RoutedDelayedBanditSpecialistPolicy,
)

_SUMMARY_METRICS = (
    "overall_accuracy",
    "served_accuracy",
    "coverage",
    "mean_utility",
    "alerts",
    "risk_alerts",
    "adaptations",
    "resets",
    "abstains",
    "mean_shift_score",
    "mean_risk_capital",
    "mean_reliability",
    "mean_parameter_drift",
)

_TEMPORAL_METRICS = (
    "revealed_accuracy",
    "revealed_coverage",
    "eventual_revealed_accuracy",
    "mean_retro_gap",
)


@dataclass(frozen=True)
class TemporalStrategyAggregate:
    name: str
    metrics: dict[str, AggregateStat]
    regime_accuracy: dict[str, AggregateStat]
    diagnostics: dict[str, AggregateStat]
    temporal_metrics: dict[str, AggregateStat]


@dataclass(frozen=True)
class TemporalBenchmarkAggregate:
    name: str
    severity: str
    reveal_delay_steps: int
    steps: int
    seeds: tuple[int, ...]
    strategies: tuple[TemporalStrategyAggregate, ...]


@dataclass(frozen=True)
class TemporalBenchmarkSuiteResult:
    seeds: tuple[int, ...]
    severities: tuple[str, ...]
    reveal_delays: tuple[int, ...]
    summary: tuple["TemporalStrategySummary", ...]
    benchmarks: tuple[TemporalBenchmarkAggregate, ...]


@dataclass(frozen=True)
class TemporalStrategySummary:
    name: str
    utility_wins: int
    accuracy_wins: int
    mean_utility_margin_vs_frozen: float
    mean_accuracy_margin_vs_frozen: float


def _aggregate(values: list[float]) -> AggregateStat:
    array = np.asarray(values, dtype=np.float64)
    return AggregateStat(mean=float(array.mean()), std=float(array.std(ddof=0)))


def _suite_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        ("delayed_bandit", lambda reference: DelayedBanditTabularPolicy(reference)),
        (
            "regime_aware_delayed_bandit",
            lambda reference: RegimeAwareDelayedBanditTabularPolicy(reference),
        ),
        (
            "routed_delayed_bandit",
            lambda reference: RoutedDelayedBanditSpecialistPolicy(reference, distance_threshold=0.55),
        ),
        ("hybrid", lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.15)),
        (
            "delayed_hybrid",
            lambda reference: DelayedHybridBanditSpecialistPolicy(reference, distance_threshold=0.55),
        ),
    ]


def _aggregate_temporal_results(
    *,
    name: str,
    severity: str,
    reveal_delay_steps: int,
    steps: int,
    seeds: tuple[int, ...],
    results: list[TemporalFashionBenchmarkResult],
) -> TemporalBenchmarkAggregate:
    by_strategy: dict[str, list] = {}
    for result in results:
        for strategy in result.strategies:
            by_strategy.setdefault(strategy.base.name, []).append(strategy)

    strategy_aggregates: list[TemporalStrategyAggregate] = []
    for strategy_name, strategy_runs in sorted(by_strategy.items()):
        metrics = {
            metric_name: _aggregate([float(getattr(strategy.base, metric_name)) for strategy in strategy_runs])
            for metric_name in _SUMMARY_METRICS
        }
        regime_names = sorted({regime for strategy in strategy_runs for regime in strategy.base.regime_accuracy})
        regime_accuracy = {
            regime_name: _aggregate(
                [float(strategy.base.regime_accuracy.get(regime_name, 0.0)) for strategy in strategy_runs]
            )
            for regime_name in regime_names
        }
        diagnostic_names = sorted({name for strategy in strategy_runs for name in strategy.base.diagnostics})
        diagnostics = {
            diagnostic_name: _aggregate(
                [float(strategy.base.diagnostics.get(diagnostic_name, 0.0)) for strategy in strategy_runs]
            )
            for diagnostic_name in diagnostic_names
        }
        temporal_metrics = {
            metric_name: _aggregate([float(getattr(strategy, metric_name)) for strategy in strategy_runs])
            for metric_name in _TEMPORAL_METRICS
        }
        strategy_aggregates.append(
            TemporalStrategyAggregate(
                name=strategy_name,
                metrics=metrics,
                regime_accuracy=regime_accuracy,
                diagnostics=diagnostics,
                temporal_metrics=temporal_metrics,
            )
        )

    return TemporalBenchmarkAggregate(
        name=name,
        severity=severity,
        reveal_delay_steps=reveal_delay_steps,
        steps=steps,
        seeds=seeds,
        strategies=tuple(strategy_aggregates),
    )


def _steps_for_severity(
    severity: str,
    *,
    standard_steps: int,
    harsh_steps: int,
    extreme_steps: int,
) -> int:
    if severity == "standard":
        return standard_steps
    if severity == "harsh":
        return harsh_steps
    if severity == "extreme":
        return extreme_steps
    raise ValueError(f"unsupported temporal suite severity: {severity}")


def _summarize_suite(benchmarks: list[TemporalBenchmarkAggregate]) -> tuple[TemporalStrategySummary, ...]:
    by_strategy: dict[str, dict[str, float | list[float]]] = {}
    for benchmark in benchmarks:
        utility_scores = {
            strategy.name: strategy.metrics["mean_utility"].mean
            for strategy in benchmark.strategies
        }
        accuracy_scores = {
            strategy.name: strategy.metrics["overall_accuracy"].mean
            for strategy in benchmark.strategies
        }
        best_utility = max(utility_scores.values())
        best_accuracy = max(accuracy_scores.values())
        frozen_utility = utility_scores.get("frozen", 0.0)
        frozen_accuracy = accuracy_scores.get("frozen", 0.0)
        for strategy in benchmark.strategies:
            slot = by_strategy.setdefault(
                strategy.name,
                {
                    "utility_wins": 0.0,
                    "accuracy_wins": 0.0,
                    "utility_margins": [],
                    "accuracy_margins": [],
                },
            )
            if utility_scores[strategy.name] >= best_utility - 1e-6:
                slot["utility_wins"] = float(slot["utility_wins"]) + 1.0
            if accuracy_scores[strategy.name] >= best_accuracy - 1e-6:
                slot["accuracy_wins"] = float(slot["accuracy_wins"]) + 1.0
            cast_utility = slot["utility_margins"]
            cast_accuracy = slot["accuracy_margins"]
            assert isinstance(cast_utility, list)
            assert isinstance(cast_accuracy, list)
            cast_utility.append(utility_scores[strategy.name] - frozen_utility)
            cast_accuracy.append(accuracy_scores[strategy.name] - frozen_accuracy)

    summaries: list[TemporalStrategySummary] = []
    for strategy_name in sorted(by_strategy):
        slot = by_strategy[strategy_name]
        utility_margins = slot["utility_margins"]
        accuracy_margins = slot["accuracy_margins"]
        assert isinstance(utility_margins, list)
        assert isinstance(accuracy_margins, list)
        summaries.append(
            TemporalStrategySummary(
                name=strategy_name,
                utility_wins=int(slot["utility_wins"]),
                accuracy_wins=int(slot["accuracy_wins"]),
                mean_utility_margin_vs_frozen=float(np.mean(utility_margins)) if utility_margins else 0.0,
                mean_accuracy_margin_vs_frozen=float(np.mean(accuracy_margins)) if accuracy_margins else 0.0,
            )
        )
    return tuple(summaries)


def run_temporal_benchmark_suite(
    *,
    seeds: tuple[int, ...] = (7, 11),
    severities: tuple[str, ...] = ("standard", "extreme"),
    reveal_delays: tuple[int, ...] = (2, 6, 12),
    batch_size: int = 64,
    backbone: str = "convnet",
    standard_steps: int = 36,
    harsh_steps: int = 48,
    extreme_steps: int = 48,
    source_train_size: int = 1200,
    source_epochs: int = 1,
) -> TemporalBenchmarkSuiteResult:
    benchmarks: list[TemporalBenchmarkAggregate] = []
    policy_factories = _suite_policy_factories()

    for severity in severities:
        steps = _steps_for_severity(
            severity,
            standard_steps=standard_steps,
            harsh_steps=harsh_steps,
            extreme_steps=extreme_steps,
        )
        for reveal_delay in reveal_delays:
            results = [
                run_temporal_fashion_mnist_benchmark(
                    steps=steps,
                    batch_size=batch_size,
                    seed=seed,
                    backbone=backbone,
                    severity=severity,
                    reveal_delay_steps=reveal_delay,
                    source_train_size=source_train_size,
                    source_epochs=source_epochs,
                    policy_factories=policy_factories,
                )
                for seed in seeds
            ]
            benchmarks.append(
                _aggregate_temporal_results(
                    name=f"temporal_{severity}_delay{reveal_delay}",
                    severity=severity,
                    reveal_delay_steps=reveal_delay,
                    steps=steps,
                    seeds=seeds,
                    results=results,
                )
            )

    return TemporalBenchmarkSuiteResult(
        seeds=seeds,
        severities=severities,
        reveal_delays=reveal_delays,
        summary=_summarize_suite(benchmarks),
        benchmarks=tuple(benchmarks),
    )


def run_temporal_paper_suite() -> TemporalBenchmarkSuiteResult:
    return run_temporal_benchmark_suite(
        seeds=(7, 11, 19),
        severities=("standard", "harsh", "extreme"),
        reveal_delays=(2, 4, 8, 12),
        batch_size=64,
        backbone="convnet",
        standard_steps=48,
        harsh_steps=60,
        extreme_steps=60,
        source_train_size=2000,
        source_epochs=2,
    )


def temporal_benchmark_suite_to_dict(result: TemporalBenchmarkSuiteResult) -> dict:
    return asdict(result)


def render_temporal_benchmark_suite_report(result: TemporalBenchmarkSuiteResult) -> str:
    lines = [
        "Adaptive Reliability Layer Temporal Benchmark Suite",
        f"seeds={','.join(str(seed) for seed in result.seeds)}",
        f"severities={','.join(result.severities)}",
        f"reveal_delays={','.join(str(delay) for delay in result.reveal_delays)}",
        "",
    ]

    if result.summary:
        lines.append("suite_summary")
        lines.append(
            "strategy                      utility_wins   accuracy_wins   mean_utility_delta_vs_frozen   mean_accuracy_delta_vs_frozen"
        )
        for strategy in result.summary:
            lines.append(
                f"{strategy.name:<28}"
                f"{strategy.utility_wins:>6}"
                f"{strategy.accuracy_wins:>16}"
                f"{strategy.mean_utility_margin_vs_frozen:>30.3f}"
                f"{strategy.mean_accuracy_margin_vs_frozen:>31.3f}"
            )
        lines.append("")

    for benchmark in result.benchmarks:
        lines.append(
            f"[severity={benchmark.severity} delay={benchmark.reveal_delay_steps} steps={benchmark.steps}]"
        )
        lines.append(
            "strategy                      acc(mean±std)   utility(mean±std)   "
            "risk_capital(mean±std)   revealed_acc(mean±std)   eventual_acc(mean±std)   retro_gap(mean±std)"
        )
        for strategy in benchmark.strategies:
            accuracy = strategy.metrics["overall_accuracy"]
            utility = strategy.metrics["mean_utility"]
            risk_capital = strategy.metrics["mean_risk_capital"]
            revealed_accuracy = strategy.temporal_metrics["revealed_accuracy"]
            eventual_accuracy = strategy.temporal_metrics["eventual_revealed_accuracy"]
            retro_gap = strategy.temporal_metrics["mean_retro_gap"]
            lines.append(
                f"{strategy.name:<28}"
                f"{accuracy.mean:>7.3f}±{accuracy.std:<6.3f}"
                f"{utility.mean:>18.3f}±{utility.std:<6.3f}"
                f"{risk_capital.mean:>23.3f}±{risk_capital.std:<6.3f}"
                f"{revealed_accuracy.mean:>24.3f}±{revealed_accuracy.std:<6.3f}"
                f"{eventual_accuracy.mean:>24.3f}±{eventual_accuracy.std:<6.3f}"
                f"{retro_gap.mean:>22.3f}±{retro_gap.std:<6.3f}"
            )
            diagnostic_summary = ", ".join(
                f"{name}={stat.mean:.3f}±{stat.std:.3f}"
                for name, stat in strategy.diagnostics.items()
                if name
                in {
                    "mean_retrospective_reward",
                    "mean_raw_retrospective_reward",
                    "mean_reward_trust",
                    "regime_reward_ema",
                    "regime_recurrence_similarity",
                    "revealed_accuracy",
                    "reveal_coverage",
                    "specialist_count",
                    "specialist_route_reuses",
                    "specialist_route_fallbacks",
                    "specialist_warm_starts_applied",
                    "specialist_warm_starts_skipped",
                    "specialist_last_reuse_confidence",
                }
            )
            if diagnostic_summary:
                lines.append(f"  diag:    {diagnostic_summary}")
        lines.append("")

    return "\n".join(lines).rstrip()
