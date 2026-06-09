from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np

from .digits_shift_benchmark import run_digits_shift_benchmark
from .tabular_benchmark import TabularBenchmarkResult, run_tabular_benchmark


@dataclass(frozen=True)
class AggregateStat:
    mean: float
    std: float


@dataclass(frozen=True)
class StrategyAggregate:
    name: str
    metrics: dict[str, AggregateStat]
    regime_accuracy: dict[str, AggregateStat]
    diagnostics: dict[str, AggregateStat]


@dataclass(frozen=True)
class BenchmarkAggregate:
    name: str
    seeds: tuple[int, ...]
    strategies: tuple[StrategyAggregate, ...]


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    seeds: tuple[int, ...]
    benchmarks: tuple[BenchmarkAggregate, ...]


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


def _aggregate(values: list[float]) -> AggregateStat:
    array = np.asarray(values, dtype=np.float64)
    return AggregateStat(mean=float(array.mean()), std=float(array.std(ddof=0)))


def _aggregate_benchmark_results(
    *,
    name: str,
    seeds: tuple[int, ...],
    results: list[TabularBenchmarkResult],
) -> BenchmarkAggregate:
    by_strategy: dict[str, list] = {}
    for result in results:
        for strategy in result.strategies:
            by_strategy.setdefault(strategy.name, []).append(strategy)

    strategy_aggregates: list[StrategyAggregate] = []
    for strategy_name, strategy_runs in sorted(by_strategy.items()):
        metrics = {
            metric_name: _aggregate([float(getattr(strategy, metric_name)) for strategy in strategy_runs])
            for metric_name in _SUMMARY_METRICS
        }

        regime_names = sorted({name for strategy in strategy_runs for name in strategy.regime_accuracy})
        regime_accuracy = {
            regime_name: _aggregate(
                [float(strategy.regime_accuracy.get(regime_name, 0.0)) for strategy in strategy_runs]
            )
            for regime_name in regime_names
        }

        diagnostic_names = sorted({name for strategy in strategy_runs for name in strategy.diagnostics})
        diagnostics = {
            diagnostic_name: _aggregate(
                [float(strategy.diagnostics.get(diagnostic_name, 0.0)) for strategy in strategy_runs]
            )
            for diagnostic_name in diagnostic_names
        }
        strategy_aggregates.append(
            StrategyAggregate(
                name=strategy_name,
                metrics=metrics,
                regime_accuracy=regime_accuracy,
                diagnostics=diagnostics,
            )
        )

    return BenchmarkAggregate(name=name, seeds=seeds, strategies=tuple(strategy_aggregates))


def run_benchmark_suite(
    *,
    seeds: tuple[int, ...] = (7, 11, 19, 23, 29),
    tabular_steps: int = 90,
    tabular_batch_size: int = 48,
    digits_steps: int = 90,
    digits_batch_size: int = 48,
) -> BenchmarkSuiteResult:
    tabular_results = [
        run_tabular_benchmark(steps=tabular_steps, batch_size=tabular_batch_size, seed=seed)
        for seed in seeds
    ]
    digits_results = [
        run_digits_shift_benchmark(steps=digits_steps, batch_size=digits_batch_size, seed=seed)
        for seed in seeds
    ]
    return BenchmarkSuiteResult(
        seeds=seeds,
        benchmarks=(
            _aggregate_benchmark_results(name="tabular", seeds=seeds, results=tabular_results),
            _aggregate_benchmark_results(name="digits_shift", seeds=seeds, results=digits_results),
        ),
    )


def benchmark_suite_to_dict(result: BenchmarkSuiteResult) -> dict:
    return asdict(result)


def render_benchmark_suite_report(result: BenchmarkSuiteResult) -> str:
    lines = [
        "Adaptive Reliability Layer Multi-Seed Benchmark Suite",
        f"seeds={','.join(str(seed) for seed in result.seeds)}",
        "",
    ]

    for benchmark in result.benchmarks:
        lines.append(f"[{benchmark.name}]")
        lines.append(
            "strategy         acc(mean±std)   utility(mean±std)   coverage(mean±std)   "
            "risk_capital(mean±std)   risk_alerts(mean±std)   drift(mean±std)"
        )
        for strategy in benchmark.strategies:
            accuracy = strategy.metrics["overall_accuracy"]
            utility = strategy.metrics["mean_utility"]
            coverage = strategy.metrics["coverage"]
            risk_capital = strategy.metrics["mean_risk_capital"]
            risk_alerts = strategy.metrics["risk_alerts"]
            parameter_drift = strategy.metrics["mean_parameter_drift"]
            lines.append(
                f"{strategy.name:<16}"
                f"{accuracy.mean:>7.3f}±{accuracy.std:<6.3f}"
                f"{utility.mean:>18.3f}±{utility.std:<6.3f}"
                f"{coverage.mean:>19.3f}±{coverage.std:<6.3f}"
                f"{risk_capital.mean:>23.3f}±{risk_capital.std:<6.3f}"
                f"{risk_alerts.mean:>22.3f}±{risk_alerts.std:<6.3f}"
                f"{parameter_drift.mean:>16.3f}±{parameter_drift.std:<6.3f}"
            )
            regime_summary = ", ".join(
                f"{name}={stat.mean:.3f}±{stat.std:.3f}"
                for name, stat in strategy.regime_accuracy.items()
            )
            if regime_summary:
                lines.append(f"  regimes: {regime_summary}")
            diagnostic_summary = ", ".join(
                f"{name}={stat.mean:.3f}±{stat.std:.3f}"
                for name, stat in strategy.diagnostics.items()
            )
            if diagnostic_summary:
                lines.append(f"  diag:    {diagnostic_summary}")
        lines.append("")

    return "\n".join(lines).rstrip()
