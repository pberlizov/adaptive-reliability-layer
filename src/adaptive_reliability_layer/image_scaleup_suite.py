from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from .benchmark_suite import BenchmarkAggregate, _aggregate_benchmark_results
from .fashion_mnist_shift_benchmark import run_fashion_mnist_shift_benchmark_with_factories
from .tabular_benchmark import (
    BanditTabularPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    MultiActionTabularPolicy,
    NaiveTabularPolicy,
)


@dataclass(frozen=True)
class ImageScaleupSuiteResult:
    benchmarks: tuple[BenchmarkAggregate, ...]


def run_image_scaleup_suite(*, include_resnet_confirmations: bool = False) -> ImageScaleupSuiteResult:
    representative_policies = [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        ("hybrid", lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.15)),
    ]
    variant_specs = [
        ("fashion_convnet_standard", "convnet", "standard", (7, 11), 60),
        ("fashion_convnet_harsh", "convnet", "harsh", (7, 11), 60),
    ]
    if include_resnet_confirmations:
        variant_specs.extend(
            [
                ("fashion_resnet_small_standard", "resnet_small", "standard", (7,), 45),
                ("fashion_resnet_small_harsh", "resnet_small", "harsh", (7,), 45),
            ]
        )

    benchmarks: list[BenchmarkAggregate] = []
    for benchmark_name, backbone, severity, seeds, steps in variant_specs:
        results = [
            run_fashion_mnist_shift_benchmark_with_factories(
                policy_factories=representative_policies,
                steps=steps,
                batch_size=64,
                seed=seed,
                backbone=backbone,
                severity=severity,
            )
            for seed in seeds
        ]
        benchmarks.append(
            _aggregate_benchmark_results(
                name=benchmark_name,
                seeds=seeds,
                results=results,
            )
        )
    return ImageScaleupSuiteResult(benchmarks=tuple(benchmarks))


def image_scaleup_suite_to_dict(result: ImageScaleupSuiteResult) -> dict:
    return asdict(result)


def render_image_scaleup_suite_report(result: ImageScaleupSuiteResult) -> str:
    lines = ["Adaptive Reliability Layer Image Scale-Up Suite", ""]
    for benchmark in result.benchmarks:
        lines.append(f"[{benchmark.name}] seeds={','.join(str(seed) for seed in benchmark.seeds)}")
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
