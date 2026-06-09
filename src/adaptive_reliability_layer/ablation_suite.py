from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from .benchmark_suite import BenchmarkAggregate, _aggregate_benchmark_results
from .digits_shift_benchmark import run_digits_shift_benchmark_with_factories
from .tabular_benchmark import (
    BanditTabularPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    MultiActionTabularPolicy,
    PolicyFactory,
    SpecialistMemoryTabularPolicy,
    run_tabular_benchmark_with_factories,
)


@dataclass(frozen=True)
class AblationSuiteResult:
    seeds: tuple[int, ...]
    benchmarks: tuple[BenchmarkAggregate, ...]


def _tabular_ablation_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        (
            "multi_action_no_label_shift",
            lambda reference: MultiActionTabularPolicy(reference, enable_label_shift=False),
        ),
        (
            "multi_action_no_bn_refresh",
            lambda reference: MultiActionTabularPolicy(reference, enable_bn_refresh=False),
        ),
        (
            "multi_action_no_reset",
            lambda reference: MultiActionTabularPolicy(reference, enable_reset=False),
        ),
        (
            "multi_action_no_adapt",
            lambda reference: MultiActionTabularPolicy(reference, enable_adapt=False),
        ),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        (
            "bandit_no_reset",
            lambda reference: BanditTabularPolicy(
                reference,
                allowed_actions=("none", "bn_refresh", "label_shift", "recalibrate", "adapt"),
            ),
        ),
        (
            "bandit_risk_light",
            lambda reference: BanditTabularPolicy(reference, capital_penalty_scale=0.002),
        ),
        (
            "bandit_risk_heavy",
            lambda reference: BanditTabularPolicy(reference, capital_penalty_scale=0.030),
        ),
        ("specialist_memory", lambda reference: SpecialistMemoryTabularPolicy(reference)),
        (
            "specialist_memory_small_reservoir",
            lambda reference: SpecialistMemoryTabularPolicy(reference, max_specialists=2),
        ),
        (
            "specialist_memory_tight_routing",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.20),
        ),
        (
            "specialist_memory_wide_routing",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=2.10),
        ),
        ("hybrid", lambda reference: HybridBanditSpecialistPolicy(reference)),
    ]


def _digits_ablation_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        (
            "multi_action_no_label_shift",
            lambda reference: MultiActionTabularPolicy(reference, enable_label_shift=False),
        ),
        (
            "multi_action_no_bn_refresh",
            lambda reference: MultiActionTabularPolicy(reference, enable_bn_refresh=False),
        ),
        (
            "multi_action_no_reset",
            lambda reference: MultiActionTabularPolicy(reference, enable_reset=False),
        ),
        (
            "multi_action_no_adapt",
            lambda reference: MultiActionTabularPolicy(reference, enable_adapt=False),
        ),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        (
            "bandit_no_reset",
            lambda reference: BanditTabularPolicy(
                reference,
                allowed_actions=("none", "bn_refresh", "label_shift", "recalibrate", "adapt"),
            ),
        ),
        (
            "bandit_risk_light",
            lambda reference: BanditTabularPolicy(reference, capital_penalty_scale=0.002),
        ),
        (
            "bandit_risk_heavy",
            lambda reference: BanditTabularPolicy(reference, capital_penalty_scale=0.030),
        ),
        (
            "specialist_memory",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.10),
        ),
        (
            "specialist_memory_small_reservoir",
            lambda reference: SpecialistMemoryTabularPolicy(reference, max_specialists=2, distance_threshold=1.10),
        ),
        (
            "specialist_memory_tight_routing",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=0.85),
        ),
        (
            "specialist_memory_wide_routing",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.40),
        ),
        (
            "hybrid",
            lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.10),
        ),
    ]


def run_ablation_suite(
    *,
    seeds: tuple[int, ...] = (7, 11, 19),
    tabular_steps: int = 90,
    tabular_batch_size: int = 48,
    digits_steps: int = 90,
    digits_batch_size: int = 48,
) -> AblationSuiteResult:
    tabular_results = [
        run_tabular_benchmark_with_factories(
            policy_factories=_tabular_ablation_factories(),
            steps=tabular_steps,
            batch_size=tabular_batch_size,
            seed=seed,
        )
        for seed in seeds
    ]
    digits_results = [
        run_digits_shift_benchmark_with_factories(
            policy_factories=_digits_ablation_factories(),
            steps=digits_steps,
            batch_size=digits_batch_size,
            seed=seed,
        )
        for seed in seeds
    ]
    return AblationSuiteResult(
        seeds=seeds,
        benchmarks=(
            _aggregate_benchmark_results(name="tabular_ablation", seeds=seeds, results=tabular_results),
            _aggregate_benchmark_results(name="digits_ablation", seeds=seeds, results=digits_results),
        ),
    )


def ablation_suite_to_dict(result: AblationSuiteResult) -> dict:
    return asdict(result)


def render_ablation_suite_report(result: AblationSuiteResult) -> str:
    lines = [
        "Adaptive Reliability Layer Ablation Suite",
        f"seeds={','.join(str(seed) for seed in result.seeds)}",
        "",
    ]
    for benchmark in result.benchmarks:
        lines.append(f"[{benchmark.name}]")
        lines.append(
            "strategy                           acc(mean±std)   utility(mean±std)   "
            "risk_capital(mean±std)   drift(mean±std)"
        )
        for strategy in benchmark.strategies:
            accuracy = strategy.metrics["overall_accuracy"]
            utility = strategy.metrics["mean_utility"]
            risk_capital = strategy.metrics["mean_risk_capital"]
            parameter_drift = strategy.metrics["mean_parameter_drift"]
            lines.append(
                f"{strategy.name:<34}"
                f"{accuracy.mean:>7.3f}±{accuracy.std:<6.3f}"
                f"{utility.mean:>18.3f}±{utility.std:<6.3f}"
                f"{risk_capital.mean:>23.3f}±{risk_capital.std:<6.3f}"
                f"{parameter_drift.mean:>16.3f}±{parameter_drift.std:<6.3f}"
            )
            diagnostic_summary = ", ".join(
                f"{name}={stat.mean:.3f}±{stat.std:.3f}"
                for name, stat in strategy.diagnostics.items()
            )
            if diagnostic_summary:
                lines.append(f"  diag: {diagnostic_summary}")
        lines.append("")
    return "\n".join(lines).rstrip()
