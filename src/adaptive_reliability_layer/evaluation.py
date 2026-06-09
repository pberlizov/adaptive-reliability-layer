from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .adaptation import AdaptationController, AdaptationPolicy, FrozenPolicy, NaiveAdaptationPolicy
from .environment import Batch, build_stream
from .model import OnlineLogisticModel
from .monitoring import ReferenceProfile, ShiftMonitor, build_reference_profile
from .uncertainty import UncertaintyWrapper


@dataclass(frozen=True)
class SimulationResult:
    steps: int
    accuracy: float
    alerts: int
    adaptations: int
    resets: int


@dataclass(frozen=True)
class StrategyResult:
    name: str
    steps: int
    overall_accuracy: float
    alerts: int
    adaptations: int
    resets: int
    mean_shift_score: float
    mean_reliability: float
    mean_parameter_drift: float
    regime_accuracy: dict[str, float]
    traces: tuple["StepTrace", ...]


@dataclass(frozen=True)
class BenchmarkResult:
    steps: int
    seed: int
    batch_size: int
    strategies: tuple[StrategyResult, ...]
    reference: ReferenceProfile


@dataclass(frozen=True)
class StepTrace:
    step: int
    regime: str
    batch_accuracy: float
    shift_score: float
    feature_score: float
    output_score: float
    collapse_risk: float
    mean_entropy: float
    positive_rate: float
    mean_confidence: float
    action: str
    selected_fraction: float
    reliability_score: float
    trust_state: str
    parameter_drift: float


def _build_reference(seed: int, batch_size: int, reference_steps: int = 10) -> ReferenceProfile:
    reference_batches = build_stream(steps=reference_steps, seed=seed + 101, batch_size=batch_size)
    model = OnlineLogisticModel()
    features_batches = [batch.features for batch in reference_batches]
    probabilities_batches = [model.predict_proba(batch.features) for batch in reference_batches]
    return build_reference_profile(features_batches, probabilities_batches)


def _evaluate_policy(
    name: str,
    policy: AdaptationPolicy,
    batches: Iterable[Batch],
    reference: ReferenceProfile,
) -> StrategyResult:
    model = OnlineLogisticModel()
    monitor = ShiftMonitor(reference=reference)
    uncertainty = UncertaintyWrapper()

    total = 0
    correct = 0
    alerts = 0
    adaptations = 0
    resets = 0
    shift_score_sum = 0.0
    reliability_sum = 0.0
    parameter_drift_sum = 0.0
    batch_count = 0
    regime_correct: dict[str, int] = {}
    regime_total: dict[str, int] = {}
    traces: list[StepTrace] = []

    for step, batch in enumerate(batches):
        pre_probabilities = model.predict_proba(batch.features)
        signal = monitor.evaluate(batch.features, pre_probabilities)
        decision = policy.apply(model, signal, batch.features, pre_probabilities)
        probabilities = model.predict_proba(batch.features)
        reliability = uncertainty.summarize(probabilities, signal, decision)

        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        batch_correct = sum(int(pred == label) for pred, label in zip(predictions, batch.labels))
        batch_accuracy = batch_correct / max(1, len(batch.labels))

        total += len(predictions)
        correct += batch_correct
        alerts += int(signal.alert)
        adaptations += int(decision.action == "adapt")
        resets += int(decision.action == "reset")
        shift_score_sum += signal.score
        reliability_sum += reliability.reliability_score
        parameter_drift_sum += model.parameter_drift()
        batch_count += 1
        regime_correct[batch.regime] = regime_correct.get(batch.regime, 0) + batch_correct
        regime_total[batch.regime] = regime_total.get(batch.regime, 0) + len(batch.labels)
        traces.append(
            StepTrace(
                step=step,
                regime=batch.regime,
                batch_accuracy=batch_accuracy,
                shift_score=signal.score,
                feature_score=signal.feature_score,
                output_score=signal.output_score,
                collapse_risk=signal.collapse_risk,
                mean_entropy=signal.stats.mean_entropy,
                positive_rate=signal.stats.positive_rate,
                mean_confidence=signal.stats.mean_confidence,
                action=decision.action,
                selected_fraction=decision.selected_fraction,
                reliability_score=reliability.reliability_score,
                trust_state=reliability.trust_state,
                parameter_drift=model.parameter_drift(),
            )
        )

    regime_accuracy = {
        regime: regime_correct[regime] / max(1, regime_total[regime])
        for regime in sorted(regime_total.keys())
    }
    return StrategyResult(
        name=name,
        steps=batch_count,
        overall_accuracy=correct / max(1, total),
        alerts=alerts,
        adaptations=adaptations,
        resets=resets,
        mean_shift_score=shift_score_sum / max(1, batch_count),
        mean_reliability=reliability_sum / max(1, batch_count),
        mean_parameter_drift=parameter_drift_sum / max(1, batch_count),
        regime_accuracy=regime_accuracy,
        traces=tuple(traces),
    )


def run_benchmark(steps: int = 70, seed: int = 7, batch_size: int = 64) -> BenchmarkResult:
    reference = _build_reference(seed=seed, batch_size=batch_size)
    batches = build_stream(steps=steps, seed=seed, batch_size=batch_size)
    strategies = (
        _evaluate_policy("frozen", FrozenPolicy(), batches, reference),
        _evaluate_policy("naive", NaiveAdaptationPolicy(), batches, reference),
        _evaluate_policy("controller", AdaptationController(), batches, reference),
    )
    return BenchmarkResult(
        steps=steps,
        seed=seed,
        batch_size=batch_size,
        strategies=strategies,
        reference=reference,
    )


def run_simulation(steps: int = 70, seed: int = 7, batch_size: int = 64) -> SimulationResult:
    benchmark = run_benchmark(steps=steps, seed=seed, batch_size=batch_size)
    controller_result = next(result for result in benchmark.strategies if result.name == "controller")
    return SimulationResult(
        steps=controller_result.steps,
        accuracy=controller_result.overall_accuracy,
        alerts=controller_result.alerts,
        adaptations=controller_result.adaptations,
        resets=controller_result.resets,
    )


def render_benchmark_report(result: BenchmarkResult) -> str:
    frozen_accuracy = next(
        strategy.overall_accuracy for strategy in result.strategies if strategy.name == "frozen"
    )
    lines = [
        "Adaptive Reliability Layer Benchmark",
        f"steps={result.steps} seed={result.seed} batch_size={result.batch_size}",
        (
            "reference "
            f"feature_mean={result.reference.feature_mean:.3f} "
            f"feature_var={result.reference.feature_variance:.3f} "
            f"entropy={result.reference.mean_entropy:.3f} "
            f"positive_rate={result.reference.positive_rate:.3f}"
        ),
        "",
        "strategy     accuracy   delta_vs_frozen   alerts   adapts   resets   mean_shift   reliability   param_drift",
    ]
    for strategy in result.strategies:
        lines.append(
            f"{strategy.name:<12}"
            f"{strategy.overall_accuracy:>8.3f}"
            f"{strategy.overall_accuracy - frozen_accuracy:>18.3f}"
            f"{strategy.alerts:>9}"
            f"{strategy.adaptations:>9}"
            f"{strategy.resets:>9}"
            f"{strategy.mean_shift_score:>13.3f}"
            f"{strategy.mean_reliability:>14.3f}"
            f"{strategy.mean_parameter_drift:>14.3f}"
        )
        regime_summary = ", ".join(
            f"{regime}={accuracy:.3f}" for regime, accuracy in strategy.regime_accuracy.items()
        )
        lines.append(f"  regimes: {regime_summary}")
        worst_traces = sorted(strategy.traces, key=lambda trace: trace.batch_accuracy)[:3]
        worst_summary = ", ".join(
            (
                f"step={trace.step}:{trace.regime}:acc={trace.batch_accuracy:.2f}:"
                f"action={trace.action}:shift={trace.shift_score:.2f}"
            )
            for trace in worst_traces
        )
        lines.append(f"  worst:   {worst_summary}")
    return "\n".join(lines)
