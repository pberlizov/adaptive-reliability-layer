from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .fashion_mnist_shift_benchmark import (
    _build_fashion_mnist_source,
    _build_fashion_reference_profile,
    _build_reference_batches,
    _image_model_hparams,
    build_fashion_mnist_stream,
)
from .tabular_benchmark import (
    BanditTabularPolicy,
    ControllerTabularPolicy,
    DelayedBanditTabularPolicy,
    DelayedHybridBanditSpecialistPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    MultiActionTabularPolicy,
    NaiveTabularPolicy,
    PolicyFactory,
    RegimeAwareDelayedBanditTabularPolicy,
    RoutedDelayedBanditSpecialistPolicy,
    SpecialistMemoryTabularPolicy,
    RiskState,
    TabularBatch,
    TabularDecision,
    TabularReferenceProfile,
    TabularShiftSignal,
    TabularShiftMonitor,
    TabularStrategyResult,
    TabularTrace,
    _compute_batch_utility,
    _compute_reliability,
)
from .risk import MartingaleRiskMonitor
from .torch_image_model import TorchImageAdapterModel
from .torch_model import SourceFitSummary


@dataclass(frozen=True)
class TemporalFashionStrategyResult:
    base: TabularStrategyResult
    revealed_accuracy: float
    revealed_coverage: float
    eventual_revealed_accuracy: float
    reveal_delay_steps: int
    mean_retro_gap: float


@dataclass(frozen=True)
class TemporalFashionBenchmarkResult:
    steps: int
    batch_size: int
    reveal_delay_steps: int
    source_summary: SourceFitSummary
    reference: TabularReferenceProfile
    strategies: tuple[TemporalFashionStrategyResult, ...]


@dataclass(frozen=True)
class TemporalRewardConfig:
    smoothing_window: int = 5
    delay_penalty_scale: float = 0.22
    calibration_penalty_scale: float = 0.55
    coverage_penalty_scale: float = 0.15
    min_trust: float = 0.30


@dataclass(frozen=True)
class PendingReveal:
    decision_step: int
    reveal_step: int
    batch_correct: int
    batch_total: int
    batch: TabularBatch
    signal: TabularShiftSignal
    risk_state: RiskState
    decision: TabularDecision
    batch_accuracy: float
    reliability: float
    utility: float
    predicted_positive_rate: float
    feedback_state: object | None = None


def _compute_temporal_retrospective_reward(
    pending: PendingReveal,
    *,
    revealed_baseline_accuracy: float,
    revealed_accuracy: float,
    revealed_coverage: float,
) -> float:
    reward = pending.utility
    reward += 0.20 * (pending.batch_accuracy - revealed_baseline_accuracy)
    reward += 0.05 * (revealed_accuracy - revealed_baseline_accuracy)
    reward -= 0.18 * abs(pending.batch_accuracy - pending.reliability)
    reward -= 0.02 * max(0.0, 1.0 - revealed_coverage)
    return reward


def _compute_reward_trust(
    pending: PendingReveal,
    *,
    revealed_coverage: float,
    reward_config: TemporalRewardConfig,
) -> float:
    realized_delay = pending.reveal_step - pending.decision_step
    normalized_delay = min(1.0, realized_delay / 12.0)
    calibration_gap = abs(pending.batch_accuracy - pending.reliability)
    trust = 1.0
    trust -= reward_config.delay_penalty_scale * normalized_delay
    trust -= reward_config.calibration_penalty_scale * calibration_gap
    trust -= reward_config.coverage_penalty_scale * max(0.0, 1.0 - revealed_coverage)
    return max(reward_config.min_trust, min(1.0, trust))


def _default_temporal_fashion_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
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
        (
            "specialist_memory",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.15),
        ),
        (
            "hybrid",
            lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.15),
        ),
        (
            "delayed_hybrid",
            lambda reference: DelayedHybridBanditSpecialistPolicy(reference, distance_threshold=0.55),
        ),
    ]


def _evaluate_temporal_strategy(
    *,
    name: str,
    model: TorchImageAdapterModel,
    policy: object,
    batches: list[TabularBatch],
    reference: TabularReferenceProfile,
    reference_scores: list[float],
    reveal_delay_steps: int,
    reward_config: TemporalRewardConfig,
) -> TemporalFashionStrategyResult:
    monitor = TabularShiftMonitor(reference)
    risk_monitor = MartingaleRiskMonitor(reference_scores)

    total = 0
    correct = 0
    served_total = 0
    served_correct = 0
    alerts = 0
    risk_alerts = 0
    adaptations = 0
    resets = 0
    abstains = 0
    shift_sum = 0.0
    risk_capital_sum = 0.0
    reliability_sum = 0.0
    utility_sum = 0.0
    parameter_drift_sum = 0.0
    regime_correct: dict[str, int] = {}
    regime_total: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    traces: list[TabularTrace] = []

    pending_reveals: list[PendingReveal] = []
    revealed_total = 0
    revealed_correct = 0
    retro_gap_sum = 0.0
    retro_gap_count = 0
    retrospective_reward_sum = 0.0
    retrospective_reward_count = 0
    raw_retrospective_reward_sum = 0.0
    reward_trust_sum = 0.0
    reward_history: deque[float] = deque(maxlen=max(1, reward_config.smoothing_window))

    for step, batch in enumerate(batches):
        if hasattr(policy, "update_pending_feedback_summary"):
            pending_count = len(pending_reveals)
            if pending_count == 0:
                policy.update_pending_feedback_summary(
                    pending_count=0,
                    mean_age=0.0,
                    max_age=0.0,
                    stale_fraction=0.0,
                )
            else:
                ages = [max(0, step - item.decision_step) for item in pending_reveals]
                stale_fraction = sum(age >= max(4, reveal_delay_steps * 2) for age in ages) / max(1, pending_count)
                policy.update_pending_feedback_summary(
                    pending_count=pending_count,
                    mean_age=float(sum(ages) / max(1, pending_count)),
                    max_age=float(max(ages)),
                    stale_fraction=float(stale_fraction),
                )
        matured = [item for item in pending_reveals if item.reveal_step <= step]
        pending_reveals = [item for item in pending_reveals if item.reveal_step > step]
        for item in matured:
            revealed_baseline_accuracy = (
                revealed_correct / max(1, revealed_total)
                if revealed_total > 0
                else item.batch_accuracy
            )
            revealed_correct += item.batch_correct
            revealed_total += item.batch_total
            current_revealed_accuracy = revealed_correct / max(1, revealed_total)
            current_revealed_coverage = revealed_total / max(1, total)
            raw_retrospective_reward = _compute_temporal_retrospective_reward(
                item,
                revealed_baseline_accuracy=revealed_baseline_accuracy,
                revealed_accuracy=current_revealed_accuracy,
                revealed_coverage=current_revealed_coverage,
            )
            trust_weight = _compute_reward_trust(
                item,
                revealed_coverage=current_revealed_coverage,
                reward_config=reward_config,
            )
            smoothed_baseline = (
                sum(reward_history) / max(1, len(reward_history))
                if reward_history
                else raw_retrospective_reward
            )
            retrospective_reward = trust_weight * raw_retrospective_reward + (1.0 - trust_weight) * smoothed_baseline
            reward_history.append(retrospective_reward)
            retrospective_reward_sum += retrospective_reward
            retrospective_reward_count += 1
            raw_retrospective_reward_sum += raw_retrospective_reward
            reward_trust_sum += trust_weight
            if hasattr(policy, "observe_delayed_outcome"):
                policy.observe_delayed_outcome(
                    feedback_state=item.feedback_state,
                    model=model,
                    batch=item.batch,
                    signal=item.signal,
                    risk_state=item.risk_state,
                    decision=item.decision,
                    batch_accuracy=item.batch_accuracy,
                    reliability=item.reliability,
                    utility=item.utility,
                    retrospective_reward=retrospective_reward,
                    revealed_accuracy=current_revealed_accuracy,
                    revealed_coverage=current_revealed_coverage,
                    revealed_baseline_accuracy=revealed_baseline_accuracy,
                    pending_delay_steps=max(0, step - item.decision_step),
                    pending_outstanding_count=len(pending_reveals),
                    revealed_mean_residual=current_revealed_accuracy - item.predicted_positive_rate,
                    predicted_positive_rate=item.predicted_positive_rate,
                    revealed_positive_rate=float(item.batch.labels.mean()),
                )

        if hasattr(policy, "prepare_model"):
            policy.prepare_model(model, batch)

        pre_probabilities = model.predict_proba(batch.features)
        signal = monitor.evaluate(batch.features, pre_probabilities)
        raw_risk_score = signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk
        risk_state = risk_monitor.update(raw_risk_score)
        decision = policy.apply(model, signal, risk_state, batch, pre_probabilities)
        if decision.action == "reset":
            risk_monitor.reset()
            risk_state = RiskState(
                raw_score=risk_state.raw_score,
                p_value=risk_state.p_value,
                e_value=risk_state.e_value,
                capital=1.0,
                alert=False,
            )

        probabilities = model.predict_proba(batch.features)
        if hasattr(policy, "correct_probabilities"):
            probabilities = policy.correct_probabilities(
                probabilities,
                signal=signal,
                risk_state=risk_state,
                batch=batch,
            )
        predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
        batch_correct = int(sum(int(prediction == label) for prediction, label in zip(predictions, batch.labels)))
        batch_total = len(batch.labels)
        batch_accuracy = batch_correct / max(1, batch_total)
        reliability = _compute_reliability(signal, risk_state, decision)
        utility = _compute_batch_utility(
            batch_accuracy=batch_accuracy,
            risk_state=risk_state,
            decision=decision,
            parameter_drift=model.parameter_drift(),
        )

        total += batch_total
        if decision.action != "abstain":
            correct += batch_correct
            served_correct += batch_correct
            served_total += batch_total
        alerts += int(signal.alert)
        risk_alerts += int(risk_state.alert)
        adaptations += int(decision.action == "adapt")
        resets += int(decision.action == "reset")
        abstains += int(decision.action == "abstain")
        shift_sum += signal.score
        risk_capital_sum += risk_state.capital
        reliability_sum += reliability
        utility_sum += utility
        parameter_drift_sum += model.parameter_drift()
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        regime_correct[batch.regime] = regime_correct.get(batch.regime, 0) + (
            batch_correct if decision.action != "abstain" else 0
        )
        regime_total[batch.regime] = regime_total.get(batch.regime, 0) + batch_total
        traces.append(
            TabularTrace(
                step=step,
                regime=batch.regime,
                batch_accuracy=batch_accuracy,
                shift_score=signal.score,
                martingale_capital=risk_state.capital,
                martingale_p_value=risk_state.p_value,
                action=decision.action,
                selected_fraction=decision.selected_fraction,
                reliability_score=reliability,
                parameter_drift=model.parameter_drift(),
            )
        )
        feedback_state = None
        if hasattr(policy, "capture_feedback_state"):
            feedback_state = policy.capture_feedback_state(
                model=model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
            )
        pending_reveals.append(
            PendingReveal(
                decision_step=step,
                reveal_step=step + reveal_delay_steps,
                batch_correct=batch_correct,
                batch_total=batch_total,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
                batch_accuracy=batch_accuracy,
                reliability=reliability,
                utility=utility,
                predicted_positive_rate=float(sum(probabilities) / max(1, len(probabilities))),
                feedback_state=feedback_state,
            )
        )
        retro_gap_sum += abs(batch_accuracy - reliability)
        retro_gap_count += 1

        if hasattr(policy, "observe_outcome") and not hasattr(policy, "observe_delayed_outcome"):
            policy.observe_outcome(
                model=model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
                batch_accuracy=batch_accuracy,
                reliability=reliability,
                utility=utility,
            )

    eventual_revealed_correct = revealed_correct + sum(item.batch_correct for item in pending_reveals)
    eventual_revealed_total = revealed_total + sum(item.batch_total for item in pending_reveals)

    regime_accuracy = {
        regime: regime_correct[regime] / max(1, regime_total[regime])
        for regime in sorted(regime_total.keys())
    }
    diagnostics = policy.get_diagnostics() if hasattr(policy, "get_diagnostics") else {}
    diagnostics = dict(diagnostics)
    diagnostics["revealed_accuracy"] = revealed_correct / max(1, revealed_total)
    diagnostics["reveal_coverage"] = revealed_total / max(1, total)
    diagnostics["mean_retrospective_reward"] = retrospective_reward_sum / max(1, retrospective_reward_count)
    diagnostics["mean_raw_retrospective_reward"] = raw_retrospective_reward_sum / max(1, retrospective_reward_count)
    diagnostics["mean_reward_trust"] = reward_trust_sum / max(1, retrospective_reward_count)

    base = TabularStrategyResult(
        name=name,
        overall_accuracy=correct / max(1, total),
        served_accuracy=served_correct / max(1, served_total),
        coverage=served_total / max(1, total),
        mean_utility=utility_sum / max(1, len(traces)),
        alerts=alerts,
        risk_alerts=risk_alerts,
        adaptations=adaptations,
        resets=resets,
        abstains=abstains,
        mean_shift_score=shift_sum / max(1, len(traces)),
        mean_risk_capital=risk_capital_sum / max(1, len(traces)),
        mean_reliability=reliability_sum / max(1, len(traces)),
        mean_parameter_drift=parameter_drift_sum / max(1, len(traces)),
        regime_accuracy=regime_accuracy,
        action_counts=action_counts,
        diagnostics=diagnostics,
        traces=tuple(traces),
    )
    return TemporalFashionStrategyResult(
        base=base,
        revealed_accuracy=revealed_correct / max(1, revealed_total),
        revealed_coverage=revealed_total / max(1, total),
        eventual_revealed_accuracy=eventual_revealed_correct / max(1, eventual_revealed_total),
        reveal_delay_steps=reveal_delay_steps,
        mean_retro_gap=retro_gap_sum / max(1, retro_gap_count),
    )


def run_temporal_fashion_mnist_benchmark(
    *,
    steps: int = 90,
    batch_size: int = 64,
    seed: int = 7,
    backbone: str = "convnet",
    severity: str = "standard",
    reveal_delay_steps: int = 6,
    source_train_size: int | None = None,
    source_epochs: int | None = None,
    reward_config: TemporalRewardConfig | None = None,
    stream_batches: list[TabularBatch] | None = None,
    policy_factories: list[tuple[str, PolicyFactory]] | None = None,
) -> TemporalFashionBenchmarkResult:
    effective_reward_config = TemporalRewardConfig() if reward_config is None else reward_config
    effective_source_train_size = 10000 if source_train_size is None else source_train_size
    source = _build_fashion_mnist_source(seed=seed, source_train_size=effective_source_train_size)
    hidden_dim, adapter_dim, epochs, train_batch_size = _image_model_hparams(backbone)
    if source_epochs is not None:
        epochs = source_epochs
    model = TorchImageAdapterModel(
        seed=seed,
        backbone=backbone,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
    )
    source_summary = model.fit_source(
        source.x_train,
        source.y_train,
        source.x_validation,
        source.y_validation,
        epochs=epochs,
        batch_size=train_batch_size,
        learning_rate=1e-3,
    )
    reference_batches = _build_reference_batches(
        source.x_validation,
        source.y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    reference, reference_scores = _build_fashion_reference_profile(model, reference_batches)
    stream = (
        build_fashion_mnist_stream(
            source,
            steps=steps,
            batch_size=batch_size,
            seed=seed + 31,
            severity=severity,
        )
        if stream_batches is None
        else stream_batches
    )
    factories = policy_factories if policy_factories is not None else _default_temporal_fashion_policy_factories()
    strategies = tuple(
        _evaluate_temporal_strategy(
            name=name,
            model=model.clone(),
            policy=factory(reference),
            batches=stream,
            reference=reference,
            reference_scores=reference_scores,
            reveal_delay_steps=reveal_delay_steps,
            reward_config=effective_reward_config,
        )
        for name, factory in factories
    )
    return TemporalFashionBenchmarkResult(
        steps=steps,
        batch_size=batch_size,
        reveal_delay_steps=reveal_delay_steps,
        source_summary=source_summary,
        reference=reference,
        strategies=strategies,
    )


def render_temporal_fashion_mnist_report(result: TemporalFashionBenchmarkResult) -> str:
    frozen_accuracy = next(strategy.base.overall_accuracy for strategy in result.strategies if strategy.base.name == "frozen")
    lines = [
        "Adaptive Reliability Layer Temporal Fashion-MNIST Benchmark",
        (
            f"steps={result.steps} batch_size={result.batch_size} reveal_delay={result.reveal_delay_steps} "
            f"source_val_acc={result.source_summary.best_validation_accuracy:.3f}"
        ),
        "",
        "strategy     accuracy   revealed_acc   revealed_cov   eventual_acc   utility   delta_vs_frozen   risk_alerts   mean_capital   retro_gap",
    ]
    for strategy in result.strategies:
        base = strategy.base
        lines.append(
            f"{base.name:<12}"
            f"{base.overall_accuracy:>8.3f}"
            f"{strategy.revealed_accuracy:>15.3f}"
            f"{strategy.revealed_coverage:>15.3f}"
            f"{strategy.eventual_revealed_accuracy:>15.3f}"
            f"{base.mean_utility:>10.3f}"
            f"{base.overall_accuracy - frozen_accuracy:>18.3f}"
            f"{base.risk_alerts:>14}"
            f"{base.mean_risk_capital:>15.3f}"
            f"{strategy.mean_retro_gap:>11.3f}"
        )
    return "\n".join(lines)
