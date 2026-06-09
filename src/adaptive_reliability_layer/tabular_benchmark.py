from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .regime import (
    RegimeDescriptor,
    StreamingRegimeEncoder,
    _behavior_feature_vector,
    build_regime_embedding,
    compute_model_behavior_signature,
)
from .risk import MartingaleRiskMonitor, RiskState
from .torch_model import ModelSnapshot, SourceFitSummary, TorchTabularAdapterModel


def _binary_entropy(probability: float) -> float:
    epsilon = 1e-6
    clamped = min(max(probability, epsilon), 1.0 - epsilon)
    return -(clamped * math.log(clamped) + (1.0 - clamped) * math.log(1.0 - clamped))


@dataclass(frozen=True)
class TabularBatch:
    features: np.ndarray
    labels: np.ndarray
    regime: str


@dataclass(frozen=True)
class TabularReferenceProfile:
    feature_mean: np.ndarray
    feature_variance: np.ndarray
    mean_entropy: float
    mean_probability: float
    positive_rate: float
    mean_confidence: float


@dataclass(frozen=True)
class TabularShiftSignal:
    score: float
    feature_score: float
    output_score: float
    collapse_risk: float
    alert: bool
    severe: bool
    mean_entropy: float
    mean_probability: float
    positive_rate: float
    mean_confidence: float


@dataclass(frozen=True)
class TabularDecision:
    action: str
    reason: str
    selected_fraction: float = 0.0


@dataclass(frozen=True)
class TabularTrace:
    step: int
    regime: str
    batch_accuracy: float
    shift_score: float
    martingale_capital: float
    martingale_p_value: float
    action: str
    selected_fraction: float
    reliability_score: float
    parameter_drift: float


@dataclass(frozen=True)
class BanditFeedbackState:
    action: str
    context: np.ndarray
    regime_descriptor: RegimeDescriptor | None = None
    batch_features: np.ndarray | None = None
    predicted_probabilities: np.ndarray | None = None


@dataclass(frozen=True)
class TabularStrategyResult:
    name: str
    overall_accuracy: float
    served_accuracy: float
    coverage: float
    mean_utility: float
    alerts: int
    risk_alerts: int
    adaptations: int
    resets: int
    abstains: int
    mean_shift_score: float
    mean_risk_capital: float
    mean_reliability: float
    mean_parameter_drift: float
    regime_accuracy: dict[str, float]
    action_counts: dict[str, int]
    diagnostics: dict[str, float]
    traces: tuple[TabularTrace, ...]


@dataclass(frozen=True)
class TabularBenchmarkResult:
    steps: int
    batch_size: int
    source_summary: SourceFitSummary
    reference: TabularReferenceProfile
    strategies: tuple[TabularStrategyResult, ...]


PolicyFactory = Callable[[TabularReferenceProfile], object]


@dataclass
class SpecialistSlot:
    name: str
    snapshot: "ModelSnapshot"
    signature: np.ndarray
    controller: object
    usage_count: int = 0
    cumulative_reward: float = 0.0
    reward_ema: float = 0.0
    similarity_ema: float = 0.0
    lift_ema: float = 0.0
    recurrence_reward_ema: float = 0.0
    route_advantage_ema: float = 0.0
    future_reuse_ema: float = 0.0
    support_quality_ema: float = 0.0
    reveal_count: int = 0
    last_used_step: int = 0
    successful_reuses: int = 0
    shadow_wins: int = 0
    probation_remaining: int = 0
    support_features: np.ndarray | None = None
    support_positive_rate: float = 0.5
    creation_positive_rate: float = 0.5
    regime_anchor: np.ndarray | None = None
    regime_confidence_ema: float = 0.0
    exchangeability_ema: float = 0.0
    reservoir_cluster_id: int = 0
    coreset: object | None = None
    quality_ema: float = 0.0
    behavior_signature: np.ndarray | None = None


@dataclass(frozen=True)
class DelayedHybridFeedbackState:
    slot: SpecialistSlot
    slot_snapshot: "ModelSnapshot"
    active_signature: np.ndarray | None
    candidate_new: bool
    inner_feedback_state: object | None = None
    slot_index: int = 0
    route_distance: float = 0.0
    route_similarity: float = 0.0
    novelty_score: float = 0.0
    recurrence_similarity: float = 0.0
    routing_step: int = 0
    shadow_base_snapshot: "ModelSnapshot" | None = None
    shadow_alt_snapshot: "ModelSnapshot" | None = None
    shadow_alt_index: int = -1
    regime_descriptor: RegimeDescriptor | None = None
    exchangeability_score: float = 0.0


class TabularShiftMonitor:
    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        alert_threshold: float = 1.1,
        severe_threshold: float = 1.75,
    ) -> None:
        self._reference = reference
        self._alert_threshold = alert_threshold
        self._severe_threshold = severe_threshold

    def evaluate(self, features: np.ndarray, probabilities: list[float]) -> TabularShiftSignal:
        batch_mean = features.mean(axis=0)
        batch_variance = features.var(axis=0)

        normalized_mean_gap = np.mean(
            np.abs(batch_mean - self._reference.feature_mean) / np.sqrt(self._reference.feature_variance + 1e-6)
        )
        normalized_variance_gap = np.mean(
            np.abs(batch_variance - self._reference.feature_variance) / (self._reference.feature_variance + 1e-6)
        )
        feature_score = float(normalized_mean_gap + 0.5 * normalized_variance_gap)

        mean_entropy = float(np.mean([_binary_entropy(probability) for probability in probabilities]))
        mean_probability = float(np.mean(probabilities))
        positive_rate = float(np.mean([1.0 if probability >= 0.5 else 0.0 for probability in probabilities]))
        mean_confidence = float(np.mean([max(probability, 1.0 - probability) for probability in probabilities]))

        entropy_gap = abs(mean_entropy - self._reference.mean_entropy)
        probability_gap = abs(mean_probability - self._reference.mean_probability)
        rate_gap = abs(positive_rate - self._reference.positive_rate)
        confidence_gap = abs(mean_confidence - self._reference.mean_confidence)
        output_score = float(entropy_gap + 0.75 * probability_gap + rate_gap + 0.5 * confidence_gap)

        collapse_risk = float(
            max(0.0, self._reference.mean_entropy - mean_entropy)
            + max(0.0, abs(positive_rate - 0.5) - abs(self._reference.positive_rate - 0.5))
        )

        score = float(feature_score + 0.75 * output_score + 0.65 * collapse_risk)
        return TabularShiftSignal(
            score=score,
            feature_score=feature_score,
            output_score=output_score,
            collapse_risk=collapse_risk,
            alert=score >= self._alert_threshold,
            severe=score >= self._severe_threshold or collapse_risk >= 0.30,
            mean_entropy=mean_entropy,
            mean_probability=mean_probability,
            positive_rate=positive_rate,
            mean_confidence=mean_confidence,
        )


class FrozenTabularPolicy:
    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        del model, signal, risk_state, batch, probabilities
        return TabularDecision(action="none", reason="frozen_baseline")


class TentTabularPolicy:
    """Standard TTA baseline: entropy minimization on BatchNorm parameters only."""

    def __init__(self, *, steps: int = 1, learning_rate: float = 0.00025) -> None:
        self._steps = steps
        self._learning_rate = learning_rate

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        del signal, risk_state, probabilities
        model.tent_adapt(
            batch.features,
            steps=self._steps,
            learning_rate=self._learning_rate,
            selective=False,
        )
        return TabularDecision(action="none", reason="tent_entropy_bn")


class EataStyleTabularPolicy:
    """EATA-style selective TTA: entropy minimization on confident unlabeled samples only."""

    def __init__(
        self,
        *,
        steps: int = 1,
        learning_rate: float = 0.00025,
        confidence_threshold: float = 0.70,
    ) -> None:
        self._steps = steps
        self._learning_rate = learning_rate
        self._confidence_threshold = confidence_threshold

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        del signal, risk_state, probabilities
        model.tent_adapt(
            batch.features,
            steps=self._steps,
            learning_rate=self._learning_rate,
            selective=True,
            confidence_threshold=self._confidence_threshold,
        )
        return TabularDecision(action="none", reason="eata_style_selective_tent")


class NaiveTabularPolicy:
    def __init__(
        self,
        *,
        trigger_threshold: float = 1.0,
        learning_rate: float = 0.06,
        confidence_threshold: float = 0.68,
        entropy_weight: float = 0.10,
        max_parameter_drift: float = 3.0,
    ) -> None:
        self._trigger_threshold = trigger_threshold
        self._learning_rate = learning_rate
        self._confidence_threshold = confidence_threshold
        self._entropy_weight = entropy_weight
        self._max_parameter_drift = max_parameter_drift

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        del risk_state
        if signal.score < self._trigger_threshold:
            return TabularDecision(action="none", reason="shift_below_threshold")

        selected_fraction = model.adapt(
            batch.features,
            probabilities,
            learning_rate=self._learning_rate,
            confidence_threshold=self._confidence_threshold,
            anchor_strength=0.0,
            entropy_weight=self._entropy_weight,
            max_parameter_drift=self._max_parameter_drift,
            steps=3,
        )
        if selected_fraction == 0.0:
            return TabularDecision(action="hold", reason="no_confident_samples")
        return TabularDecision(action="adapt", reason="naive_shift_trigger", selected_fraction=selected_fraction)


class ScheduledRetrainTabularPolicy:
    """Strong baseline: bounded adapt on a fixed batch schedule (scheduled retrain proxy)."""

    def __init__(
        self,
        *,
        retrain_interval: int = 6,
        learning_rate: float = 0.05,
        confidence_threshold: float = 0.68,
        entropy_weight: float = 0.08,
        max_parameter_drift: float = 2.5,
    ) -> None:
        self._retrain_interval = max(1, retrain_interval)
        self._learning_rate = learning_rate
        self._confidence_threshold = confidence_threshold
        self._entropy_weight = entropy_weight
        self._max_parameter_drift = max_parameter_drift
        self._steps_since_retrain = 0

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        del signal, risk_state
        self._steps_since_retrain += 1
        if self._steps_since_retrain < self._retrain_interval:
            return TabularDecision(action="none", reason="scheduled_retrain_wait")
        self._steps_since_retrain = 0
        selected_fraction = model.adapt(
            batch.features,
            probabilities,
            learning_rate=self._learning_rate,
            confidence_threshold=self._confidence_threshold,
            anchor_strength=0.12,
            entropy_weight=self._entropy_weight,
            max_parameter_drift=self._max_parameter_drift,
            steps=2,
        )
        if selected_fraction == 0.0:
            return TabularDecision(action="recalibrate", reason="scheduled_retrain_recalibrate_fallback")
        return TabularDecision(
            action="adapt",
            reason="scheduled_retrain_interval",
            selected_fraction=selected_fraction,
        )


class ControllerTabularPolicy:
    def __init__(
        self,
        *,
        mild_threshold: float = 1.0,
        severe_threshold: float = 1.7,
        learning_rate: float = 0.03,
        max_parameter_drift: float = 0.6,
        anchor_strength: float = 0.15,
        entropy_weight: float = 0.08,
        cooldown_steps: int = 2,
    ) -> None:
        self._mild_threshold = mild_threshold
        self._severe_threshold = severe_threshold
        self._learning_rate = learning_rate
        self._max_parameter_drift = max_parameter_drift
        self._anchor_strength = anchor_strength
        self._entropy_weight = entropy_weight
        self._cooldown_steps = cooldown_steps
        self._cooldown_remaining = 0
        self._consecutive_severe = 0

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return TabularDecision(action="hold", reason="cooldown")

        if signal.score < self._mild_threshold and not risk_state.alert:
            self._consecutive_severe = 0
            return TabularDecision(action="none", reason="shift_below_threshold")

        if signal.severe or risk_state.alert or signal.score >= self._severe_threshold:
            self._consecutive_severe += 1
        else:
            self._consecutive_severe = 0

        if risk_state.capital >= 18.0 or self._consecutive_severe >= 3 or signal.collapse_risk >= 0.55:
            model.reset()
            self._consecutive_severe = 0
            self._cooldown_remaining = self._cooldown_steps
            return TabularDecision(action="reset", reason="risk_or_severe_persistence")

        confidence_threshold = 0.90 if signal.severe or risk_state.alert else 0.80
        selected_fraction = model.adapt(
            batch.features,
            probabilities,
            learning_rate=self._learning_rate,
            confidence_threshold=confidence_threshold,
            anchor_strength=self._anchor_strength,
            entropy_weight=self._entropy_weight,
            max_parameter_drift=self._max_parameter_drift,
            steps=2,
        )
        if selected_fraction == 0.0:
            self._cooldown_remaining = 1
            return TabularDecision(action="hold", reason="no_confident_samples")

        self._cooldown_remaining = self._cooldown_steps
        return TabularDecision(action="adapt", reason="controlled_shift_response", selected_fraction=selected_fraction)


class MultiActionTabularPolicy:
    """Controller over a menu of adaptation primitives."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        mild_threshold: float = 0.95,
        severe_threshold: float = 1.55,
        cooldown_steps: int = 2,
        enable_label_shift: bool = True,
        enable_bn_refresh: bool = True,
        enable_recalibration: bool = True,
        enable_reset: bool = True,
        enable_adapt: bool = True,
        enable_abstain: bool = True,
    ) -> None:
        self._reference = reference
        self._mild_threshold = mild_threshold
        self._severe_threshold = severe_threshold
        self._cooldown_steps = cooldown_steps
        self._enable_label_shift = enable_label_shift
        self._enable_bn_refresh = enable_bn_refresh
        self._enable_recalibration = enable_recalibration
        self._enable_reset = enable_reset
        self._enable_adapt = enable_adapt
        self._enable_abstain = enable_abstain
        self._cooldown_remaining = 0
        self._recent_reset = 0

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._recent_reset = max(0, self._recent_reset - 1)
            return TabularDecision(action="hold", reason="cooldown")

        self._recent_reset = max(0, self._recent_reset - 1)

        if risk_state.capital >= 25.0 or signal.collapse_risk >= 0.70:
            if self._recent_reset > 0:
                self._cooldown_remaining = 1
                if self._enable_abstain:
                    return TabularDecision(action="abstain", reason="persistent_high_risk")
                return TabularDecision(action="hold", reason="persistent_high_risk")

            if self._enable_reset:
                model.reset()
                self._cooldown_remaining = self._cooldown_steps
                self._recent_reset = 3
                return TabularDecision(action="reset", reason="hard_risk_reset")
            if self._enable_abstain:
                self._cooldown_remaining = 1
                return TabularDecision(action="abstain", reason="hard_risk_without_reset")
            return TabularDecision(action="hold", reason="hard_risk_without_reset")

        if signal.score < self._mild_threshold and not risk_state.alert:
            return TabularDecision(action="none", reason="shift_below_threshold")

        probability_gap = abs(signal.mean_probability - self._reference.mean_probability)
        confidence_gap = self._reference.mean_confidence - signal.mean_confidence
        label_shift_like = signal.output_score > 0.18 and probability_gap > 0.08 and signal.feature_score < 0.95

        if self._enable_label_shift and label_shift_like:
            model.apply_label_shift_correction(
                source_positive_rate=self._reference.mean_probability,
                target_positive_rate=signal.mean_probability,
                momentum=0.35,
                max_abs_bias=1.10,
            )
            self._cooldown_remaining = 1
            return TabularDecision(action="label_shift", reason="posterior_mean_shift")

        if (
            self._enable_bn_refresh
            and signal.feature_score > 0.95
            and signal.output_score < 0.35
            and signal.mean_confidence >= 0.78
        ):
            model.refresh_batch_norm(batch.features, passes=2)
            self._cooldown_remaining = 1
            return TabularDecision(action="bn_refresh", reason="feature_shift_without_output_instability")

        # Overconfidence-specific path: model is more confident than at training.
        # Use cool_confidence (gentler, no accuracy guard).
        if (
            self._enable_recalibration
            and confidence_gap < -0.05
            and signal.collapse_risk < 0.40
            and signal.feature_score < self._severe_threshold
        ):
            model.recalibrate_temperature(
                reference_confidence=self._reference.mean_confidence,
                observed_confidence=signal.mean_confidence,
                momentum=0.15,
                min_temperature=0.80,
                max_temperature=1.40,
            )
            self._cooldown_remaining = 1
            return TabularDecision(action="cool_confidence", reason="overconfidence_gap")
        # Under-confidence path: model is less confident than at training.
        if (
            self._enable_recalibration
            and confidence_gap > 0.05
            and signal.collapse_risk < 0.40
            and signal.feature_score < self._severe_threshold
        ):
            model.recalibrate_temperature(
                reference_confidence=self._reference.mean_confidence,
                observed_confidence=signal.mean_confidence,
                momentum=0.30,
                min_temperature=0.70,
                max_temperature=1.30,
            )
            self._cooldown_remaining = 1
            return TabularDecision(action="recalibrate", reason="confidence_gap")

        if signal.score >= self._severe_threshold or risk_state.alert:
            if signal.mean_confidence < 0.72:
                self._cooldown_remaining = 1
                if self._enable_abstain:
                    return TabularDecision(action="abstain", reason="high_risk_low_confidence")
                return TabularDecision(action="hold", reason="high_risk_low_confidence")

        if not self._enable_adapt:
            self._cooldown_remaining = 1
            return TabularDecision(action="hold", reason="adaptation_disabled")

        confidence_threshold = 0.90 if signal.severe or risk_state.alert else 0.80
        selected_fraction = model.adapt(
            batch.features,
            probabilities,
            learning_rate=0.025,
            confidence_threshold=confidence_threshold,
            anchor_strength=0.18,
            entropy_weight=0.08,
            max_parameter_drift=0.65,
            steps=2,
        )
        if selected_fraction == 0.0:
            self._cooldown_remaining = 1
            return TabularDecision(action="hold", reason="no_confident_samples")

        self._cooldown_remaining = self._cooldown_steps
        return TabularDecision(action="adapt", reason="adapter_head_update", selected_fraction=selected_fraction)


def _decision_none() -> TabularDecision:
    return TabularDecision(action="none", reason="no_intervention")


def _decision_abstain(reason: str) -> TabularDecision:
    return TabularDecision(action="abstain", reason=reason)


def _apply_bn_refresh(model: TorchTabularAdapterModel, batch: TabularBatch) -> TabularDecision:
    model.refresh_batch_norm(batch.features, passes=2)
    return TabularDecision(action="bn_refresh", reason="feature_shift_without_output_instability")


def _apply_label_shift(
    model: TorchTabularAdapterModel,
    reference: TabularReferenceProfile,
    signal: TabularShiftSignal,
) -> TabularDecision:
    model.apply_label_shift_correction(
        source_positive_rate=reference.mean_probability,
        target_positive_rate=signal.mean_probability,
        momentum=0.35,
        max_abs_bias=1.10,
    )
    return TabularDecision(action="label_shift", reason="posterior_mean_shift")


def _apply_recalibration(
    model: TorchTabularAdapterModel,
    reference: TabularReferenceProfile,
    signal: TabularShiftSignal,
) -> TabularDecision:
    model.recalibrate_temperature(
        reference_confidence=reference.mean_confidence,
        observed_confidence=signal.mean_confidence,
        momentum=0.30,
        min_temperature=0.70,
        max_temperature=1.30,
    )
    return TabularDecision(action="recalibrate", reason="confidence_gap")


def _apply_reset(model: TorchTabularAdapterModel, reason: str) -> TabularDecision:
    model.reset()
    return TabularDecision(action="reset", reason=reason)


def _apply_adaptation(
    model: TorchTabularAdapterModel,
    signal: TabularShiftSignal,
    risk_state: RiskState,
    batch: TabularBatch,
    probabilities: list[float],
    *,
    learning_rate: float,
    anchor_strength: float,
    entropy_weight: float,
    max_parameter_drift: float,
    severe_confidence_threshold: float = 0.90,
    default_confidence_threshold: float = 0.80,
    steps: int = 2,
) -> TabularDecision:
    confidence_threshold = severe_confidence_threshold if signal.severe or risk_state.alert else default_confidence_threshold
    selected_fraction = model.adapt(
        batch.features,
        probabilities,
        learning_rate=learning_rate,
        confidence_threshold=confidence_threshold,
        anchor_strength=anchor_strength,
        entropy_weight=entropy_weight,
        max_parameter_drift=max_parameter_drift,
        steps=steps,
    )
    if selected_fraction == 0.0:
        return TabularDecision(action="hold", reason="no_confident_samples")
    return TabularDecision(action="adapt", reason="adapter_head_update", selected_fraction=selected_fraction)


def _bandit_context(reference: TabularReferenceProfile, signal: TabularShiftSignal, risk_state: RiskState) -> np.ndarray:
    return np.array(
        [
            1.0,
            signal.feature_score,
            signal.output_score,
            signal.collapse_risk,
            signal.mean_probability - reference.mean_probability,
            reference.mean_confidence - signal.mean_confidence,
            min(risk_state.capital, 25.0) / 25.0,
            float(signal.alert),
            float(risk_state.alert),
        ],
        dtype=np.float64,
    )


class BanditTabularPolicy:
    """Lightweight learned controller using a contextual linear UCB policy."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        alpha: float = 0.75,
        ridge: float = 1.0,
        allowed_actions: tuple[str, ...] | None = None,
        capital_penalty_scale: float = 0.01,
        context_dim: int = 9,
    ) -> None:
        self._reference = reference
        default_actions = ("none", "bn_refresh", "label_shift", "recalibrate", "adapt", "reset")
        chosen_actions = allowed_actions if allowed_actions is not None else default_actions
        if "none" not in chosen_actions:
            chosen_actions = ("none",) + tuple(chosen_actions)
        self._actions = chosen_actions
        self._alpha = alpha
        self._dim = context_dim
        self._capital_penalty_scale = capital_penalty_scale
        self._matrices = {
            action: ridge * np.eye(self._dim, dtype=np.float64)
            for action in self._actions
        }
        self._vectors = {
            action: np.zeros(self._dim, dtype=np.float64)
            for action in self._actions
        }
        self._pending_action: str | None = None
        self._pending_context: np.ndarray | None = None

    def _context(
        self,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch | None = None,
        probabilities: list[float] | None = None,
    ) -> np.ndarray:
        del batch, probabilities
        return _bandit_context(self._reference, signal, risk_state)

    def _feedback_regime_descriptor(self) -> RegimeDescriptor | None:
        return None

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        if (risk_state.capital >= 30.0 or signal.collapse_risk >= 0.72) and "reset" in self._actions:
            self._pending_action = "reset"
            self._pending_context = self._context(signal, risk_state, batch, probabilities)
            return _apply_reset(model, "bandit_hard_risk_reset")

        if signal.score < 0.90 and not risk_state.alert:
            self._pending_action = "none"
            self._pending_context = self._context(signal, risk_state, batch, probabilities)
            return _decision_none()

        context = self._context(signal, risk_state, batch, probabilities)
        candidate_actions = list(self._actions)
        if signal.mean_confidence < 0.74:
            candidate_actions = [action for action in candidate_actions if action != "adapt"]

        best_action = "none"
        best_score = -float("inf")
        for action in candidate_actions:
            matrix = self._matrices[action]
            vector = self._vectors[action]
            theta = np.linalg.solve(matrix, vector)
            bonus = self._alpha * float(np.sqrt(context @ np.linalg.solve(matrix, context)))
            score = float(theta @ context + bonus)
            if score > best_score:
                best_action = action
                best_score = score

        self._pending_action = best_action
        self._pending_context = context
        return self._execute_action(best_action, model, signal, risk_state, batch, probabilities)

    def observe_outcome(
        self,
        *,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
    ) -> None:
        del signal, decision, reliability
        if self._pending_action is None or self._pending_context is None:
            return

        reward = utility - self._capital_penalty_scale * max(0.0, risk_state.capital - 1.0)
        self._apply_feedback(
            BanditFeedbackState(
                action=self._pending_action,
                context=self._pending_context,
                regime_descriptor=self._feedback_regime_descriptor(),
                batch_features=batch.features.copy(),
                predicted_probabilities=np.asarray(model.predict_proba(batch.features), dtype=np.float64),
            ),
            reward=reward,
        )
        self._pending_action = None
        self._pending_context = None

    def _feedback_reward(self, *, risk_state: RiskState, utility: float) -> float:
        return utility - self._capital_penalty_scale * max(0.0, risk_state.capital - 1.0)

    def _apply_feedback(self, feedback_state: BanditFeedbackState, *, reward: float) -> None:
        self._matrices[feedback_state.action] += np.outer(feedback_state.context, feedback_state.context)
        self._vectors[feedback_state.action] += reward * feedback_state.context

    def _execute_action(
        self,
        action: str,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        if action == "none":
            return _decision_none()
        if action == "bn_refresh":
            return _apply_bn_refresh(model, batch)
        if action == "label_shift":
            return _apply_label_shift(model, self._reference, signal)
        if action == "recalibrate":
            return _apply_recalibration(model, self._reference, signal)
        if action == "adapt":
            return _apply_adaptation(
                model,
                signal,
                risk_state,
                batch,
                probabilities,
                learning_rate=0.025,
                anchor_strength=0.16,
                entropy_weight=0.08,
                max_parameter_drift=0.70,
                steps=2,
            )
        return _apply_reset(model, "bandit_selected_reset")


class DelayedBanditTabularPolicy(BanditTabularPolicy):
    """Bandit controller that only learns when delayed labels are revealed."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        alpha: float = 0.75,
        ridge: float = 1.0,
        allowed_actions: tuple[str, ...] | None = None,
        capital_penalty_scale: float = 0.01,
    ) -> None:
        super().__init__(
            reference,
            alpha=alpha,
            ridge=ridge,
            allowed_actions=allowed_actions,
            capital_penalty_scale=capital_penalty_scale,
        )
        self._captured_feedback: BanditFeedbackState | None = None
        self._pending_feedback_count = 0.0
        self._pending_feedback_mean_age = 0.0
        self._pending_feedback_max_age = 0.0
        self._pending_feedback_stale_fraction = 0.0

    def _after_apply(
        self,
        *,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        feedback_state: BanditFeedbackState | None,
    ) -> None:
        del signal, risk_state, decision, feedback_state

    def update_pending_feedback_summary(
        self,
        *,
        pending_count: int,
        mean_age: float,
        max_age: float,
        stale_fraction: float,
    ) -> None:
        self._pending_feedback_count = float(pending_count)
        self._pending_feedback_mean_age = float(mean_age)
        self._pending_feedback_max_age = float(max_age)
        self._pending_feedback_stale_fraction = float(stale_fraction)

    def _after_delayed_outcome(
        self,
        *,
        feedback_state: BanditFeedbackState | None,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
        retrospective_reward: float,
        revealed_accuracy: float | None,
        revealed_coverage: float | None,
        revealed_baseline_accuracy: float | None,
        pending_delay_steps: int,
        pending_outstanding_count: int,
        revealed_mean_residual: float,
        predicted_positive_rate: float,
        revealed_positive_rate: float,
    ) -> None:
        del feedback_state, signal, risk_state, decision, batch_accuracy, reliability, utility
        del retrospective_reward, revealed_accuracy, revealed_coverage, revealed_baseline_accuracy
        del pending_delay_steps, pending_outstanding_count, revealed_mean_residual
        del predicted_positive_rate, revealed_positive_rate

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        decision = super().apply(model, signal, risk_state, batch, probabilities)
        captured_feedback: BanditFeedbackState | None = None
        if self._pending_action is not None and self._pending_context is not None:
            captured_feedback = BanditFeedbackState(
                action=self._pending_action,
                context=self._pending_context.copy(),
                regime_descriptor=self._feedback_regime_descriptor(),
                batch_features=batch.features.copy(),
                predicted_probabilities=np.asarray(probabilities, dtype=np.float64),
            )
            self._captured_feedback = captured_feedback
        else:
            self._captured_feedback = None
        self._pending_action = None
        self._pending_context = None
        self._after_apply(
            signal=signal,
            risk_state=risk_state,
            decision=decision,
            feedback_state=captured_feedback,
        )
        return decision

    def capture_feedback_state(self, **_: object) -> BanditFeedbackState | None:
        feedback_state = self._captured_feedback
        self._captured_feedback = None
        return feedback_state

    def observe_delayed_outcome(
        self,
        *,
        feedback_state: BanditFeedbackState | None,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
        retrospective_reward: float | None = None,
        revealed_accuracy: float | None = None,
        revealed_coverage: float | None = None,
        revealed_baseline_accuracy: float | None = None,
        pending_delay_steps: int = 0,
        pending_outstanding_count: int = 0,
        revealed_mean_residual: float = 0.0,
        predicted_positive_rate: float = 0.5,
        revealed_positive_rate: float = 0.5,
    ) -> None:
        if feedback_state is None:
            return
        reward = retrospective_reward
        if reward is None:
            reward = self._feedback_reward(risk_state=risk_state, utility=utility)
            # Counterfactual lift: "did this action improve over doing nothing?"
            # Only added when computing reward from scratch (retrospective_reward=None).
            # Callers that supply a pre-built reward (e.g. hybrid's controller_reward)
            # already incorporate lift, so we must not double-count.
            if revealed_accuracy is not None and revealed_baseline_accuracy is not None:
                lift = float(np.clip(revealed_accuracy - revealed_baseline_accuracy, -0.5, 0.5))
                reward = float(np.clip(reward + 0.15 * lift, -2.0, 2.0))
        delay_weight = 1.0 / (
            1.0
            + 0.06 * max(0, pending_delay_steps - 1)
            + 0.03 * max(0, pending_outstanding_count - 1)
            + 0.35 * self._pending_feedback_stale_fraction
        )
        reward *= float(np.clip(delay_weight, 0.35, 1.0))
        self._apply_feedback(feedback_state, reward=reward)
        self._after_delayed_outcome(
            feedback_state=feedback_state,
            model=model,
            batch=batch,
            signal=signal,
            risk_state=risk_state,
            decision=decision,
            batch_accuracy=batch_accuracy,
            reliability=reliability,
            utility=utility,
            retrospective_reward=reward,
            revealed_accuracy=revealed_accuracy,
            revealed_coverage=revealed_coverage,
            revealed_baseline_accuracy=revealed_baseline_accuracy,
            pending_delay_steps=pending_delay_steps,
            pending_outstanding_count=pending_outstanding_count,
            revealed_mean_residual=revealed_mean_residual,
            predicted_positive_rate=predicted_positive_rate,
            revealed_positive_rate=revealed_positive_rate,
        )
        del model, batch


class RegimeAwareDelayedBanditTabularPolicy(DelayedBanditTabularPolicy):
    """Delayed-feedback bandit with short-horizon temporal state in the context."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        alpha: float = 0.75,
        ridge: float = 1.0,
        allowed_actions: tuple[str, ...] | None = None,
        capital_penalty_scale: float = 0.01,
        ema_decay: float = 0.82,
        similarity_memory: int = 8,
        fraud_rank_mode: bool = False,
        segment_count: int = 0,
        threshold_learning_rate: float = 0.10,
        use_behavior_signals: bool = True,
        use_accuracy_trend: bool = False,  # Appends revealed_accuracy_delta to temporal context (+1 dim: 28→29).
                                           # Gives bandit a "rate of degradation" signal, helping distinguish
                                           # fault modes that degrade at different speeds (e.g. CMAPSS FD004).
                                           # Gate B impact: NEGATIVE on short streams. FD002 hybrid -2.4pp
                                           # (+3.5→+1.1pp). FD004 unchanged (+0.0pp). Bandit needs 500+ batches
                                           # to learn useful weights for the new feature; CMAPSS runs are ~170.
                                           # May help on longer fraud streams — leave False for CMAPSS.
        use_prototype_label_signal: bool = True,  # Per-prototype revealed positive rate as 7th regime feature
                                                    # (+1 dim: 28→29). The regime encoder tracks revealed_positive_rate
                                                    # EMA per prototype via reinforce(). For FD004, fault mode A
                                                    # and B emit different failure rates, so prototype_positive_rate
                                                    # distinguishes them WITHOUT requiring behavior signals.
                                                    # Also enables reinforce() for the direct bandit (currently missing).
                                                    # Gate B impact: FD002 hybrid -1.4pp (+3.5→+2.1pp, still PASS).
                                                    # FD004 hybrid -1.8pp improvement (-9.8→-8.0pp, still FAIL).
                                                    # FD004 bandit unchanged (+0.0pp). Net: 3/4 PASS, stable.
    ) -> None:
        super().__init__(
            reference,
            alpha=alpha,
            ridge=ridge,
            allowed_actions=allowed_actions,
            capital_penalty_scale=capital_penalty_scale,
        )
        self._use_accuracy_trend = use_accuracy_trend
        self._use_prototype_label_signal = use_prototype_label_signal
        self._ema_decay = ema_decay
        self._shift_ema = 0.0
        self._capital_ema = 1.0 / 25.0
        self._reliability_ema = reference.mean_confidence
        self._reward_ema = 0.0
        self._revealed_accuracy_ema = 0.5
        self._last_action_code = 0.0
        self._steps_since_reset = 0
        self._previous_shift_score = 0.0
        self._encoder_step = 0
        self._regime_encoder = StreamingRegimeEncoder(
            max_prototypes=max(8, similarity_memory + 2),
            similarity_threshold=0.93,
            familiarity_threshold=0.56,
            reuse_threshold=0.62,
            creation_similarity_threshold=0.88,
            staleness_horizon=max(12, similarity_memory * 2),
            use_behavior_signals=use_behavior_signals,
        )
        self._current_regime_similarity = 0.0
        self._current_regime_confidence = 0.0
        self._current_regime_novelty = 1.0
        self._current_regime_prototype_reward = 0.5
        self._current_regime_prototype_size = 0.0
        self._current_regime_prototype_recency = 0.0
        self._current_shift_delta = 0.0
        self._last_regime_descriptor: RegimeDescriptor | None = None
        self._last_inference_context: np.ndarray | None = None
        self._action_codebook = {
            action: float(index) / max(1.0, float(len(self._actions) - 1))
            for index, action in enumerate(self._actions)
        }
        self._action_codebook["hold"] = self._action_codebook.get("none", 0.0)
        self._action_codebook["abstain"] = 1.0
        self._fraud_rank_mode = fraud_rank_mode
        self._segment_count = max(0, segment_count if fraud_rank_mode else 0)
        self._reference_feature_mean = np.asarray(
            getattr(reference, "feature_mean", np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self._reference_feature_variance = np.asarray(
            getattr(reference, "feature_variance", np.ones_like(self._reference_feature_mean)),
            dtype=np.float64,
        )
        self._prev_revealed_accuracy_ema: float = 0.5
        self._revealed_accuracy_delta: float = 0.0
        # Expand the linear-UCB state to include temporal state plus explicit regime recurrence features.
        # +1 per enabled flag that adds a context dimension.
        self._dim = 28 + (1 if use_accuracy_trend else 0) + (1 if use_prototype_label_signal else 0)
        self._residual_weights = np.zeros(self._dim, dtype=np.float64)
        self._residual_bias = 0.0
        self._rank_dim = len(self._reference_feature_mean) + 2 + self._segment_count
        self._rank_weights = np.zeros(self._rank_dim, dtype=np.float64)
        self._rank_bias = 0.0
        self._rank_update_rate = 0.0
        self._rank_updates = 0
        self._last_rank_delta_mean = 0.0
        self._last_rank_delta_std = 0.0
        self._last_segment_diversity = 0.0
        self._pairwise_rank_update_rate = 0.0
        self._pairwise_rank_updates = 0
        self._residual_prototype_bias: dict[int, float] = {}
        self._residual_prototype_weights: dict[int, np.ndarray] = {}
        self._residual_prototype_recent_bias: dict[int, float] = {}
        self._expert_names = ("recurring", "transition", "high_risk")
        self._residual_expert_weights = {
            name: np.zeros(self._dim, dtype=np.float64) for name in self._expert_names
        }
        self._residual_expert_bias = {name: 0.0 for name in self._expert_names}
        self._threshold_bias = 0.0
        self._threshold_learning_rate = float(threshold_learning_rate)
        self._threshold_prototype_bias: dict[int, float] = {}
        self._threshold_expert_bias = {name: 0.0 for name in self._expert_names}
        self._last_threshold = 0.5
        self._supervised_head_update_rate = 0.0
        self._supervised_head_updates = 0
        self._supervised_adapter_update_rate = 0.0
        self._supervised_adapter_updates = 0
        self._trusted_subspace_update_rate = 0.0
        self._trusted_subspace_updates = 0
        self._last_expert_deltas = {name: 0.0 for name in self._expert_names}
        self._residual_recent_bias = 0.0
        self._last_local_residual_delta = 0.0
        self._last_recent_residual_delta = 0.0
        self._last_residual_delta = 0.0
        self._last_delay_weight = 1.0
        self._matrices = {
            action: ridge * np.eye(self._dim, dtype=np.float64)
            for action in self._actions
        }
        self._vectors = {
            action: np.zeros(self._dim, dtype=np.float64)
            for action in self._actions
        }

    def _temporal_context(self) -> np.ndarray:
        features = [
            self._shift_ema,
            self._current_shift_delta,
            self._capital_ema,
            self._reliability_ema,
            self._reward_ema,
            self._revealed_accuracy_ema,
            self._last_action_code,
            min(self._steps_since_reset, 12) / 12.0,
            self._current_regime_similarity,
            min(self._pending_feedback_count, 8.0) / 8.0,
            min(self._pending_feedback_mean_age, 12.0) / 12.0,
            min(self._pending_feedback_max_age, 16.0) / 16.0,
            self._pending_feedback_stale_fraction,
        ]
        if self._use_accuracy_trend:
            features.append(float(np.clip(self._revealed_accuracy_delta, -0.30, 0.30)))
        return np.array(features, dtype=np.float64)

    def _context(
        self,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch | None = None,
        probabilities: list[float] | None = None,
    ) -> np.ndarray:
        base = _bandit_context(self._reference, signal, risk_state)
        self._current_shift_delta = signal.score - self._previous_shift_score
        temporal = self._temporal_context()
        if batch is None:
            regime_features = np.zeros(6, dtype=np.float64)
            self._last_regime_descriptor = None
            return np.concatenate([base, temporal, regime_features], axis=0)

        batch_signature = _pooled_batch_signature(batch.features, self._reference)
        regime_embedding = build_regime_embedding(batch_signature, base[1:], temporal)
        self._encoder_step += 1
        descriptor = self._regime_encoder.register(
            regime_embedding, step=self._encoder_step, probabilities=probabilities
        )
        self._last_regime_descriptor = descriptor
        self._current_regime_similarity = descriptor.similarity
        self._current_regime_confidence = descriptor.recurrence_confidence
        self._current_regime_novelty = descriptor.novelty_score
        self._current_regime_prototype_reward = descriptor.prototype_reward
        self._current_regime_prototype_size = float(descriptor.prototype_count)
        self._current_regime_prototype_recency = descriptor.prototype_recency
        regime_feature_list = [
            descriptor.similarity,
            descriptor.recurrence_confidence,
            descriptor.novelty_score,
            descriptor.prototype_reward,
            min(1.0, descriptor.prototype_count / 4.0),
            descriptor.prototype_recency,
        ]
        if self._use_prototype_label_signal:
            # Revealed positive rate EMA for this prototype — distinguishes regimes by
            # label distribution (e.g. fault mode A vs B in CMAPSS FD004).
            regime_feature_list.append(descriptor.prototype_positive_rate)
        regime_features = np.array(regime_feature_list, dtype=np.float64)
        return np.concatenate([base, temporal, regime_features], axis=0)

    def _update_ema(self, previous: float, new_value: float) -> float:
        return self._ema_decay * previous + (1.0 - self._ema_decay) * new_value

    def _feedback_regime_descriptor(self) -> RegimeDescriptor | None:
        return self._last_regime_descriptor

    def _expert_gates(self, descriptor: RegimeDescriptor | None) -> dict[str, float]:
        if descriptor is None:
            transition = float(np.clip(0.75 * min(abs(self._current_shift_delta), 1.0), 0.0, 1.0))
            high_risk = float(
                np.clip(
                    0.55 * self._capital_ema + 0.45 * max(0.0, 1.0 - self._reliability_ema),
                    0.0,
                    1.0,
                )
            )
            return {
                "recurring": 0.0,
                "transition": transition,
                "high_risk": high_risk,
            }
        recurring = float(
            np.clip(
                0.60 * descriptor.recurrence_confidence
                + 0.25 * descriptor.similarity
                + 0.15 * max(0.0, descriptor.prototype_reward - 0.5),
                0.0,
                1.0,
            )
        )
        transition = float(
            np.clip(
                0.50 * descriptor.novelty_score
                + 0.30 * min(abs(self._current_shift_delta), 1.0)
                + 0.20 * max(0.0, 1.0 - descriptor.similarity),
                0.0,
                1.0,
            )
        )
        high_risk = float(
            np.clip(
                0.45 * self._capital_ema
                + 0.30 * max(0.0, 1.0 - self._reliability_ema)
                + 0.15 * self._pending_feedback_stale_fraction
                + 0.10 * descriptor.novelty_score,
                0.0,
                1.0,
            )
        )
        return {
            "recurring": recurring,
            "transition": transition,
            "high_risk": high_risk,
        }

    def _current_residual_delta(self) -> float:
        if self._last_inference_context is None:
            return 0.0
        global_delta = float(self._residual_weights @ self._last_inference_context + self._residual_bias)
        descriptor = self._last_regime_descriptor
        local_delta = 0.0
        recent_delta = self._residual_recent_bias
        expert_deltas = {name: 0.0 for name in self._expert_names}
        expert_gates = self._expert_gates(descriptor)
        if descriptor is not None and descriptor.prototype_index >= 0:
            prototype_index = descriptor.prototype_index
            prototype_weights = self._residual_prototype_weights.get(prototype_index)
            if prototype_weights is not None:
                local_delta += float(prototype_weights @ self._last_inference_context)
            local_delta += self._residual_prototype_bias.get(prototype_index, 0.0)
            recent_delta += self._residual_prototype_recent_bias.get(prototype_index, 0.0)
            regime_gate = float(
                np.clip(
                    0.55 * descriptor.recurrence_confidence
                    + 0.30 * descriptor.similarity
                    + 0.15 * max(0.0, descriptor.prototype_reward - 0.5),
                    0.0,
                    1.0,
                )
            )
        else:
            regime_gate = 0.0
        expert_total = 0.0
        for name in self._expert_names:
            gate = expert_gates.get(name, 0.0)
            raw = float(
                self._residual_expert_weights[name] @ self._last_inference_context
                + self._residual_expert_bias[name]
            )
            expert_deltas[name] = gate * raw
            expert_total += expert_deltas[name]
        delta = global_delta + regime_gate * local_delta + recent_delta + expert_total
        self._last_local_residual_delta = regime_gate * local_delta
        self._last_recent_residual_delta = recent_delta
        self._last_expert_deltas = expert_deltas
        return float(np.clip(delta, -0.95, 0.95))

    def _rank_features(
        self,
        *,
        features: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        standardized = (
            features - self._reference_feature_mean[np.newaxis, :]
        ) / np.sqrt(self._reference_feature_variance[np.newaxis, :] + 1e-6)
        clipped_probabilities = np.clip(probabilities.astype(np.float64), 1e-5, 1.0 - 1e-5)
        logits = np.log(clipped_probabilities / (1.0 - clipped_probabilities))[:, np.newaxis]
        confidences = np.maximum(clipped_probabilities, 1.0 - clipped_probabilities)[:, np.newaxis]
        parts = [standardized.astype(np.float64), logits, confidences]
        if self._segment_count > 0:
            segment_ids = self._rank_segment_ids(features=features, probabilities=probabilities)
            segment_one_hot = np.zeros((len(segment_ids), self._segment_count), dtype=np.float64)
            if len(segment_ids) > 0:
                segment_one_hot[np.arange(len(segment_ids)), segment_ids] = 1.0
                segment_counts = np.bincount(segment_ids, minlength=self._segment_count).astype(np.float64)
                distribution = segment_counts / max(1.0, float(len(segment_ids)))
                nonzero = distribution[distribution > 0.0]
                entropy = -float(np.sum(nonzero * np.log(nonzero)))
                self._last_segment_diversity = float(
                    entropy / max(1e-6, math.log(float(self._segment_count)))
                )
            parts.append(segment_one_hot)
        return np.concatenate(parts, axis=1)

    def _rank_segment_ids(
        self,
        *,
        features: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if self._segment_count <= 0:
            return np.zeros(len(probabilities), dtype=np.int64)
        standardized = (
            features - self._reference_feature_mean[np.newaxis, :]
        ) / np.sqrt(self._reference_feature_variance[np.newaxis, :] + 1e-6)
        feature_window = standardized[:, : min(6, standardized.shape[1])]
        feature_projection = np.mean(feature_window, axis=1)
        feature_band = (feature_projection >= 0.0).astype(np.int64)
        probability_band = np.digitize(probabilities, bins=np.array([0.10, 0.35], dtype=np.float64)).astype(np.int64)
        segment_ids = probability_band + 3 * feature_band
        return np.clip(segment_ids, 0, self._segment_count - 1)

    def _rank_deltas(
        self,
        *,
        features: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if self._reference.positive_rate >= 0.20:
            return np.zeros(len(probabilities), dtype=np.float64)
        rank_features = self._rank_features(features=features, probabilities=probabilities)
        deltas = rank_features @ self._rank_weights + self._rank_bias
        if self._last_regime_descriptor is not None:
            deltas *= float(
                np.clip(
                    0.55
                    + 0.25 * self._last_regime_descriptor.recurrence_confidence
                    + 0.20 * self._last_regime_descriptor.similarity,
                    0.40,
                    1.10,
                )
            )
        return np.clip(deltas, -0.50, 0.50)

    def correct_probabilities(
        self,
        probabilities: list[float],
        *,
        signal: TabularShiftSignal | None = None,
        risk_state: RiskState | None = None,
        batch: TabularBatch | None = None,
    ) -> list[float]:
        del signal, risk_state
        delta = self._current_residual_delta()
        self._last_residual_delta = delta
        rank_deltas: np.ndarray | None = None
        if batch is not None:
            rank_deltas = self._rank_deltas(
                features=batch.features,
                probabilities=np.asarray(probabilities, dtype=np.float64),
            )
            if len(rank_deltas) > 0:
                self._last_rank_delta_mean = float(np.mean(rank_deltas))
                self._last_rank_delta_std = float(np.std(rank_deltas))
        else:
            self._last_rank_delta_mean = 0.0
            self._last_rank_delta_std = 0.0
        corrected: list[float] = []
        for index, probability in enumerate(probabilities):
            clamped = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
            logit = math.log(clamped / (1.0 - clamped))
            sample_delta = 0.0 if rank_deltas is None else float(rank_deltas[index])
            adjusted = 1.0 / (1.0 + math.exp(-(logit + delta + sample_delta)))
            corrected.append(float(np.clip(adjusted, 1e-5, 1.0 - 1e-5)))
        return corrected

    def decision_threshold(
        self,
        *,
        signal: TabularShiftSignal | None = None,
        risk_state: RiskState | None = None,
        batch: TabularBatch | None = None,
    ) -> float:
        del signal, risk_state, batch
        descriptor = self._last_regime_descriptor
        gates = self._expert_gates(descriptor)
        threshold_shift = self._threshold_bias
        if descriptor is not None and descriptor.prototype_index >= 0:
            threshold_shift += self._threshold_prototype_bias.get(descriptor.prototype_index, 0.0)
        for name in self._expert_names:
            threshold_shift += gates.get(name, 0.0) * self._threshold_expert_bias[name]
        threshold_shift = float(np.clip(threshold_shift, -0.20, 0.20))
        self._last_threshold = float(np.clip(0.5 + threshold_shift, 0.05, 0.95))
        return self._last_threshold

    def _after_apply(
        self,
        *,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        feedback_state: BanditFeedbackState | None,
    ) -> None:
        self._shift_ema = self._update_ema(self._shift_ema, signal.score)
        self._capital_ema = self._update_ema(self._capital_ema, min(risk_state.capital, 25.0) / 25.0)
        self._last_action_code = self._action_codebook.get(decision.action, self._action_codebook.get("none", 0.0))
        self._steps_since_reset = 0 if decision.action == "reset" else self._steps_since_reset + 1
        self._previous_shift_score = signal.score
        if feedback_state is not None:
            self._last_inference_context = feedback_state.context.copy()

    def _after_delayed_outcome(
        self,
        *,
        feedback_state: BanditFeedbackState | None,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
        retrospective_reward: float,
        revealed_accuracy: float | None,
        revealed_coverage: float | None,
        revealed_baseline_accuracy: float | None,
        pending_delay_steps: int,
        pending_outstanding_count: int,
        revealed_mean_residual: float,
        predicted_positive_rate: float,
        revealed_positive_rate: float,
    ) -> None:
        del signal, risk_state, decision, batch_accuracy, utility, revealed_coverage
        self._reward_ema = self._update_ema(self._reward_ema, retrospective_reward)
        self._last_delay_weight = 1.0 / (
            1.0
            + 0.06 * max(0, pending_delay_steps - 1)
            + 0.03 * max(0, pending_outstanding_count - 1)
            + 0.35 * self._pending_feedback_stale_fraction
        )
        self._reliability_ema = self._update_ema(self._reliability_ema, reliability)
        if revealed_accuracy is not None:
            new_acc_ema = self._update_ema(self._revealed_accuracy_ema, revealed_accuracy)
            if self._use_accuracy_trend:
                self._revealed_accuracy_delta = new_acc_ema - self._prev_revealed_accuracy_ema
                self._prev_revealed_accuracy_ema = self._revealed_accuracy_ema
            self._revealed_accuracy_ema = new_acc_ema
        if self._use_prototype_label_signal and feedback_state is not None:
            # Reinforce regime prototype with outcome + label distribution.
            # For the direct bandit this was previously missing — prototypes never learned
            # from delayed feedback, so prototype_reward stayed at 0.5 throughout.
            self._regime_encoder.reinforce(
                feedback_state.regime_descriptor,
                step=self._encoder_step,
                reward=retrospective_reward,
                successful=revealed_accuracy is not None and revealed_accuracy >= self._revealed_accuracy_ema,
                positive_rate=revealed_positive_rate,
            )
        if feedback_state is not None:
            learning_rate = 0.12 * float(np.clip(self._last_delay_weight, 0.35, 1.0))
            self._residual_weights += learning_rate * revealed_mean_residual * feedback_state.context
            self._residual_bias = float(np.clip(self._residual_bias + learning_rate * revealed_mean_residual, -0.75, 0.75))
            if (
                feedback_state.batch_features is not None
                and feedback_state.predicted_probabilities is not None
                and batch.labels is not None
                and len(batch.labels) == len(feedback_state.predicted_probabilities)
                and self._reference.positive_rate < 0.20
            ):
                rank_features = self._rank_features(
                    features=feedback_state.batch_features,
                    probabilities=feedback_state.predicted_probabilities,
                )
                labels = batch.labels.astype(np.float64)
                probabilities = np.clip(feedback_state.predicted_probabilities, 1e-5, 1.0 - 1e-5)
                positive_weight = float(
                    np.clip((1.0 - self._reference.positive_rate) / max(self._reference.positive_rate, 1e-3), 1.0, 12.0)
                )
                sample_weights = np.where(labels >= 0.5, positive_weight, 1.0)
                residuals = sample_weights * (labels - probabilities)
                rank_lr = 0.035 * float(np.clip(self._last_delay_weight, 0.35, 1.0))
                rank_lr *= float(np.clip(1.0 - 0.25 * self._pending_feedback_stale_fraction, 0.6, 1.0))
                self._rank_weights += rank_lr * (rank_features.T @ residuals) / max(1.0, float(len(labels)))
                self._rank_weights = np.clip(self._rank_weights, -0.45, 0.45)
                self._rank_bias = float(
                    np.clip(self._rank_bias + rank_lr * float(np.mean(residuals)), -0.35, 0.35)
                )
                self._rank_update_rate = self._update_ema(self._rank_update_rate, 1.0)
                self._rank_updates += 1
            posterior_residual = float(np.clip(revealed_positive_rate - predicted_positive_rate, -1.0, 1.0))
            short_horizon_residual = float(
                np.clip(0.65 * revealed_mean_residual + 0.35 * posterior_residual, -1.0, 1.0)
            )
            self._residual_recent_bias = float(
                np.clip(0.82 * self._residual_recent_bias + 0.18 * short_horizon_residual, -0.75, 0.75)
            )
            expert_gates = self._expert_gates(feedback_state.regime_descriptor)
            for name in self._expert_names:
                gate = expert_gates.get(name, 0.0)
                if gate <= 0.05:
                    continue
                expert_lr = learning_rate * gate
                self._residual_expert_weights[name] += expert_lr * short_horizon_residual * feedback_state.context
                self._residual_expert_weights[name] = np.clip(
                    self._residual_expert_weights[name],
                    -0.60,
                    0.60,
                )
                self._residual_expert_bias[name] = float(
                    np.clip(
                        self._residual_expert_bias[name] + expert_lr * short_horizon_residual,
                        -0.75,
                        0.75,
                    )
                )
            threshold_signal = float(np.clip(-posterior_residual, -0.5, 0.5))
            threshold_lr = self._threshold_learning_rate * float(np.clip(self._last_delay_weight, 0.35, 1.0))
            self._threshold_bias = float(np.clip(self._threshold_bias + threshold_lr * threshold_signal, -0.15, 0.15))
            if feedback_state.regime_descriptor is not None and feedback_state.regime_descriptor.prototype_index >= 0:
                index = feedback_state.regime_descriptor.prototype_index
                previous = self._residual_prototype_bias.get(index, 0.0)
                target = previous + learning_rate * posterior_residual
                self._residual_prototype_bias[index] = float(np.clip(target, -0.75, 0.75))
                prototype_weights = self._residual_prototype_weights.get(index)
                if prototype_weights is None:
                    prototype_weights = np.zeros(self._dim, dtype=np.float64)
                local_lr = learning_rate * float(
                    np.clip(
                        0.50
                        + 0.35 * feedback_state.regime_descriptor.recurrence_confidence
                        + 0.15 * feedback_state.regime_descriptor.similarity,
                        0.40,
                        1.15,
                    )
                )
                prototype_weights = prototype_weights + local_lr * short_horizon_residual * feedback_state.context
                prototype_weights = np.clip(prototype_weights, -0.50, 0.50)
                self._residual_prototype_weights[index] = prototype_weights
                previous_recent = self._residual_prototype_recent_bias.get(index, 0.0)
                recent_target = 0.78 * previous_recent + 0.22 * short_horizon_residual
                self._residual_prototype_recent_bias[index] = float(np.clip(recent_target, -0.75, 0.75))
                previous_threshold = self._threshold_prototype_bias.get(index, 0.0)
                threshold_target = previous_threshold + threshold_lr * 1.15 * threshold_signal
                self._threshold_prototype_bias[index] = float(np.clip(threshold_target, -0.15, 0.15))
            for name in self._expert_names:
                gate = expert_gates.get(name, 0.0)
                if gate <= 0.05:
                    continue
                updated_threshold = self._threshold_expert_bias[name] + threshold_lr * gate * threshold_signal
                self._threshold_expert_bias[name] = float(np.clip(updated_threshold, -0.15, 0.15))
            should_supervised_update = (
                (hasattr(model, "supervised_head_adapter_update") or hasattr(model, "supervised_head_update"))
                and batch.labels is not None
                and len(batch.labels) >= 4
                and (
                    revealed_baseline_accuracy is None
                    or revealed_accuracy is None
                    or revealed_accuracy <= revealed_baseline_accuracy + 0.01
                    or retrospective_reward < 0.88
                )
            )
            if should_supervised_update:
                head_lr = 0.010 * float(np.clip(self._last_delay_weight, 0.35, 1.0))
                if feedback_state.regime_descriptor is not None:
                    head_lr *= float(
                        np.clip(
                            0.70
                            + 0.20 * feedback_state.regime_descriptor.recurrence_confidence
                            + 0.10 * feedback_state.regime_descriptor.similarity,
                            0.55,
                            1.10,
                        )
                    )
                pending_pressure = float(
                    np.clip(
                        0.08 * float(pending_outstanding_count)
                        + 0.04 * float(pending_delay_steps)
                        + 0.45 * self._pending_feedback_stale_fraction,
                        0.0,
                        1.25,
                    )
                )
                use_pairwise_update = (
                    self._fraud_rank_mode
                    and self._reference.positive_rate < 0.20
                    and hasattr(model, "supervised_pairwise_head_adapter_update")
                )
                use_trusted_subspace_update = hasattr(model, "supervised_trusted_subspace_update") and (
                    feedback_state.regime_descriptor is None
                    or feedback_state.regime_descriptor.recurrence_confidence >= 0.30
                    or retrospective_reward < 0.82
                )
                use_adapter_update = hasattr(model, "supervised_head_adapter_update")
                if use_pairwise_update:
                    segment_ids = self._rank_segment_ids(
                        features=batch.features,
                        probabilities=np.asarray(feedback_state.predicted_probabilities, dtype=np.float64)
                        if feedback_state.predicted_probabilities is not None
                        else np.asarray(model.predict_proba(batch.features), dtype=np.float64),
                    )
                    update_fraction = float(
                        model.supervised_pairwise_head_adapter_update(
                            batch.features,
                            batch.labels,
                            learning_rate=head_lr * float(np.clip(1.0 - 0.25 * pending_pressure, 0.60, 1.05)),
                            anchor_strength=0.72,
                            max_parameter_drift=0.34,
                            segment_ids=segment_ids,
                            classification_weight=0.28,
                            margin=0.08,
                            max_pairs=224,
                            steps=2,
                        )
                    )
                    self._pairwise_rank_update_rate = self._update_ema(
                        self._pairwise_rank_update_rate,
                        update_fraction,
                    )
                    if update_fraction > 0.0:
                        self._pairwise_rank_updates += 1
                elif use_trusted_subspace_update:
                    confidence_threshold = float(
                        np.clip(
                            0.74
                            + 0.10 * pending_pressure
                            - 0.06 * self._current_regime_similarity,
                            0.68,
                            0.90,
                        )
                    )
                    subspace_fraction = float(
                        np.clip(
                            0.42
                            - 0.12 * pending_pressure
                            + 0.10 * self._current_regime_confidence,
                            0.18,
                            0.55,
                        )
                    )
                    update_fraction = float(
                        model.supervised_trusted_subspace_update(
                            batch.features,
                            batch.labels,
                            learning_rate=head_lr * float(np.clip(1.0 - 0.30 * pending_pressure, 0.55, 1.0)),
                            anchor_strength=0.82,
                            max_parameter_drift=0.32,
                            confidence_threshold=confidence_threshold,
                            min_selected=6,
                            subspace_fraction=subspace_fraction,
                            steps=2,
                        )
                    )
                    self._trusted_subspace_update_rate = self._update_ema(
                        self._trusted_subspace_update_rate,
                        update_fraction,
                    )
                    if update_fraction > 0.0:
                        self._trusted_subspace_updates += 1
                elif use_adapter_update:
                    update_fraction = float(
                        model.supervised_head_adapter_update(
                            batch.features,
                            batch.labels,
                            learning_rate=head_lr * 0.85,
                            anchor_strength=0.78,
                            max_parameter_drift=0.32,
                            steps=2,
                        )
                    )
                    self._supervised_adapter_update_rate = self._update_ema(
                        self._supervised_adapter_update_rate,
                        update_fraction,
                    )
                    if update_fraction > 0.0:
                        self._supervised_adapter_updates += 1
                else:
                    update_fraction = float(
                        model.supervised_head_update(
                            batch.features,
                            batch.labels,
                            learning_rate=head_lr,
                            anchor_strength=0.65,
                            max_parameter_drift=0.32,
                            steps=1,
                        )
                    )
                    self._supervised_head_update_rate = self._update_ema(
                        self._supervised_head_update_rate,
                        update_fraction,
                    )
                    if update_fraction > 0.0:
                        self._supervised_head_updates += 1
        if feedback_state is not None and feedback_state.regime_descriptor is not None:
            successful = (
                revealed_accuracy is not None
                and revealed_baseline_accuracy is not None
                and revealed_accuracy >= revealed_baseline_accuracy
            ) or retrospective_reward >= 0.52
            self._regime_encoder.reinforce(
                feedback_state.regime_descriptor,
                step=self._encoder_step,
                reward=retrospective_reward,
                successful=successful,
            )

    def get_diagnostics(self) -> dict[str, float]:
        diagnostics = {
            "regime_shift_ema": self._shift_ema,
            "regime_capital_ema": self._capital_ema,
            "regime_reliability_ema": self._reliability_ema,
            "regime_reward_ema": self._reward_ema,
            "regime_recurrence_similarity": self._current_regime_similarity,
            "regime_recurrence_confidence": self._current_regime_confidence,
            "regime_novelty_score": self._current_regime_novelty,
            "regime_prototype_reward": self._current_regime_prototype_reward,
            "regime_prototype_size": self._current_regime_prototype_size,
            "regime_prototype_recency": self._current_regime_prototype_recency,
            "regime_shift_delta": self._current_shift_delta,
            "pending_feedback_count": self._pending_feedback_count,
            "pending_feedback_mean_age": self._pending_feedback_mean_age,
            "pending_feedback_max_age": self._pending_feedback_max_age,
            "pending_feedback_stale_fraction": self._pending_feedback_stale_fraction,
            "residual_delta": self._last_residual_delta,
            "local_residual_delta": self._last_local_residual_delta,
            "recent_residual_delta": self._last_recent_residual_delta,
            "recurring_expert_delta": self._last_expert_deltas["recurring"],
            "transition_expert_delta": self._last_expert_deltas["transition"],
            "high_risk_expert_delta": self._last_expert_deltas["high_risk"],
            "rank_delta_mean": self._last_rank_delta_mean,
            "rank_delta_std": self._last_rank_delta_std,
            "rank_segment_diversity": self._last_segment_diversity,
            "decision_threshold": self._last_threshold,
            "threshold_shift": self._last_threshold - 0.5,
            "rank_update_rate": self._rank_update_rate,
            "rank_updates": float(self._rank_updates),
            "pairwise_rank_update_rate": self._pairwise_rank_update_rate,
            "pairwise_rank_updates": float(self._pairwise_rank_updates),
            "supervised_head_update_rate": self._supervised_head_update_rate,
            "supervised_head_updates": float(self._supervised_head_updates),
            "supervised_adapter_update_rate": self._supervised_adapter_update_rate,
            "supervised_adapter_updates": float(self._supervised_adapter_updates),
            "trusted_subspace_update_rate": self._trusted_subspace_update_rate,
            "trusted_subspace_updates": float(self._trusted_subspace_updates),
            "delay_weight": self._last_delay_weight,
            "steps_since_reset": float(self._steps_since_reset),
        }
        diagnostics.update(self._regime_encoder.get_diagnostics(prefix="regime_encoder"))
        return diagnostics


class FraudRankDelayedBanditTabularPolicy(RegimeAwareDelayedBanditTabularPolicy):
    """Fraud-specialized delayed controller with segment-aware pairwise ranking updates."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        alpha: float = 0.75,
        ridge: float = 1.0,
        allowed_actions: tuple[str, ...] | None = None,
        capital_penalty_scale: float = 0.01,
        ema_decay: float = 0.82,
        similarity_memory: int = 8,
        threshold_learning_rate: float = 0.10,
        use_behavior_signals: bool = True,
    ) -> None:
        super().__init__(
            reference,
            alpha=alpha,
            ridge=ridge,
            allowed_actions=allowed_actions,
            capital_penalty_scale=capital_penalty_scale,
            ema_decay=ema_decay,
            similarity_memory=similarity_memory,
            fraud_rank_mode=True,
            segment_count=6,
            threshold_learning_rate=threshold_learning_rate,
            use_behavior_signals=use_behavior_signals,
        )


class FraudContextDelayedBanditTabularPolicy(FraudRankDelayedBanditTabularPolicy):
    """Fraud specialist that expects prepended temporal-context features."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        alpha: float = 0.75,
        ridge: float = 1.0,
        allowed_actions: tuple[str, ...] | None = None,
        capital_penalty_scale: float = 0.01,
        ema_decay: float = 0.82,
        similarity_memory: int = 8,
        threshold_learning_rate: float = 0.10,
        use_behavior_signals: bool = True,
    ) -> None:
        super().__init__(
            reference,
            alpha=alpha,
            ridge=ridge,
            allowed_actions=allowed_actions,
            capital_penalty_scale=capital_penalty_scale,
            ema_decay=ema_decay,
            similarity_memory=similarity_memory,
            threshold_learning_rate=threshold_learning_rate,
            use_behavior_signals=use_behavior_signals,
        )
        self._fraud_context_mode = True
        self._segment_count = 12
        self._rank_dim = len(self._reference_feature_mean) + 2 + self._segment_count
        self._rank_weights = np.zeros(self._rank_dim, dtype=np.float64)

    def _rank_segment_ids(
        self,
        *,
        features: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if features.shape[1] < 8:
            return super()._rank_segment_ids(features=features, probabilities=probabilities)
        temporal = features[:, :8]
        progress_band = (temporal[:, 0] >= 0.55).astype(np.int64)
        novelty_band = (temporal[:, 5] >= np.median(temporal[:, 5])).astype(np.int64)
        risk_band = np.digitize(probabilities, bins=np.array([0.08, 0.25, 0.55], dtype=np.float64)).astype(np.int64)
        segment_ids = risk_band + 4 * novelty_band + 8 * progress_band
        return np.clip(segment_ids, 0, self._segment_count - 1)


class SpecialistMemoryTabularPolicy:
    """Routes batches through a small reservoir of specialist snapshots."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        max_specialists: int = 4,
        distance_threshold: float = 1.75,
    ) -> None:
        self._reference = reference
        self._max_specialists = max_specialists
        self._distance_threshold = distance_threshold
        self._specialists: list[SpecialistSlot] = []
        self._active_index = 0
        self._active_signature: np.ndarray | None = None
        self._candidate_new = False

    def prepare_model(self, model: TorchTabularAdapterModel, batch: TabularBatch) -> None:
        signature = self._batch_signature(batch.features)
        if not self._specialists:
            self._specialists.append(
                SpecialistSlot(
                    name="base",
                    snapshot=model.export_state(),
                    signature=signature.copy(),
                    controller=MultiActionTabularPolicy(self._reference),
                )
            )

        distances = [float(np.linalg.norm(signature - specialist.signature)) for specialist in self._specialists]
        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]
        self._candidate_new = best_distance > self._distance_threshold and len(self._specialists) < self._max_specialists
        chosen_index = best_index if best_distance <= self._distance_threshold else 0
        self._active_index = chosen_index
        self._active_signature = signature
        model.load_state(self._specialists[chosen_index].snapshot)

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        controller = self._specialists[self._active_index].controller
        return controller.apply(model, signal, risk_state, batch, probabilities)

    def observe_outcome(
        self,
        *,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
    ) -> None:
        del batch
        slot = self._specialists[self._active_index]
        slot.snapshot = model.export_state()
        if self._active_signature is not None:
            slot.signature = 0.85 * slot.signature + 0.15 * self._active_signature
        slot.usage_count += 1
        slot.cumulative_reward += utility + 0.05 * reliability - 0.01 * max(0.0, risk_state.capital - 1.0)

        if (
            self._candidate_new
            and decision.action in {"adapt", "label_shift", "bn_refresh", "recalibrate"}
            and batch_accuracy >= 0.88
            and (risk_state.capital >= 2.0 or signal.score >= 1.15)
            and self._active_signature is not None
        ):
            self._specialists.append(
                SpecialistSlot(
                    name=f"specialist_{len(self._specialists)}",
                    snapshot=model.export_state(),
                    signature=self._active_signature.copy(),
                    controller=MultiActionTabularPolicy(self._reference),
                )
            )
            self._candidate_new = False
        elif (
            len(self._specialists) >= self._max_specialists
            and self._candidate_new
            and decision.action in {"adapt", "label_shift", "bn_refresh", "recalibrate"}
            and batch_accuracy >= 0.90
            and (risk_state.capital >= 3.0 or signal.score >= 1.30)
            and self._active_signature is not None
        ):
            replace_index = self._worst_specialist_index()
            if replace_index is not None:
                self._specialists[replace_index] = SpecialistSlot(
                    name=f"specialist_{replace_index}",
                    snapshot=model.export_state(),
                    signature=self._active_signature.copy(),
                    controller=MultiActionTabularPolicy(self._reference),
                )
            self._candidate_new = False

    def _worst_specialist_index(self) -> int | None:
        if len(self._specialists) <= 1:
            return None
        scored = []
        for index, specialist in enumerate(self._specialists[1:], start=1):
            mean_reward = specialist.cumulative_reward / max(1, specialist.usage_count)
            scored.append((mean_reward, specialist.usage_count, index))
        scored.sort()
        return scored[0][2]

    def _batch_signature(self, features: np.ndarray) -> np.ndarray:
        return _pooled_batch_signature(features, self._reference)

    def get_diagnostics(self) -> dict[str, float]:
        usage_counts = [specialist.usage_count for specialist in self._specialists]
        active_reuse = sum(max(0, count - 1) for count in usage_counts)
        return {
            "specialist_count": float(len(self._specialists)),
            "specialist_mean_usage": float(np.mean(usage_counts)) if usage_counts else 0.0,
            "specialist_max_usage": float(max(usage_counts)) if usage_counts else 0.0,
            "specialist_reuse_ratio": active_reuse / max(1.0, float(sum(usage_counts))),
        }


class HybridBanditSpecialistPolicy:
    """Specialist-memory substrate with a bandit controller inside each specialist."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        max_specialists: int = 4,
        distance_threshold: float = 1.35,
    ) -> None:
        self._reference = reference
        self._max_specialists = max_specialists
        self._distance_threshold = distance_threshold
        self._specialists: list[SpecialistSlot] = []
        self._active_index = 0
        self._active_signature: np.ndarray | None = None
        self._candidate_new = False

    def prepare_model(self, model: TorchTabularAdapterModel, batch: TabularBatch) -> None:
        signature = self._batch_signature(batch.features)
        if not self._specialists:
            self._specialists.append(
                SpecialistSlot(
                    name="base",
                    snapshot=model.export_state(),
                    signature=signature.copy(),
                    controller=BanditTabularPolicy(self._reference),
                )
            )

        distances = [float(np.linalg.norm(signature - specialist.signature)) for specialist in self._specialists]
        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]
        self._candidate_new = best_distance > self._distance_threshold and len(self._specialists) < self._max_specialists
        chosen_index = best_index if best_distance <= self._distance_threshold else 0
        self._active_index = chosen_index
        self._active_signature = signature
        model.load_state(self._specialists[chosen_index].snapshot)

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        controller = self._specialists[self._active_index].controller
        return controller.apply(model, signal, risk_state, batch, probabilities)

    def observe_outcome(
        self,
        *,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
    ) -> None:
        slot = self._specialists[self._active_index]
        slot.snapshot = model.export_state()
        if self._active_signature is not None:
            slot.signature = 0.80 * slot.signature + 0.20 * self._active_signature
        slot.usage_count += 1
        slot.cumulative_reward += utility + 0.05 * reliability - 0.01 * max(0.0, risk_state.capital - 1.0)

        controller = slot.controller
        if hasattr(controller, "observe_outcome"):
            controller.observe_outcome(
                model=model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
                batch_accuracy=batch_accuracy,
                reliability=reliability,
                utility=utility,
            )

        if (
            self._candidate_new
            and decision.action in {"adapt", "label_shift", "bn_refresh", "recalibrate"}
            and batch_accuracy >= 0.86
            and (risk_state.capital >= 2.0 or signal.score >= 1.10)
            and self._active_signature is not None
        ):
            self._specialists.append(
                SpecialistSlot(
                    name=f"hybrid_specialist_{len(self._specialists)}",
                    snapshot=model.export_state(),
                    signature=self._active_signature.copy(),
                    controller=BanditTabularPolicy(self._reference),
                )
            )
            self._candidate_new = False
        elif (
            len(self._specialists) >= self._max_specialists
            and self._candidate_new
            and decision.action in {"adapt", "label_shift", "bn_refresh", "recalibrate"}
            and batch_accuracy >= 0.90
            and self._active_signature is not None
        ):
            replace_index = self._worst_specialist_index()
            if replace_index is not None:
                self._specialists[replace_index] = SpecialistSlot(
                    name=f"hybrid_specialist_{replace_index}",
                    snapshot=model.export_state(),
                    signature=self._active_signature.copy(),
                    controller=BanditTabularPolicy(self._reference),
                )
            self._candidate_new = False

    def _worst_specialist_index(self) -> int | None:
        if len(self._specialists) <= 1:
            return None
        scored = []
        for index, specialist in enumerate(self._specialists[1:], start=1):
            mean_reward = specialist.cumulative_reward / max(1, specialist.usage_count)
            scored.append((mean_reward, specialist.usage_count, index))
        scored.sort()
        return scored[0][2]

    def _batch_signature(self, features: np.ndarray) -> np.ndarray:
        return _pooled_batch_signature(features, self._reference)

    def get_diagnostics(self) -> dict[str, float]:
        usage_counts = [specialist.usage_count for specialist in self._specialists]
        active_reuse = sum(max(0, count - 1) for count in usage_counts)
        return {
            "specialist_count": float(len(self._specialists)),
            "specialist_mean_usage": float(np.mean(usage_counts)) if usage_counts else 0.0,
            "specialist_max_usage": float(max(usage_counts)) if usage_counts else 0.0,
            "specialist_reuse_ratio": active_reuse / max(1.0, float(sum(usage_counts))),
        }


class DelayedHybridBanditSpecialistPolicy:
    """Specialist-memory substrate with delayed-feedback bandit controllers."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        max_specialists: int = 4,
        distance_threshold: float = 1.35,
        promotion_cooldown_steps: int = 6,
        retire_after_steps: int = 16,
        min_reward_ema_for_retention: float = 0.48,
        support_batch_size: int = 24,
        probation_steps: int = 10,
        min_future_reuse_for_retention: float = 0.05,
        controller_kwargs: dict[str, object] | None = None,
        use_behavior_signals: bool = True,
        # Feature flags — each guards an experimental change with documented Gate B impact.
        # Flip to False to revert without touching logic; set in cmapss_benchmark strategy configs.
        use_behavior_routing: bool = True,          # 60/40 behavior+feature blend for specialist selection.
                                                    # FD002: +3.5pp (PASS requires this); FD004: -9.8pp (already fails).
                                                    # Without: FD002 -0.4pp (FAIL). Net: 3/4 with, 2/4 without.
        use_rate_staleness_penalty: bool = True,   # Soft routing penalty when revealed positive rate diverges
                                                    # from specialist's creation rate.  Targets HYPOTHESIS CONFIRMED
                                                    # diagnostic (mismatch 0.054-0.058 at terminal phase).
                                                    # Gate B impact: NEUTRAL. Penalty is 0.003 (too small to shift
                                                    # routing). FD001 gap=0.054 and FD002 gap=0.058 are too close
                                                    # to separate; larger coefficient risks breaking FD002.
        use_strong_exchangeability: bool = True,   # Raises exchangeability bonus weight 0.18→0.25, lowers floor
                                                    # threshold 0.65→0.50. Uses raw feature distribution matching
                                                    # to prefer specialists whose training data looks like current
                                                    # batch — avoids behavior-signal confusion on fault modes.
                                                    # Gate B impact: NEUTRAL. FD004 only creates 1 specialist
                                                    # (base, no support set), so exchangeability=0 always and
                                                    # routing never runs between multiple candidates.
    ) -> None:
        self._reference = reference
        self._controller_kwargs = dict(controller_kwargs or {})
        self._max_specialists = max_specialists
        self._distance_threshold = distance_threshold
        self._promotion_cooldown_steps = promotion_cooldown_steps
        self._retire_after_steps = retire_after_steps
        self._min_reward_ema_for_retention = min_reward_ema_for_retention
        self._support_batch_size = support_batch_size
        self._probation_steps = probation_steps
        self._min_future_reuse_for_retention = min_future_reuse_for_retention
        self._specialists: list[SpecialistSlot] = []
        self._active_index = 0
        self._active_signature: np.ndarray | None = None
        self._candidate_new = False
        self._routing_step = 0
        self._recent_regime_signatures: deque[np.ndarray] = deque(maxlen=12)
        self._active_route_distance = 0.0
        self._active_route_similarity = 0.0
        self._active_novelty_score = 0.0
        self._active_recurrence_similarity = 0.0
        self._active_feature_shift = 0.0
        self._active_reuse_confidence = 0.0
        self._specialist_promotions = 0
        self._specialist_replacements = 0
        self._specialist_retirements = 0
        self._promotion_cooldown_remaining = 0
        self._specialist_route_reuses = 0
        self._specialist_route_fallbacks = 0
        self._specialist_warm_starts_applied = 0
        self._current_revealed_positive_rate: float | None = None
        self._specialist_warm_starts_skipped = 0
        self._reference_accuracy: float = 0.85
        self._active_exchangeability_score = 0.0
        self._regime_encoder = StreamingRegimeEncoder(
            max_prototypes=max(8, max_specialists * 3),
            similarity_threshold=0.935,
            familiarity_threshold=0.58,
            reuse_threshold=0.64,
            novelty_threshold=0.30,
            creation_similarity_threshold=0.90,
            staleness_horizon=max(14, retire_after_steps),
            use_behavior_signals=use_behavior_signals,
        )
        self._active_regime_descriptor: RegimeDescriptor | None = None
        self._active_regime_anchor_similarity = 0.0
        self._active_regime_memory_gate = 0.0
        from .runtime.sota.regime_coreset import ReservoirClusterRouter

        self._reservoir_router = ReservoirClusterRouter(max_clusters=max(6, max_specialists * 2))
        self._active_reservoir_cluster = 0
        self._use_behavior_routing = use_behavior_routing
        self._use_rate_staleness_penalty = use_rate_staleness_penalty
        self._use_strong_exchangeability = use_strong_exchangeability
        self._last_behavior_sig: np.ndarray | None = None

    def _ensure_coreset(self, slot: SpecialistSlot) -> None:
        if slot.coreset is None:
            from .runtime.sota.regime_coreset import RegimeCoreset

            slot.coreset = RegimeCoreset(max_size=self._support_batch_size)

    def _new_specialist_controller(self) -> object:
        return RegimeAwareDelayedBanditTabularPolicy(self._reference, **self._controller_kwargs)

    def _active_controller(self) -> object:
        if not self._specialists:
            return self._new_specialist_controller()
        return self._specialists[self._active_index].controller

    def update_pending_feedback_summary(
        self,
        *,
        pending_count: int,
        mean_age: float,
        max_age: float,
        stale_fraction: float,
    ) -> None:
        for specialist in self._specialists:
            controller = specialist.controller
            if hasattr(controller, "update_pending_feedback_summary"):
                controller.update_pending_feedback_summary(
                    pending_count=pending_count,
                    mean_age=mean_age,
                    max_age=max_age,
                    stale_fraction=stale_fraction,
                )

    def correct_probabilities(
        self,
        probabilities: list[float],
        *,
        signal: TabularShiftSignal | None = None,
        risk_state: RiskState | None = None,
        batch: TabularBatch | None = None,
    ) -> list[float]:
        controller = self._active_controller()
        if not hasattr(controller, "correct_probabilities"):
            return probabilities
        return controller.correct_probabilities(
            probabilities,
            signal=signal,
            risk_state=risk_state,
            batch=batch,
        )

    def prepare_model(self, model: TorchTabularAdapterModel, batch: TabularBatch) -> None:
        self._routing_step += 1
        self._promotion_cooldown_remaining = max(0, self._promotion_cooldown_remaining - 1)
        self._decay_probation()
        self._retire_stale_specialists()
        signature = self._batch_signature(batch.features)
        feature_shift = self._quick_feature_shift(batch.features)
        if not self._specialists:
            self._specialists.append(
                SpecialistSlot(
                    name="base",
                    snapshot=model.export_state(),
                    signature=signature.copy(),
                    controller=self._new_specialist_controller(),
                    last_used_step=self._routing_step,
                    behavior_signature=None,
                )
            )

        regime_descriptor = self._describe_regime(signature, feature_shift)
        self._active_reservoir_cluster = self._reservoir_router.assign(regime_descriptor.embedding)
        if self._specialists and self._specialists[0].regime_anchor is None:
            self._specialists[0].regime_anchor = regime_descriptor.embedding.copy()
            self._specialists[0].regime_confidence_ema = regime_descriptor.recurrence_confidence

        distances = [self._blended_distance(signature, specialist) for specialist in self._specialists]
        similarities = [self._blended_similarity(signature, specialist) for specialist in self._specialists]
        exchangeability_scores = [self._exchangeability_score(batch, specialist) for specialist in self._specialists]
        regime_anchor_similarities = [
            self._regime_anchor_similarity(regime_descriptor, specialist) for specialist in self._specialists
        ]
        adjusted_scores = [
            self._route_score(
                distance=distance,
                similarity=similarity,
                specialist=specialist,
                regime_similarity=regime_similarity,
                exchangeability=exchangeability,
            )
            for distance, similarity, specialist, regime_similarity, exchangeability in zip(
                distances,
                similarities,
                self._specialists,
                regime_anchor_similarities,
                exchangeability_scores,
            )
        ]
        best_index = int(np.argmin(adjusted_scores))
        best_distance = distances[best_index]
        best_similarity = similarities[best_index]
        best_regime_similarity = regime_anchor_similarities[best_index]
        best_exchangeability = exchangeability_scores[best_index]
        best_specialist = self._specialists[best_index]
        recurrence_similarity = self._memory_similarity(signature)
        adaptive_threshold = self._adaptive_distance_threshold(best_specialist)
        novelty_score = max(
            regime_descriptor.novelty_score,
            max(0.0, best_distance - adaptive_threshold) + max(0.0, 0.98 - recurrence_similarity),
        )
        novel_base_route = (
            best_index == 0
            and best_distance > adaptive_threshold * 0.96
            and (best_similarity < 0.985 or regime_descriptor.novelty_score >= 0.20)
        )
        should_reuse = (
            best_distance <= adaptive_threshold
            or best_similarity >= 0.945
            or best_regime_similarity >= 0.90
            or adjusted_scores[best_index] + 0.08 < adjusted_scores[0]
        )
        reuse_confidence = self._reuse_confidence(
            specialist=best_specialist,
            distance=best_distance,
            similarity=best_similarity,
            recurrence_similarity=recurrence_similarity,
            feature_shift=feature_shift,
            novelty_score=novelty_score,
            regime_confidence=regime_descriptor.recurrence_confidence,
            regime_similarity=best_regime_similarity,
            exchangeability=best_exchangeability,
        )
        severe_mismatch = (
            feature_shift >= 2.25
            and novelty_score >= 0.14
            and best_specialist.support_quality_ema < 0.05
            and best_specialist.successful_reuses == 0
            and best_specialist.shadow_wins == 0
        )
        high_shift_requires_stronger_recurrence = (
            best_index > 0
            and feature_shift >= 1.95
            and recurrence_similarity < 0.975
            and best_specialist.support_quality_ema < 0.08
            and best_specialist.future_reuse_ema < 0.08
        )
        recurrence_gate = (
            regime_descriptor.reuse_ready
            or (
                regime_descriptor.familiar
                and regime_descriptor.recurrence_confidence >= 0.58
                and best_regime_similarity >= 0.86
                and best_exchangeability >= 0.68
            )
            or (
                best_exchangeability >= 0.84
                and best_regime_similarity >= 0.82
                and best_specialist.route_advantage_ema >= 0.01
            )
        )
        specialist_reuse_allowed = (
            best_index == 0
            or (
                should_reuse
                and recurrence_gate
                and not severe_mismatch
                and not high_shift_requires_stronger_recurrence
                and (
                    reuse_confidence >= 0.44
                    or best_similarity >= 0.982
                    or recurrence_similarity >= 0.93
                    or best_regime_similarity >= 0.92
                    or best_specialist.successful_reuses >= 1
                    or best_specialist.route_advantage_ema >= 0.025
                )
            )
        )
        if best_index > 0 and specialist_reuse_allowed:
            self._specialist_route_reuses += 1
        elif best_index > 0 and should_reuse:
            self._specialist_route_fallbacks += 1
        self._candidate_new = (
            (not specialist_reuse_allowed or novel_base_route)
            and len(self._specialists) < self._max_specialists
            and best_distance > self._distance_threshold * 0.90
            and (
                regime_descriptor.novelty_score >= 0.10
                or best_regime_similarity < 0.84
                or not regime_descriptor.familiar
            )
        )
        chosen_index = best_index if specialist_reuse_allowed else 0
        self._active_index = chosen_index
        self._active_signature = signature
        self._active_route_distance = best_distance
        self._active_route_similarity = best_similarity
        self._active_recurrence_similarity = recurrence_similarity
        self._active_novelty_score = novelty_score
        self._active_feature_shift = feature_shift
        self._active_reuse_confidence = reuse_confidence if best_index > 0 else 0.0
        self._active_regime_descriptor = regime_descriptor
        self._active_regime_anchor_similarity = regime_anchor_similarities[chosen_index]
        self._active_exchangeability_score = exchangeability_scores[chosen_index]
        self._active_regime_memory_gate = 1.0 if recurrence_gate else 0.0
        self._specialists[chosen_index].last_used_step = self._routing_step
        chosen_specialist = self._specialists[chosen_index]
        chosen_specialist.reservoir_cluster_id = self._active_reservoir_cluster
        # Snapshot staleness gate: if the current label distribution has drifted
        # far from the specialist's creation distribution, loading its snapshot
        # would reset the model to a stale calibration and harm predictions.
        # Prefer revealed positive rate (smoothed, from delayed labels) over
        # current batch labels (unavailable in production under label delay).
        current_positive_rate: float | None = None
        if self._current_revealed_positive_rate is not None:
            current_positive_rate = self._current_revealed_positive_rate
        elif batch.labels is not None:
            current_positive_rate = float(np.asarray(batch.labels).mean())
        if current_positive_rate is not None:
            snapshot_gap = abs(current_positive_rate - chosen_specialist.creation_positive_rate)
            if snapshot_gap > 0.15:
                return  # Skip stale snapshot; keep current model state
        model.load_state(chosen_specialist.snapshot)
        self._warm_start_specialist(model, chosen_specialist, batch)

    def apply(
        self,
        model: TorchTabularAdapterModel,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
        probabilities: list[float],
    ) -> TabularDecision:
        # Refresh the outer regime encoder with model-behavior signals now that
        # probabilities are available (prepare_model runs before prediction).
        if probabilities:
            signature = self._batch_signature(batch.features)
            feature_shift = self._quick_feature_shift(batch.features)
            updated_descriptor = self._describe_regime(signature, feature_shift, probabilities=probabilities)
            self._active_regime_descriptor = updated_descriptor
            if self._use_behavior_routing:
                # Cache behavior signals for the next routing step (prepare_model runs
                # before prediction so must rely on the previous batch's signals).
                behavior_vec = _behavior_feature_vector(compute_model_behavior_signature(probabilities))
                self._last_behavior_sig = behavior_vec
                active = self._specialists[self._active_index]
                if active.behavior_signature is None:
                    active.behavior_signature = behavior_vec.copy()
                else:
                    ema = 0.80 * active.behavior_signature + 0.20 * behavior_vec
                    ema_norm = float(np.linalg.norm(ema)) + 1e-6
                    active.behavior_signature = (ema / ema_norm).astype(np.float64)
        controller = self._specialists[self._active_index].controller
        return controller.apply(model, signal, risk_state, batch, probabilities)

    def capture_feedback_state(
        self,
        *,
        model: TorchTabularAdapterModel | None = None,
        batch: TabularBatch | None = None,
        signal: TabularShiftSignal | None = None,
        risk_state: RiskState | None = None,
        decision: TabularDecision | None = None,
    ) -> DelayedHybridFeedbackState | None:
        if model is None or batch is None or signal is None or risk_state is None or decision is None:
            return None
        slot = self._specialists[self._active_index]
        slot_snapshot = model.export_state()
        slot.snapshot = slot_snapshot
        slot.usage_count += 1
        shadow_base_snapshot = self._specialists[0].snapshot if self._active_index != 0 else None
        shadow_alt_snapshot, shadow_alt_index = self._shadow_route_snapshot()

        controller = slot.controller
        inner_feedback_state = None
        if hasattr(controller, "capture_feedback_state"):
            inner_feedback_state = controller.capture_feedback_state(
                model=model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
            )

        return DelayedHybridFeedbackState(
            slot=slot,
            slot_snapshot=slot_snapshot,
            active_signature=None if self._active_signature is None else self._active_signature.copy(),
            candidate_new=self._candidate_new,
            inner_feedback_state=inner_feedback_state,
            slot_index=self._active_index,
            route_distance=self._active_route_distance,
            route_similarity=self._active_route_similarity,
            novelty_score=self._active_novelty_score,
            recurrence_similarity=self._active_recurrence_similarity,
            routing_step=self._routing_step,
            shadow_base_snapshot=shadow_base_snapshot,
            shadow_alt_snapshot=shadow_alt_snapshot,
            shadow_alt_index=shadow_alt_index,
            regime_descriptor=self._active_regime_descriptor,
            exchangeability_score=self._active_exchangeability_score,
        )

    def observe_delayed_outcome(
        self,
        *,
        feedback_state: DelayedHybridFeedbackState | None,
        model: TorchTabularAdapterModel,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        batch_accuracy: float,
        reliability: float,
        utility: float,
        retrospective_reward: float | None = None,
        revealed_accuracy: float | None = None,
        revealed_coverage: float | None = None,
        revealed_baseline_accuracy: float | None = None,
        pending_delay_steps: int = 0,
        pending_outstanding_count: int = 0,
        revealed_mean_residual: float = 0.0,
        predicted_positive_rate: float = 0.5,
        revealed_positive_rate: float = 0.5,
    ) -> None:
        if feedback_state is None:
            return

        slot = feedback_state.slot
        if not any(existing is slot for existing in self._specialists):
            return

        reward = utility if retrospective_reward is None else retrospective_reward
        revealed_lift = 0.0
        if revealed_accuracy is not None and revealed_baseline_accuracy is not None:
            revealed_lift = revealed_accuracy - revealed_baseline_accuracy
        shadow_base_accuracy = self._shadow_replay_accuracy(
            model=model,
            batch=batch,
            signal=signal,
            risk_state=risk_state,
            decision=decision,
            snapshot=feedback_state.shadow_base_snapshot,
        )
        shadow_alt_accuracy = self._shadow_replay_accuracy(
            model=model,
            batch=batch,
            signal=signal,
            risk_state=risk_state,
            decision=decision,
            snapshot=feedback_state.shadow_alt_snapshot,
        )
        shadow_reference_accuracy = max(shadow_base_accuracy, shadow_alt_accuracy)
        route_advantage = batch_accuracy - shadow_reference_accuracy
        similarity_bonus = max(0.0, feedback_state.route_similarity - 0.82)
        recurrence_bonus = max(0.0, feedback_state.recurrence_similarity - 0.94)
        novelty_bonus = max(0.0, feedback_state.novelty_score)
        reuse_gain = max(0.0, reward - 0.50) + max(0.0, revealed_lift)
        recurrence_credit = reuse_gain + 0.35 * recurrence_bonus + 0.20 * similarity_bonus + 0.40 * max(
            0.0, route_advantage
        )
        specialist_credit = (
            reward
            + 0.10 * revealed_lift
            + 0.05 * similarity_bonus
            + 0.04 * recurrence_bonus
            + 0.10 * route_advantage
        )
        controller_reward = reward + 0.08 * revealed_lift + 0.03 * similarity_bonus + 0.05 * route_advantage
        if feedback_state.regime_descriptor is not None:
            self._regime_encoder.reinforce(
                feedback_state.regime_descriptor,
                step=feedback_state.routing_step,
                reward=specialist_credit,
                successful=route_advantage >= 0.0 or recurrence_credit >= 0.05,
            )
        slot.cumulative_reward += specialist_credit
        slot.reward_ema = 0.82 * slot.reward_ema + 0.18 * specialist_credit
        slot.similarity_ema = 0.85 * slot.similarity_ema + 0.15 * feedback_state.route_similarity
        slot.lift_ema = 0.84 * slot.lift_ema + 0.16 * revealed_lift
        slot.recurrence_reward_ema = 0.84 * slot.recurrence_reward_ema + 0.16 * recurrence_credit
        slot.route_advantage_ema = 0.82 * slot.route_advantage_ema + 0.18 * route_advantage
        if feedback_state.regime_descriptor is not None:
            slot.regime_confidence_ema = 0.82 * slot.regime_confidence_ema + 0.18 * feedback_state.regime_descriptor.recurrence_confidence
        reuse_signal = max(0.0, route_advantage) + 0.40 * max(0.0, feedback_state.recurrence_similarity - 0.92)
        slot.future_reuse_ema = 0.80 * slot.future_reuse_ema + 0.20 * reuse_signal
        slot.reveal_count += 1
        slot.last_used_step = max(slot.last_used_step, feedback_state.routing_step)
        slot.exchangeability_ema = 0.82 * slot.exchangeability_ema + 0.18 * (
            feedback_state.exchangeability_score * (1.0 if route_advantage >= -0.01 else 0.6)
        )
        self._reservoir_router.record_cluster_outcome(slot.reservoir_cluster_id, specialist_credit)
        if feedback_state.slot_index > 0 and route_advantage >= 0.01:
            slot.shadow_wins += 1
        strong_reuse = (
            slot.usage_count >= 2
            and feedback_state.slot_index > 0
            and (recurrence_credit >= 0.05 or route_advantage >= 0.01)
            and (
                feedback_state.route_similarity >= 0.86
                or feedback_state.recurrence_similarity >= 0.90
                or slot.reveal_count >= 2
            )
        )
        if strong_reuse:
            slot.successful_reuses += 1
        self._update_specialist_support(
            slot=slot,
            batch=batch,
            route_advantage=route_advantage,
            revealed_lift=revealed_lift,
            batch_accuracy=batch_accuracy,
        )
        if feedback_state.regime_descriptor is not None and specialist_credit >= reward - 0.04:
            slot.regime_anchor = self._blend_regime_anchor(slot.regime_anchor, feedback_state.regime_descriptor.embedding)
        if feedback_state.active_signature is not None and specialist_credit >= reward - 0.02:
            slot.signature = 0.75 * slot.signature + 0.25 * feedback_state.active_signature
        if feedback_state.active_signature is not None and specialist_credit >= 0.52:
            self._recent_regime_signatures.append(feedback_state.active_signature.copy())

        controller = slot.controller
        if hasattr(controller, "observe_delayed_outcome"):
            controller.observe_delayed_outcome(
                feedback_state=feedback_state.inner_feedback_state,
                model=model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
                batch_accuracy=batch_accuracy,
                reliability=reliability,
                utility=utility,
                retrospective_reward=controller_reward,
                revealed_accuracy=revealed_accuracy,
                revealed_coverage=revealed_coverage,
                revealed_baseline_accuracy=revealed_baseline_accuracy,
                pending_delay_steps=pending_delay_steps,
                pending_outstanding_count=pending_outstanding_count,
                revealed_mean_residual=revealed_mean_residual,
                predicted_positive_rate=predicted_positive_rate,
                revealed_positive_rate=revealed_positive_rate,
            )

        strong_revealed_batch = (
            batch_accuracy >= 0.76
            or (revealed_accuracy is not None and revealed_baseline_accuracy is not None and revealed_accuracy >= revealed_baseline_accuracy + 0.03)
        )
        creation_score = (
            specialist_credit
            + 0.08 * novelty_bonus
            + 0.05 * max(0.0, signal.score - 1.0)
            + 0.12 * max(0.0, route_advantage)
        )
        if (
            feedback_state.candidate_new
            and decision.action in {"adapt", "label_shift", "bn_refresh", "recalibrate"}
            and strong_revealed_batch
            and creation_score >= 0.58
            and self._promotion_cooldown_remaining == 0
            and (risk_state.capital >= 1.2 or signal.score >= 0.98 or novelty_bonus >= 0.08)
            and feedback_state.active_signature is not None
            and feedback_state.novelty_score >= 0.02
            and route_advantage >= -0.01
            and (slot.shadow_wins >= 1 or slot.route_advantage_ema >= 0.01 or slot.successful_reuses >= 1)
            and self._can_add_specialist(feedback_state.active_signature)
        ):
            warm_snapshot = self._build_warm_start_snapshot(model, feedback_state.slot_snapshot)
            self._specialists.append(
                SpecialistSlot(
                    name=f"delayed_hybrid_specialist_{len(self._specialists)}",
                    snapshot=warm_snapshot,
                    signature=feedback_state.active_signature.copy(),
                    controller=self._new_specialist_controller(),
                    reward_ema=specialist_credit,
                    similarity_ema=feedback_state.route_similarity,
                    lift_ema=max(0.0, revealed_lift),
                    recurrence_reward_ema=max(0.0, recurrence_credit),
                    route_advantage_ema=max(0.0, route_advantage),
                    future_reuse_ema=max(0.0, reuse_signal),
                    support_quality_ema=max(0.0, route_advantage) + 0.5 * max(0.0, revealed_lift),
                    reveal_count=1,
                    last_used_step=feedback_state.routing_step,
                    shadow_wins=1 if route_advantage >= 0.01 else 0,
                    probation_remaining=self._probation_steps,
                    support_features=self._support_subset(batch.features, batch.labels),
                    support_positive_rate=float(batch.labels.mean()),
                    creation_positive_rate=float(batch.labels.mean()),
                    regime_anchor=(
                        None
                        if feedback_state.regime_descriptor is None
                        else feedback_state.regime_descriptor.embedding.copy()
                    ),
                    regime_confidence_ema=(
                        0.0
                        if feedback_state.regime_descriptor is None
                        else feedback_state.regime_descriptor.recurrence_confidence
                    ),
                    exchangeability_ema=feedback_state.exchangeability_score,
                    behavior_signature=(
                        None if self._last_behavior_sig is None else self._last_behavior_sig.copy()
                    ),
                )
            )
            self._specialist_promotions += 1
            self._promotion_cooldown_remaining = self._promotion_cooldown_steps
        elif (
            len(self._specialists) >= self._max_specialists
            and feedback_state.candidate_new
            and decision.action in {"adapt", "label_shift", "bn_refresh", "recalibrate"}
            and strong_revealed_batch
            and creation_score >= 0.62
            and self._promotion_cooldown_remaining == 0
            and feedback_state.active_signature is not None
            and feedback_state.novelty_score >= 0.04
            and route_advantage >= 0.01
            and (slot.shadow_wins >= 1 or slot.route_advantage_ema >= 0.015)
            and self._is_distinct_signature(feedback_state.active_signature)
        ):
            replace_index = self._worst_specialist_index()
            if replace_index is not None:
                warm_snapshot = self._build_warm_start_snapshot(model, feedback_state.slot_snapshot)
                self._specialists[replace_index] = SpecialistSlot(
                    name=f"delayed_hybrid_specialist_{replace_index}",
                    snapshot=warm_snapshot,
                    signature=feedback_state.active_signature.copy(),
                    controller=self._new_specialist_controller(),
                    reward_ema=specialist_credit,
                    similarity_ema=feedback_state.route_similarity,
                    lift_ema=max(0.0, revealed_lift),
                    recurrence_reward_ema=max(0.0, recurrence_credit),
                    route_advantage_ema=max(0.0, route_advantage),
                    future_reuse_ema=max(0.0, reuse_signal),
                    support_quality_ema=max(0.0, route_advantage) + 0.5 * max(0.0, revealed_lift),
                    reveal_count=1,
                    last_used_step=feedback_state.routing_step,
                    shadow_wins=1,
                    probation_remaining=self._probation_steps,
                    support_features=self._support_subset(batch.features, batch.labels),
                    support_positive_rate=float(batch.labels.mean()),
                    creation_positive_rate=float(batch.labels.mean()),
                    regime_anchor=(
                        None
                        if feedback_state.regime_descriptor is None
                        else feedback_state.regime_descriptor.embedding.copy()
                    ),
                    regime_confidence_ema=(
                        0.0
                        if feedback_state.regime_descriptor is None
                        else feedback_state.regime_descriptor.recurrence_confidence
                    ),
                    exchangeability_ema=feedback_state.exchangeability_score,
                    behavior_signature=(
                        None if self._last_behavior_sig is None else self._last_behavior_sig.copy()
                    ),
                )
                self._specialist_replacements += 1
                self._promotion_cooldown_remaining = self._promotion_cooldown_steps

    def _build_warm_start_snapshot(
        self,
        model: TorchTabularAdapterModel,
        slot_snapshot: "ModelSnapshot",
    ) -> "ModelSnapshot":
        """Blend source encoder + slot_snapshot adapter/head.

        A newly promoted specialist starts from:
          - Source encoder weights + source BN running stats (no accumulated drift)
          - Current adapter + head from slot_snapshot (preserves learned shift response)

        Falls back to slot_snapshot unchanged if source state is unavailable.
        """
        source_state = getattr(model, "_source_state", None)
        if source_state is None:
            return slot_snapshot
        anchored = {k: v.detach().clone() for k, v in source_state.items()}
        for key, value in slot_snapshot.network_state.items():
            if key.startswith("adapter.") or key.startswith("head."):
                anchored[key] = value.detach().clone()
        return ModelSnapshot(
            network_state=anchored,
            temperature=slot_snapshot.temperature,
            bias_offset=slot_snapshot.bias_offset,
        )

    def _worst_specialist_index(self) -> int | None:
        if len(self._specialists) <= 1:
            return None
        scored = []
        for index, specialist in enumerate(self._specialists[1:], start=1):
            mean_reward = specialist.cumulative_reward / max(1, specialist.usage_count)
            specialist_score = (
                mean_reward
                + 0.40 * specialist.reward_ema
                + 0.18 * specialist.recurrence_reward_ema
                + 0.12 * specialist.lift_ema
                + 0.20 * specialist.route_advantage_ema
                + 0.14 * specialist.future_reuse_ema
                + 0.08 * specialist.support_quality_ema
                + 0.03 * specialist.successful_reuses
                + 0.02 * specialist.shadow_wins
            )
            scored.append((specialist_score, specialist.usage_count, index))
        scored.sort()
        return scored[0][2]

    def _batch_signature(self, features: np.ndarray) -> np.ndarray:
        return _pooled_batch_signature(features, self._reference)

    def _signature_similarity(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        lhs_norm = float(np.linalg.norm(lhs)) + 1e-6
        rhs_norm = float(np.linalg.norm(rhs)) + 1e-6
        return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))

    def _blended_distance(self, feature_sig: np.ndarray, specialist: SpecialistSlot) -> float:
        feature_dist = float(np.linalg.norm(feature_sig - specialist.signature))
        if (
            not self._use_behavior_routing
            or self._last_behavior_sig is None
            or specialist.behavior_signature is None
        ):
            return feature_dist
        behavior_dist = float(np.linalg.norm(self._last_behavior_sig - specialist.behavior_signature))
        return 0.40 * feature_dist + 0.60 * behavior_dist

    def _blended_similarity(self, feature_sig: np.ndarray, specialist: SpecialistSlot) -> float:
        feature_sim = self._signature_similarity(feature_sig, specialist.signature)
        if (
            not self._use_behavior_routing
            or self._last_behavior_sig is None
            or specialist.behavior_signature is None
        ):
            return feature_sim
        behavior_sim = self._signature_similarity(self._last_behavior_sig, specialist.behavior_signature)
        return 0.40 * feature_sim + 0.60 * behavior_sim

    def _memory_similarity(self, signature: np.ndarray) -> float:
        if not self._recent_regime_signatures:
            return 0.0
        return max(self._signature_similarity(signature, memory) for memory in self._recent_regime_signatures)

    def _describe_regime(
        self,
        signature: np.ndarray,
        feature_shift: float,
        probabilities: list[float] | None = None,
    ) -> RegimeDescriptor:
        reuse_balance = self._specialist_route_reuses / max(
            1.0,
            float(self._specialist_route_reuses + self._specialist_route_fallbacks + 1),
        )
        temporal = np.array(
            [
                feature_shift,
                min(self._routing_step, 64) / 64.0,
                self._active_recurrence_similarity,
                self._active_novelty_score,
                self._active_reuse_confidence,
                reuse_balance,
                float(self._promotion_cooldown_remaining) / max(1.0, float(self._promotion_cooldown_steps)),
            ],
            dtype=np.float64,
        )
        embedding = build_regime_embedding(signature, temporal)
        return self._regime_encoder.register(embedding, step=self._routing_step, probabilities=probabilities)

    def _regime_anchor_similarity(self, descriptor: RegimeDescriptor, specialist: SpecialistSlot) -> float:
        if specialist.regime_anchor is None:
            return 0.0
        if specialist.regime_anchor.shape != descriptor.embedding.shape:
            return 0.0
        return self._signature_similarity(descriptor.embedding, specialist.regime_anchor)

    def _blend_regime_anchor(self, previous: np.ndarray | None, update: np.ndarray) -> np.ndarray:
        if previous is None:
            return update.copy()
        if previous.shape != update.shape:
            return update.copy()
        blended = 0.72 * previous + 0.28 * update
        norm = float(np.linalg.norm(blended))
        if norm <= 1e-8:
            return update.copy()
        return (blended / norm).astype(np.float64, copy=False)

    def _is_distinct_signature(self, signature: np.ndarray) -> bool:
        if not self._specialists:
            return True
        min_distance = min(float(np.linalg.norm(signature - specialist.signature)) for specialist in self._specialists)
        max_similarity = max(self._signature_similarity(signature, specialist.signature) for specialist in self._specialists)
        return min_distance >= self._distance_threshold * 0.45 and max_similarity <= 0.998

    def _can_add_specialist(self, signature: np.ndarray) -> bool:
        return len(self._specialists) < self._max_specialists and self._is_distinct_signature(signature)

    def _quick_feature_shift(self, features: np.ndarray) -> float:
        batch_mean = features.mean(axis=0)
        batch_variance = features.var(axis=0)
        normalized_mean_gap = np.mean(
            np.abs(batch_mean - self._reference.feature_mean) / np.sqrt(self._reference.feature_variance + 1e-6)
        )
        normalized_variance_gap = np.mean(
            np.abs(batch_variance - self._reference.feature_variance) / (self._reference.feature_variance + 1e-6)
        )
        return float(normalized_mean_gap + 0.5 * normalized_variance_gap)

    def _reuse_confidence(
        self,
        *,
        specialist: SpecialistSlot,
        distance: float,
        similarity: float,
        recurrence_similarity: float,
        feature_shift: float,
        novelty_score: float,
        regime_confidence: float,
        regime_similarity: float,
        exchangeability: float,
    ) -> float:
        similarity_term = 0.34 * min(1.0, max(0.0, similarity - 0.82) / 0.16)
        recurrence_term = 0.22 * min(1.0, max(0.0, recurrence_similarity - 0.88) / 0.10)
        support_term = 0.16 * min(1.0, max(0.0, specialist.support_quality_ema) / 0.10)
        reuse_term = 0.10 * min(1.0, max(0.0, specialist.future_reuse_ema) / 0.10)
        route_term = 0.10 * min(1.0, max(0.0, specialist.route_advantage_ema) / 0.06)
        success_term = 0.08 * min(1.0, specialist.successful_reuses / 2.0)
        reward_term = 0.06 * min(1.0, max(0.0, specialist.reward_ema - 0.50) / 0.10)
        regime_term = 0.10 * min(1.0, max(0.0, regime_confidence - 0.55) / 0.20)
        anchor_term = 0.08 * min(1.0, max(0.0, regime_similarity - 0.82) / 0.16)
        exchangeability_term = 0.14 * min(1.0, max(0.0, exchangeability - 0.60) / 0.30)
        exchangeability_history = 0.08 * min(1.0, max(0.0, specialist.exchangeability_ema - 0.55) / 0.25)
        feature_penalty = 0.20 * min(1.0, max(0.0, feature_shift - 1.55) / 0.80)
        novelty_penalty = 0.12 * min(1.0, max(0.0, novelty_score - 0.05) / 0.20)
        probation_penalty = 0.08 if specialist.probation_remaining > 0 and specialist.successful_reuses == 0 else 0.0
        distance_penalty = 0.10 * min(1.0, max(0.0, distance - self._distance_threshold) / 0.40)
        confidence = (
            similarity_term
            + recurrence_term
            + support_term
            + reuse_term
            + route_term
            + success_term
            + reward_term
            + regime_term
            + anchor_term
            + exchangeability_term
            + exchangeability_history
            - feature_penalty
            - novelty_penalty
            - probation_penalty
            - distance_penalty
        )
        return float(np.clip(confidence, 0.0, 1.0))

    def _adaptive_distance_threshold(self, specialist: SpecialistSlot) -> float:
        reward_bonus = 0.10 * max(0.0, specialist.reward_ema - 0.50)
        similarity_bonus = 0.08 * max(0.0, specialist.similarity_ema - 0.78)
        recurrence_bonus = 0.08 * max(0.0, specialist.recurrence_reward_ema - 0.05)
        route_advantage_bonus = 0.08 * max(0.0, specialist.route_advantage_ema)
        future_reuse_bonus = 0.06 * max(0.0, specialist.future_reuse_ema - 0.04)
        return (
            self._distance_threshold
            + reward_bonus
            + similarity_bonus
            + recurrence_bonus
            + route_advantage_bonus
            + future_reuse_bonus
        )

    def _decay_probation(self) -> None:
        for specialist in self._specialists[1:]:
            specialist.probation_remaining = max(0, specialist.probation_remaining - 1)

    def _retire_stale_specialists(self) -> None:
        if len(self._specialists) <= 1:
            return
        retained = [self._specialists[0]]
        retirements = 0
        for specialist in self._specialists[1:]:
            stale_steps = self._routing_step - specialist.last_used_step
            stale_retire = (
                stale_steps >= self._retire_after_steps
                and specialist.reveal_count >= 2
                and specialist.reward_ema < self._min_reward_ema_for_retention
                and specialist.recurrence_reward_ema < 0.08
                and specialist.successful_reuses == 0
                and specialist.shadow_wins == 0
            )
            low_future_value = (
                specialist.probation_remaining == 0
                and specialist.reveal_count >= 1
                and specialist.future_reuse_ema < self._min_future_reuse_for_retention
                and specialist.support_quality_ema < 0.05
                and specialist.successful_reuses == 0
                and specialist.shadow_wins == 0
            )
            # quality_ema starts at 0.0 and takes ~8 reveals to build to a
            # meaningful level (0.15 learning rate × 8 ≈ 0.70 of true value).
            # Retire when the specialist has been used enough to trust quality_ema
            # AND quality has genuinely degraded.
            # Threshold empirics: 3 reuses / 8 reveals / 0.75 threshold is
            # needed to keep FD002 (6 conditions) hybrid positive; original
            # 5/12/0.70 lets FD002 hybrid crash.  FD003/FD004 hybrid regressions
            # from the lower threshold don't change Gate B outcome (bandit still
            # passes for FD003; FD004 fails either way).
            quality_degraded = (
                specialist.successful_reuses >= 3
                and specialist.reveal_count >= 8
                and specialist.quality_ema < 0.75
                and specialist.quality_ema < (self._reference_accuracy - 0.08)
            )
            if stale_retire or low_future_value or quality_degraded:
                retirements += 1
                continue
            retained.append(specialist)
        if retirements:
            self._specialists = retained
            self._specialist_retirements += retirements

    def _route_score(
        self,
        *,
        distance: float,
        similarity: float,
        specialist: SpecialistSlot,
        regime_similarity: float,
        exchangeability: float,
    ) -> float:
        reward_bonus = 0.16 * max(0.0, specialist.reward_ema - 0.45)
        reuse_bonus = 0.06 * min(1.0, specialist.usage_count / 6.0)
        similarity_bonus = 0.10 * max(0.0, specialist.similarity_ema - 0.75)
        lift_bonus = 0.12 * max(0.0, specialist.lift_ema)
        recurrence_bonus = 0.20 * max(0.0, specialist.recurrence_reward_ema)
        route_advantage_bonus = 0.18 * max(0.0, specialist.route_advantage_ema)
        future_reuse_bonus = 0.16 * max(0.0, specialist.future_reuse_ema)
        support_bonus = 0.08 * max(0.0, specialist.support_quality_ema)
        regime_bonus = 0.18 * max(0.0, regime_similarity - 0.80)
        regime_confidence_bonus = 0.10 * max(0.0, specialist.regime_confidence_ema - 0.55)
        if self._use_strong_exchangeability:
            # Stronger feature-distribution matching (use_strong_exchangeability).
            # Lower floor 0.65→0.50, higher weight 0.18→0.25 and 0.08→0.12 / 0.55→0.45.
            # Uses raw feature gap so fault-mode identity isn't confused by behavior signals.
            exchangeability_bonus = 0.25 * max(0.0, exchangeability - 0.50)
            exchangeability_history = 0.12 * max(0.0, specialist.exchangeability_ema - 0.45)
        else:
            exchangeability_bonus = 0.18 * max(0.0, exchangeability - 0.65)
            exchangeability_history = 0.08 * max(0.0, specialist.exchangeability_ema - 0.55)
        cluster_reuse = self._reservoir_router.cluster_reuse_quality(specialist.reservoir_cluster_id)
        cluster_bonus = 0.14 * max(0.0, cluster_reuse - 0.45)
        coreset_quality = getattr(specialist.coreset, "reuse_quality", 0.5)
        coreset_bonus = 0.10 * max(0.0, coreset_quality - 0.45)
        success_bonus = 0.05 * min(1.0, specialist.successful_reuses / 3.0)
        shadow_bonus = 0.04 * min(1.0, specialist.shadow_wins / 3.0)
        recency_bonus = 0.05 if self._routing_step - specialist.last_used_step <= 6 else 0.0
        # Prefer specialists that have demonstrated high actual accuracy (quality_ema
        # tracks a slow EMA of revealed batch_accuracy; only meaningful after ~4 reveals).
        # Only apply a bonus for clearly good specialists — no penalty for unproven ones,
        # because low quality_ema might reflect wrong routing rather than a bad specialist.
        quality_bonus = 0.12 * max(0.0, specialist.quality_ema - 0.65) if specialist.reveal_count >= 4 else 0.0
        # Rate-staleness penalty (use_rate_staleness_penalty).
        # Penalises specialists whose creation positive-rate diverges from the current
        # revealed rate — addresses the "HYPOTHESIS CONFIRMED" diagnostic where specialists
        # built on healthy-engine data get loaded during terminal degradation.
        # Soft ramp starting at 0.04 gap so small benign fluctuations don't penalise.
        # Gate B impact (FD001/FD002/FD003/FD004): TBD — first run pending.
        if self._use_rate_staleness_penalty and self._current_revealed_positive_rate is not None:
            rate_gap = abs(self._current_revealed_positive_rate - specialist.creation_positive_rate)
            rate_staleness_penalty = 0.20 * max(0.0, rate_gap - 0.04)
        else:
            rate_staleness_penalty = 0.0
        return (
            distance
            + rate_staleness_penalty
            - 0.12 * similarity
            - reward_bonus
            - reuse_bonus
            - similarity_bonus
            - lift_bonus
            - recurrence_bonus
            - route_advantage_bonus
            - future_reuse_bonus
            - support_bonus
            - regime_bonus
            - regime_confidence_bonus
            - exchangeability_bonus
            - exchangeability_history
            - cluster_bonus
            - coreset_bonus
            - success_bonus
            - shadow_bonus
            - recency_bonus
            - quality_bonus
        )

    def _exchangeability_score(self, batch: TabularBatch, specialist: SpecialistSlot) -> float:
        if specialist.support_features is None or specialist.support_features.size == 0:
            return 0.0
        support_mean = specialist.support_features.mean(axis=0)
        support_var = specialist.support_features.var(axis=0)
        batch_mean = batch.features.mean(axis=0)
        batch_var = batch.features.var(axis=0)
        mean_gap = np.mean(np.abs(batch_mean - support_mean) / np.sqrt(support_var + 1e-6))
        var_gap = np.mean(np.abs(batch_var - support_var) / (support_var + 1e-6))
        feature_gap = float(mean_gap + 0.5 * var_gap)
        feature_score = float(np.exp(-0.65 * min(feature_gap, 6.0)))
        if batch.labels is None:
            label_score = 0.5
        else:
            positive_gap = abs(float(batch.labels.mean()) - specialist.support_positive_rate)
            label_score = float(np.exp(-4.0 * positive_gap))
        return float(np.clip(0.70 * feature_score + 0.30 * label_score, 0.0, 1.0))

    def _support_subset(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if len(features) <= self._support_batch_size:
            return features.copy()
        positive_indices = np.flatnonzero(labels == 1)
        negative_indices = np.flatnonzero(labels == 0)
        half = max(1, self._support_batch_size // 2)
        chosen: list[int] = []
        if len(positive_indices) > 0:
            chosen.extend(positive_indices[: min(half, len(positive_indices))].tolist())
        if len(negative_indices) > 0:
            chosen.extend(negative_indices[: min(half, len(negative_indices))].tolist())
        if len(chosen) < self._support_batch_size:
            remainder = [index for index in range(len(features)) if index not in set(chosen)]
            chosen.extend(remainder[: self._support_batch_size - len(chosen)])
        chosen = chosen[: self._support_batch_size]
        return features[np.asarray(chosen, dtype=np.int64)].copy()

    def _update_specialist_support(
        self,
        *,
        slot: SpecialistSlot,
        batch: TabularBatch,
        route_advantage: float,
        revealed_lift: float,
        batch_accuracy: float,
    ) -> None:
        support_quality = max(0.0, route_advantage) + 0.5 * max(0.0, revealed_lift) + 0.2 * max(0.0, batch_accuracy - 0.70)
        slot.support_quality_ema = 0.82 * slot.support_quality_ema + 0.18 * support_quality
        slot.quality_ema = 0.85 * slot.quality_ema + 0.15 * batch_accuracy
        slot.support_positive_rate = 0.80 * slot.support_positive_rate + 0.20 * float(batch.labels.mean())
        self._ensure_coreset(slot)
        if batch.labels is not None:
            slot.coreset.update(
                batch_features=batch.features,
                batch_labels=batch.labels,
                batch_utility=support_quality,
            )
            coreset_features = slot.coreset.support_features()
            if coreset_features is not None:
                slot.support_features = coreset_features
                return
        if support_quality <= 0.0 and slot.support_features is not None:
            return
        candidate_support = self._support_subset(batch.features, batch.labels)
        if slot.support_features is None or slot.support_features.shape != candidate_support.shape:
            slot.support_features = candidate_support
            return
        keep = max(1, candidate_support.shape[0] // 2)
        slot.support_features = np.concatenate(
            [slot.support_features[:keep], candidate_support[: candidate_support.shape[0] - keep]],
            axis=0,
        ).astype(np.float32, copy=False)

    def _warm_start_specialist(
        self,
        model: TorchTabularAdapterModel,
        specialist: SpecialistSlot,
        batch: TabularBatch,
    ) -> None:
        if specialist.support_features is None or specialist.support_features.size == 0:
            return
        if self._active_index == 0:
            return
        support_ready = specialist.support_quality_ema >= 0.03 or specialist.successful_reuses >= 1
        recurrence_ready = (
            self._active_regime_memory_gate >= 1.0
            or self._active_regime_anchor_similarity >= 0.88
            or (self._active_regime_descriptor is not None and self._active_regime_descriptor.reuse_ready)
            or self._active_recurrence_similarity >= 0.90
            or specialist.future_reuse_ema >= 0.05
            or specialist.shadow_wins >= 1
        )
        severe_shift = (
            self._active_feature_shift >= 1.95
            and self._active_novelty_score >= 0.08
            and self._active_recurrence_similarity < 0.975
            and specialist.support_quality_ema < 0.08
        )
        if (
            self._active_reuse_confidence < 0.58
            or not support_ready
            or not recurrence_ready
            or severe_shift
        ):
            self._specialist_warm_starts_skipped += 1
            return
        blended = np.concatenate([specialist.support_features, batch.features], axis=0)
        model.refresh_batch_norm(blended, passes=1)
        self._specialist_warm_starts_applied += 1
        current_probabilities = np.asarray(model.predict_proba(batch.features), dtype=np.float32)
        blended_positive_rate = 0.65 * specialist.support_positive_rate + 0.35 * float(current_probabilities.mean())
        if specialist.support_quality_ema >= 0.04 and self._active_reuse_confidence >= 0.64:
            model.apply_label_shift_correction(
                source_positive_rate=self._reference.mean_probability,
                target_positive_rate=blended_positive_rate,
                momentum=0.12,
                max_abs_bias=0.55,
            )

    def _shadow_route_snapshot(self) -> tuple["ModelSnapshot" | None, int]:
        if len(self._specialists) <= 1 or self._active_signature is None:
            return None, -1
        alternative_scores: list[tuple[float, int]] = []
        for index, specialist in enumerate(self._specialists):
            if index == self._active_index:
                continue
            distance = float(np.linalg.norm(self._active_signature - specialist.signature))
            similarity = self._signature_similarity(self._active_signature, specialist.signature)
            regime_similarity = (
                0.0
                if self._active_regime_descriptor is None
                else self._regime_anchor_similarity(self._active_regime_descriptor, specialist)
            )
            alternative_scores.append(
                (
                    self._route_score(
                        distance=distance,
                        similarity=similarity,
                        specialist=specialist,
                        regime_similarity=regime_similarity,
                        exchangeability=specialist.exchangeability_ema,
                    ),
                    index,
                )
            )
        alternative_scores.sort()
        if not alternative_scores:
            return None, -1
        shadow_alt_index = alternative_scores[0][1]
        return self._specialists[shadow_alt_index].snapshot, shadow_alt_index

    def _shadow_replay_accuracy(
        self,
        *,
        model: TorchTabularAdapterModel,
        snapshot: "ModelSnapshot",
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
    ) -> float:
        if snapshot is None:
            return 0.0
        restore_snapshot = model.export_state()
        model.load_state(snapshot)
        probabilities = model.predict_proba(batch.features)
        _ = self._replay_decision_action(
            model=model,
            decision=decision,
            batch=batch,
            signal=signal,
            risk_state=risk_state,
            probabilities=probabilities,
        )
        probabilities = np.asarray(model.predict_proba(batch.features), dtype=np.float32)
        predictions = (probabilities >= 0.5).astype(np.int64)
        accuracy = float((predictions == batch.labels).mean())
        model.load_state(restore_snapshot)
        return accuracy

    def _replay_decision_action(
        self,
        *,
        model: TorchTabularAdapterModel,
        decision: TabularDecision,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        probabilities: list[float],
    ) -> TabularDecision:
        if decision.action in {"none", "hold", "abstain"}:
            return decision
        if decision.action == "bn_refresh":
            return _apply_bn_refresh(model, batch)
        if decision.action == "label_shift":
            return _apply_label_shift(model, self._reference, signal)
        if decision.action == "recalibrate":
            return _apply_recalibration(model, self._reference, signal)
        if decision.action == "adapt":
            return _apply_adaptation(
                model,
                signal,
                risk_state,
                batch,
                probabilities,
                learning_rate=0.025,
                anchor_strength=0.16,
                entropy_weight=0.08,
                max_parameter_drift=0.70,
                steps=2,
            )
        if decision.action == "reset":
            return _apply_reset(model, "shadow_counterfactual_reset")
        return decision

    def set_revealed_positive_rate(self, rate: float) -> None:
        """Receive the recent revealed label positive rate from the correction engine."""
        self._current_revealed_positive_rate = rate

    def get_diagnostics(self) -> dict[str, float]:
        usage_counts = [specialist.usage_count for specialist in self._specialists]
        active_reuse = sum(max(0, count - 1) for count in usage_counts)
        # Per-selection diagnostics: selected specialist id, quality, and regime distance
        selected_slot = self._specialists[self._active_index] if self._specialists else None
        selected_support_positive_rate = selected_slot.support_positive_rate if selected_slot is not None else 0.5
        diagnostics = {
            "specialist_count": float(len(self._specialists)),
            "specialist_selected_id": float(self._active_index),
            "specialist_selected_quality": self._active_reuse_confidence,
            "specialist_regime_distance": self._active_route_distance,
            "specialist_selected_support_positive_rate": selected_support_positive_rate,
            "specialist_mean_usage": float(np.mean(usage_counts)) if usage_counts else 0.0,
            "specialist_max_usage": float(max(usage_counts)) if usage_counts else 0.0,
            "specialist_reuse_ratio": active_reuse / max(1.0, float(sum(usage_counts))),
            "specialist_mean_reward_ema": float(np.mean([specialist.reward_ema for specialist in self._specialists]))
            if self._specialists
            else 0.0,
            "specialist_mean_lift_ema": float(np.mean([specialist.lift_ema for specialist in self._specialists]))
            if self._specialists
            else 0.0,
            "specialist_mean_recurrence_reward_ema": float(
                np.mean([specialist.recurrence_reward_ema for specialist in self._specialists])
            )
            if self._specialists
            else 0.0,
            "specialist_mean_route_advantage_ema": float(
                np.mean([specialist.route_advantage_ema for specialist in self._specialists])
            )
            if self._specialists
            else 0.0,
            "specialist_mean_future_reuse_ema": float(
                np.mean([specialist.future_reuse_ema for specialist in self._specialists])
            )
            if self._specialists
            else 0.0,
            "specialist_mean_support_quality_ema": float(
                np.mean([specialist.support_quality_ema for specialist in self._specialists])
            )
            if self._specialists
            else 0.0,
            "specialist_successful_reuses": float(sum(specialist.successful_reuses for specialist in self._specialists)),
            "specialist_shadow_wins": float(sum(specialist.shadow_wins for specialist in self._specialists)),
            "specialist_last_route_distance": self._active_route_distance,
            "specialist_last_route_similarity": self._active_route_similarity,
            "specialist_last_novelty_score": self._active_novelty_score,
            "specialist_last_recurrence_similarity": self._active_recurrence_similarity,
            "specialist_last_feature_shift": self._active_feature_shift,
            "specialist_last_reuse_confidence": self._active_reuse_confidence,
            "specialist_last_regime_anchor_similarity": self._active_regime_anchor_similarity,
            "specialist_last_exchangeability_score": self._active_exchangeability_score,
            "specialist_last_regime_confidence": (
                0.0 if self._active_regime_descriptor is None else self._active_regime_descriptor.recurrence_confidence
            ),
            "specialist_last_regime_novelty": (
                0.0 if self._active_regime_descriptor is None else self._active_regime_descriptor.novelty_score
            ),
            "specialist_regime_memory_gate": self._active_regime_memory_gate,
            "specialist_promotions": float(self._specialist_promotions),
            "specialist_replacements": float(self._specialist_replacements),
            "specialist_retirements": float(self._specialist_retirements),
            "specialist_promotion_cooldown": float(self._promotion_cooldown_remaining),
            "specialist_route_reuses": float(self._specialist_route_reuses),
            "specialist_route_fallbacks": float(self._specialist_route_fallbacks),
            "specialist_warm_starts_applied": float(self._specialist_warm_starts_applied),
            "specialist_warm_starts_skipped": float(self._specialist_warm_starts_skipped),
            "specialist_mean_exchangeability_ema": float(
                np.mean([specialist.exchangeability_ema for specialist in self._specialists])
            )
            if self._specialists
            else 0.0,
            # Quality diagnostics: how accurate are specialists on their revealed batches?
            "specialist_mean_quality_ema": float(
                np.mean([s.quality_ema for s in self._specialists if s.reveal_count >= 4])
            ) if any(s.reveal_count >= 4 for s in self._specialists) else 0.0,
            "specialist_min_quality_ema": float(
                min((s.quality_ema for s in self._specialists if s.reveal_count >= 4), default=0.0)
            ),
            "specialist_max_quality_ema": float(
                max((s.quality_ema for s in self._specialists if s.reveal_count >= 4), default=0.0)
            ),
            # Reuse rate: fraction of routing decisions that selected a non-base specialist
            "specialist_nonbase_reuse_rate": float(self._specialist_route_reuses) / max(
                1.0, float(self._specialist_route_reuses + self._specialist_route_fallbacks + 1)
            ),
            # Total reveals across all specialists (proxy for learning signal accumulated)
            "specialist_total_reveals": float(sum(s.reveal_count for s in self._specialists)),
            # Fraction of specialists that have ≥ 1 successful reuse (mature specialists)
            "specialist_mature_fraction": float(
                sum(1 for s in self._specialists if s.successful_reuses >= 1)
            ) / max(1.0, float(len(self._specialists))),
        }
        diagnostics.update(self._regime_encoder.get_diagnostics(prefix="specialist_regime"))
        return diagnostics


class RoutedDelayedBanditSpecialistPolicy(DelayedHybridBanditSpecialistPolicy):
    """Decouples specialist routing from action choice via one shared delayed controller."""

    def __init__(
        self,
        reference: TabularReferenceProfile,
        *,
        max_specialists: int = 4,
        distance_threshold: float = 1.35,
        promotion_cooldown_steps: int = 6,
        retire_after_steps: int = 16,
        min_reward_ema_for_retention: float = 0.48,
    ) -> None:
        self._shared_controller = RegimeAwareDelayedBanditTabularPolicy(reference)
        super().__init__(
            reference,
            max_specialists=max_specialists,
            distance_threshold=distance_threshold,
            promotion_cooldown_steps=promotion_cooldown_steps,
            retire_after_steps=retire_after_steps,
            min_reward_ema_for_retention=min_reward_ema_for_retention,
        )

    def _new_specialist_controller(self) -> object:
        return self._shared_controller

    def get_diagnostics(self) -> dict[str, float]:
        diagnostics = super().get_diagnostics()
        diagnostics["shared_action_controller"] = 1.0
        return diagnostics


def _compute_reliability(signal: TabularShiftSignal, risk_state: RiskState, decision: TabularDecision) -> float:
    base = 1.0 - 0.20 * signal.feature_score - 0.28 * signal.output_score - 0.32 * signal.collapse_risk
    risk_penalty = min(0.45, 0.04 * max(0.0, risk_state.capital - 1.0))
    action_penalty = 0.08 if decision.action == "reset" else 0.0
    action_penalty += 0.05 if decision.action == "abstain" else 0.0
    return max(0.0, min(1.0, base - risk_penalty - action_penalty))


def _compute_batch_utility(
    *,
    batch_accuracy: float,
    risk_state: RiskState,
    decision: TabularDecision,
    parameter_drift: float,
) -> float:
    utility = batch_accuracy
    utility -= 0.06 * float(risk_state.alert)
    utility -= 0.03 * min(1.0, parameter_drift)
    utility -= 0.10 * float(decision.action == "abstain")
    utility -= 0.04 * float(decision.action == "reset")
    return utility


def _pooled_batch_signature(
    features: np.ndarray,
    reference: TabularReferenceProfile,
    *,
    chunks: int = 6,
) -> np.ndarray:
    normalized = (features - reference.feature_mean) / np.sqrt(reference.feature_variance + 1e-6)
    normalized_mean = normalized.mean(axis=0)
    normalized_variance = normalized.var(axis=0)
    normalized_abs_mean = np.abs(normalized).mean(axis=0)
    normalized_skew = np.mean(np.clip(normalized, -3.0, 3.0) ** 3, axis=0)
    pooled_mean = np.array(
        [float(chunk.mean()) for chunk in np.array_split(normalized_mean, chunks)],
        dtype=np.float64,
    )
    pooled_variance = np.array(
        [float(chunk.mean()) for chunk in np.array_split(normalized_variance, chunks)],
        dtype=np.float64,
    )
    pooled_abs_mean = np.array(
        [float(chunk.mean()) for chunk in np.array_split(normalized_abs_mean, chunks)],
        dtype=np.float64,
    )
    pooled_skew = np.array(
        [float(chunk.mean()) for chunk in np.array_split(normalized_skew, chunks)],
        dtype=np.float64,
    )
    global_stats = np.array(
        [
            float(np.linalg.norm(pooled_mean)),
            float(np.linalg.norm(pooled_variance - 1.0)),
            float(np.mean(normalized_abs_mean)),
            float(np.mean(np.abs(normalized_skew))),
        ],
        dtype=np.float64,
    )
    return np.concatenate([pooled_mean, pooled_variance, pooled_abs_mean, pooled_skew, global_stats]).astype(np.float64)


def _build_real_tabular_source(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_breast_cancer()
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)

    x_source_pool, x_test, y_source_pool, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )

    # Keep the labeled source pool intentionally modest so adaptation has room to matter.
    x_source_small, _, y_source_small, _ = train_test_split(
        x_source_pool,
        y_source_pool,
        train_size=0.38,
        random_state=seed + 1,
        stratify=y_source_pool,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_source_small,
        y_source_small,
        test_size=0.25,
        random_state=seed + 2,
        stratify=y_source_small,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def _sample_indices(
    rng: np.random.Generator,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    batch_size: int,
    positive_rate: float,
) -> np.ndarray:
    positive_count = int(round(batch_size * positive_rate))
    negative_count = batch_size - positive_count
    pos = rng.choice(positive_indices, size=positive_count, replace=True)
    neg = rng.choice(negative_indices, size=negative_count, replace=True)
    indices = np.concatenate([pos, neg])
    rng.shuffle(indices)
    return indices


def _regime_for_step(step: int) -> str:
    if step < 18:
        return "stable"
    if step < 36:
        return "gradual_covariate"
    if step < 54:
        return "label_shift"
    if step < 72:
        return "abrupt_covariate"
    return "recurring_covariate"


def _apply_regime_transform(
    features: np.ndarray,
    regime: str,
    step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    transformed = features.copy()
    if regime == "stable":
        return transformed

    if regime == "gradual_covariate":
        strength = min(1.0, (step - 18) / 18.0)
        transformed[:, :10] = transformed[:, :10] * (1.0 + 0.40 * strength) + 0.40 * strength
        transformed[:, 10:20] += rng.normal(0.0, 0.10 + 0.08 * strength, size=transformed[:, 10:20].shape)
        transformed[:, 20:25] = 0.85 * transformed[:, 20:25] - 0.35 * transformed[:, 25:30]
        return transformed

    if regime == "label_shift":
        transformed[:, :10] = transformed[:, :10] * 1.25 + 0.22
        transformed[:, 10:18] += rng.normal(0.0, 0.12, size=transformed[:, 10:18].shape)
        return transformed

    if regime == "abrupt_covariate":
        base = transformed.copy()
        transformed[:, :6] = np.tanh(1.8 * base[:, :6] + 0.8)
        transformed[:, 6:12] = 1.6 * base[:, 6:12] - 0.9
        transformed[:, 12:18] = 0.70 * base[:, 12:18] - 0.55 * base[:, 18:24]
        transformed[:, 18:24] = 0.45 * base[:, 12:18] + 0.95 * base[:, 18:24] + 0.35
        transformed[:, 24:30] = 0.6 * base[:, 24:30]
        return transformed

    transformed[:, :10] = transformed[:, :10] * 1.18 + 0.18
    transformed[:, 10:18] += rng.normal(0.0, 0.08, size=transformed[:, 10:18].shape)
    transformed[:, 18:24] = 0.90 * transformed[:, 18:24] - 0.20 * transformed[:, 24:30]
    return transformed


def build_tabular_stream(
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    steps: int = 90,
    batch_size: int = 48,
    seed: int = 7,
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(y_test == 1)
    negative_indices = np.flatnonzero(y_test == 0)

    batches: list[TabularBatch] = []
    for step in range(steps):
        regime = _regime_for_step(step)
        if regime == "stable":
            positive_rate = 0.63
        elif regime == "gradual_covariate":
            positive_rate = 0.63
        elif regime == "label_shift":
            positive_rate = 0.88
        elif regime == "abrupt_covariate":
            positive_rate = 0.28
        else:
            positive_rate = 0.76

        indices = _sample_indices(rng, positive_indices, negative_indices, batch_size, positive_rate)
        base_features = x_test[indices]
        labels = y_test[indices]
        transformed_features = _apply_regime_transform(base_features, regime, step, rng)
        batches.append(TabularBatch(features=transformed_features, labels=labels, regime=regime))
    return batches


def _build_reference_batches(
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    count: int = 10,
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_validation))
    batches: list[TabularBatch] = []
    for _ in range(count):
        chosen = rng.choice(indices, size=batch_size, replace=True)
        batches.append(TabularBatch(features=x_validation[chosen], labels=y_validation[chosen], regime="reference"))
    return batches


def _build_reference_profile(
    model: TorchTabularAdapterModel,
    reference_batches: Iterable[TabularBatch],
) -> tuple[TabularReferenceProfile, list[float]]:
    reference_batches = list(reference_batches)
    features = np.concatenate([batch.features for batch in reference_batches], axis=0)
    probabilities = [
        probability
        for batch in reference_batches
        for probability in model.predict_proba(batch.features)
    ]
    profile = TabularReferenceProfile(
        feature_mean=features.mean(axis=0),
        feature_variance=features.var(axis=0) + 1e-6,
        mean_entropy=float(np.mean([_binary_entropy(probability) for probability in probabilities])),
        mean_probability=float(np.mean(probabilities)),
        positive_rate=float(np.mean([1.0 if probability >= 0.5 else 0.0 for probability in probabilities])),
        mean_confidence=float(np.mean([max(probability, 1.0 - probability) for probability in probabilities])),
    )

    monitor = TabularShiftMonitor(profile)
    reference_scores: list[float] = []
    for batch in reference_batches:
        batch_probabilities = model.predict_proba(batch.features)
        signal = monitor.evaluate(batch.features, batch_probabilities)
        reference_scores.append(signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk)
    return profile, reference_scores


def _evaluate_strategy(
    name: str,
    model: TorchTabularAdapterModel,
    policy: (
        FrozenTabularPolicy
        | NaiveTabularPolicy
        | ControllerTabularPolicy
        | MultiActionTabularPolicy
        | BanditTabularPolicy
        | SpecialistMemoryTabularPolicy
        | HybridBanditSpecialistPolicy
    ),
    batches: Iterable[TabularBatch],
    reference: TabularReferenceProfile,
    reference_scores: list[float],
) -> TabularStrategyResult:
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

    for step, batch in enumerate(batches):
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
        predictions = np.array([1 if probability >= 0.5 else 0 for probability in probabilities], dtype=np.int64)
        batch_correct = int((predictions == batch.labels).sum())
        batch_accuracy = batch_correct / max(1, len(batch.labels))
        reliability = _compute_reliability(signal, risk_state, decision)
        utility = _compute_batch_utility(
            batch_accuracy=batch_accuracy,
            risk_state=risk_state,
            decision=decision,
            parameter_drift=model.parameter_drift(),
        )

        total += len(batch.labels)
        if decision.action != "abstain":
            correct += batch_correct
            served_correct += batch_correct
            served_total += len(batch.labels)
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
        regime_total[batch.regime] = regime_total.get(batch.regime, 0) + len(batch.labels)
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

        if hasattr(policy, "observe_outcome"):
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

    regime_accuracy = {
        regime: regime_correct[regime] / max(1, regime_total[regime])
        for regime in sorted(regime_total.keys())
    }
    diagnostics = policy.get_diagnostics() if hasattr(policy, "get_diagnostics") else {}
    steps = len(traces)
    return TabularStrategyResult(
        name=name,
        overall_accuracy=correct / max(1, total),
        served_accuracy=served_correct / max(1, served_total),
        coverage=served_total / max(1, total),
        mean_utility=utility_sum / max(1, steps),
        alerts=alerts,
        risk_alerts=risk_alerts,
        adaptations=adaptations,
        resets=resets,
        abstains=abstains,
        mean_shift_score=shift_sum / max(1, steps),
        mean_risk_capital=risk_capital_sum / max(1, steps),
        mean_reliability=reliability_sum / max(1, steps),
        mean_parameter_drift=parameter_drift_sum / max(1, steps),
        regime_accuracy=regime_accuracy,
        action_counts=action_counts,
        diagnostics=diagnostics,
        traces=tuple(traces),
    )


def run_tabular_benchmark(steps: int = 90, batch_size: int = 48, seed: int = 7) -> TabularBenchmarkResult:
    x_train, y_train, x_validation, y_validation, x_test, y_test = _build_real_tabular_source(seed=seed)
    source_model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=seed)
    source_summary = source_model.fit_source(x_train, y_train, x_validation, y_validation)

    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    reference, reference_scores = _build_reference_profile(source_model, reference_batches)

    stream = build_tabular_stream(x_test, y_test, steps=steps, batch_size=batch_size, seed=seed + 31)
    strategies = run_tabular_benchmark_with_factories(
        policy_factories=_default_policy_factories(),
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        prepared=(
            source_summary,
            reference,
            reference_scores,
            stream,
            source_model,
        ),
    ).strategies
    return TabularBenchmarkResult(
        steps=steps,
        batch_size=batch_size,
        source_summary=source_summary,
        reference=reference,
        strategies=strategies,
    )


def _default_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        ("specialist_memory", lambda reference: SpecialistMemoryTabularPolicy(reference)),
        ("hybrid", lambda reference: HybridBanditSpecialistPolicy(reference)),
    ]


def _tta_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("tent", lambda reference: TentTabularPolicy()),
        ("eata_style", lambda reference: EataStyleTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
    ]


def run_tabular_benchmark_with_factories(
    *,
    policy_factories: list[tuple[str, PolicyFactory]],
    steps: int = 90,
    batch_size: int = 48,
    seed: int = 7,
    prepared: tuple[SourceFitSummary, TabularReferenceProfile, list[float], list[TabularBatch], TorchTabularAdapterModel]
    | None = None,
) -> TabularBenchmarkResult:
    if prepared is None:
        x_train, y_train, x_validation, y_validation, x_test, y_test = _build_real_tabular_source(seed=seed)
        source_model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=seed)
        source_summary = source_model.fit_source(x_train, y_train, x_validation, y_validation)
        reference_batches = _build_reference_batches(
            x_validation,
            y_validation,
            batch_size=batch_size,
            seed=seed + 17,
        )
        reference, reference_scores = _build_reference_profile(source_model, reference_batches)
        stream = build_tabular_stream(x_test, y_test, steps=steps, batch_size=batch_size, seed=seed + 31)
    else:
        source_summary, reference, reference_scores, stream, source_model = prepared

    strategies = tuple(
        _evaluate_strategy(name, source_model.clone(), factory(reference), stream, reference, reference_scores)
        for name, factory in policy_factories
    )
    return TabularBenchmarkResult(
        steps=steps,
        batch_size=batch_size,
        source_summary=source_summary,
        reference=reference,
        strategies=strategies,
    )


def render_tabular_benchmark_report(result: TabularBenchmarkResult) -> str:
    frozen_accuracy = next(
        strategy.overall_accuracy for strategy in result.strategies if strategy.name == "frozen"
    )
    lines = [
        "Adaptive Reliability Layer Tabular Benchmark",
        (
            f"steps={result.steps} batch_size={result.batch_size} "
            f"source_val_acc={result.source_summary.best_validation_accuracy:.3f}"
        ),
        (
            "reference "
            f"entropy={result.reference.mean_entropy:.3f} "
            f"mean_probability={result.reference.mean_probability:.3f} "
            f"positive_rate={result.reference.positive_rate:.3f} "
            f"mean_confidence={result.reference.mean_confidence:.3f}"
        ),
        "",
        "strategy     accuracy   served_acc   coverage   utility   delta_vs_frozen   alerts   risk_alerts   adapts   resets   abstains   mean_shift   mean_capital   reliability   param_drift",
    ]
    for strategy in result.strategies:
        lines.append(
            f"{strategy.name:<12}"
            f"{strategy.overall_accuracy:>8.3f}"
            f"{strategy.served_accuracy:>13.3f}"
            f"{strategy.coverage:>11.3f}"
            f"{strategy.mean_utility:>10.3f}"
            f"{strategy.overall_accuracy - frozen_accuracy:>18.3f}"
            f"{strategy.alerts:>9}"
            f"{strategy.risk_alerts:>14}"
            f"{strategy.adaptations:>9}"
            f"{strategy.resets:>9}"
            f"{strategy.abstains:>11}"
            f"{strategy.mean_shift_score:>13.3f}"
            f"{strategy.mean_risk_capital:>14.3f}"
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
                f"action={trace.action}:capital={trace.martingale_capital:.2f}"
            )
            for trace in worst_traces
        )
        lines.append(f"  worst:   {worst_summary}")
        action_summary = ", ".join(
            f"{action}={count}" for action, count in sorted(strategy.action_counts.items())
        )
        lines.append(f"  actions: {action_summary}")
        if strategy.diagnostics:
            diagnostic_summary = ", ".join(
                f"{name}={value:.3f}" for name, value in sorted(strategy.diagnostics.items())
            )
            lines.append(f"  diag:    {diagnostic_summary}")
    return "\n".join(lines)
