from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model import OnlineLogisticModel
from .monitoring import ShiftSignal


@dataclass(frozen=True)
class AdaptationDecision:
    action: str
    reason: str
    selected_fraction: float = 0.0


class AdaptationPolicy(Protocol):
    def apply(
        self,
        model: OnlineLogisticModel,
        signal: ShiftSignal,
        features: list[float],
        probabilities: list[float],
    ) -> AdaptationDecision:
        ...


class FrozenPolicy:
    """Baseline that never adapts."""

    def apply(
        self,
        model: OnlineLogisticModel,
        signal: ShiftSignal,
        features: list[float],
        probabilities: list[float],
    ) -> AdaptationDecision:
        del model, signal, features, probabilities
        return AdaptationDecision(action="none", reason="frozen_baseline")


class NaiveAdaptationPolicy:
    """Baseline that adapts immediately whenever drift appears meaningful."""

    def __init__(
        self,
        trigger_threshold: float = 1.0,
        step_size: float = 0.35,
        confidence_threshold: float = 0.55,
        max_parameter_drift: float = 3.0,
    ) -> None:
        self._trigger_threshold = trigger_threshold
        self._step_size = step_size
        self._confidence_threshold = confidence_threshold
        self._max_parameter_drift = max_parameter_drift

    def apply(
        self,
        model: OnlineLogisticModel,
        signal: ShiftSignal,
        features: list[float],
        probabilities: list[float],
    ) -> AdaptationDecision:
        if signal.score < self._trigger_threshold:
            return AdaptationDecision(action="none", reason="shift_below_threshold")

        selected_fraction = model.adapt(
            features,
            probabilities,
            step_size=self._step_size,
            confidence_threshold=self._confidence_threshold,
            anchor_strength=0.0,
            max_parameter_drift=self._max_parameter_drift,
        )
        if selected_fraction == 0.0:
            return AdaptationDecision(action="hold", reason="no_confident_samples")
        return AdaptationDecision(
            action="adapt",
            reason="naive_shift_trigger",
            selected_fraction=selected_fraction,
        )


class AdaptationController:
    """Simple safety-gated controller for early experiments."""

    def __init__(
        self,
        mild_threshold: float = 1.0,
        severe_threshold: float = 1.8,
        step_size: float = 0.12,
        cooldown_steps: int = 2,
        anchor_strength: float = 0.08,
        max_parameter_drift: float = 0.8,
    ) -> None:
        self._mild_threshold = mild_threshold
        self._severe_threshold = severe_threshold
        self._step_size = step_size
        self._cooldown_steps = cooldown_steps
        self._anchor_strength = anchor_strength
        self._max_parameter_drift = max_parameter_drift
        self._cooldown_remaining = 0
        self._consecutive_severe = 0

    def apply(
        self,
        model: OnlineLogisticModel,
        signal: ShiftSignal,
        features: list[float],
        probabilities: list[float],
    ) -> AdaptationDecision:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return AdaptationDecision(action="hold", reason="cooldown")

        if signal.score < self._mild_threshold:
            self._consecutive_severe = 0
            return AdaptationDecision(action="none", reason="shift_below_threshold")

        if signal.severe or signal.score >= self._severe_threshold:
            self._consecutive_severe += 1
        else:
            self._consecutive_severe = 0

        if signal.collapse_risk >= 0.55 or self._consecutive_severe >= 3:
            model.reset()
            self._consecutive_severe = 0
            self._cooldown_remaining = self._cooldown_steps
            return AdaptationDecision(action="reset", reason="persistent_or_collapse_shift")

        confidence_threshold = 0.85 if signal.severe else 0.70
        selected_fraction = model.adapt(
            features,
            probabilities,
            step_size=self._step_size,
            confidence_threshold=confidence_threshold,
            anchor_strength=self._anchor_strength,
            max_parameter_drift=self._max_parameter_drift,
        )
        if selected_fraction == 0.0:
            self._cooldown_remaining = 1
            return AdaptationDecision(action="hold", reason="no_confident_samples")

        self._cooldown_remaining = self._cooldown_steps
        return AdaptationDecision(
            action="adapt",
            reason="shift_detected",
            selected_fraction=selected_fraction,
        )
