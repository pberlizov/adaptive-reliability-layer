from __future__ import annotations

import inspect
from collections import deque
from typing import Any

import numpy as np

from ..tabular_benchmark import (
    TabularBatch,
    TabularDecision,
    TabularShiftSignal,
)
from ..risk import RiskState
from .config import RuntimeConfig
from .feedback import PendingDelayedOutcome, apply_policy_feedback
from .types import InterventionDecision


class DelayedCorrectionEngine:
    """Manages the delayed-label feedback loop: enqueue → reveal → credit policy."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._pending: deque[PendingDelayedOutcome] = deque()
        self._revealed_metrics: list[dict[str, float]] = []
        self._revealed_positive_rates: deque[float] = deque(maxlen=20)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def revealed_metrics(self) -> tuple[dict[str, float], ...]:
        return tuple(self._revealed_metrics)

    @property
    def recent_revealed_positive_rate(self) -> float | None:
        """Mean positive rate across the last 5 revealed batches, or None if fewer than 2."""
        if len(self._revealed_positive_rates) < 2:
            return None
        recent = list(self._revealed_positive_rates)[-5:]
        return float(np.mean(recent))

    def enqueue(
        self,
        *,
        step: int,
        batch_id: str | None,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
        recommended: InterventionDecision,
        action_taken: str,
        predictions: list[int],
        probabilities: list[float],
        parameter_drift: float,
        abstained: bool,
        policy: Any,
        policy_model: Any,
    ) -> None:
        feedback_state = None
        if hasattr(policy, "capture_feedback_state"):
            feedback_state = self._capture_feedback_state(
                policy=policy,
                policy_model=policy_model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
            )
        self._pending.append(
            PendingDelayedOutcome(
                step=step,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
                recommended=recommended,
                action_taken=action_taken,
                predictions=predictions,
                probabilities=probabilities,
                parameter_drift=parameter_drift,
                feedback_state=feedback_state,
                abstained=abstained,
                batch_id=batch_id,
            )
        )

    def reveal(
        self,
        step: int,
        labels: np.ndarray | list[int],
        *,
        frozen_baseline_accuracy: float | None = None,
        batch_id: str | None = None,
        reveal_step: int,
        policy: Any,
        policy_model: Any,
        on_revealed: Any = None,
    ) -> dict[str, float]:
        """Apply delayed feedback once labels arrive. Raises KeyError if not found."""
        labels_array = np.asarray(labels, dtype=np.int64)
        idx = self._find_index(step=step, batch_id=batch_id)
        if idx is None:
            key = batch_id if batch_id is not None else f"step={step}"
            raise KeyError(f"no pending batch for {key}")

        pending = self._pending[idx]
        del self._pending[idx]

        metrics = apply_policy_feedback(
            policy,
            policy_model=policy_model,
            pending=pending,
            labels=labels_array,
            frozen_baseline_accuracy=frozen_baseline_accuracy,
            reveal_step=reveal_step,
            pending_outstanding_count=len(self._pending),
        )
        metrics["batch_id"] = pending.batch_id
        metrics["step"] = float(pending.step)
        self._revealed_metrics.append(metrics)
        self._revealed_positive_rates.append(float(np.mean(labels_array)))

        if on_revealed is not None:
            on_revealed()

        return metrics

    def publish_summary(self, policy: Any, current_step: int) -> None:
        """Push pending-queue stats into policies that support it."""
        if not hasattr(policy, "update_pending_feedback_summary"):
            return
        count = len(self._pending)
        if count == 0:
            policy.update_pending_feedback_summary(
                pending_count=0,
                mean_age=0.0,
                max_age=0.0,
                stale_fraction=0.0,
            )
            return
        ages = np.asarray(
            [max(0, current_step - item.step) for item in self._pending],
            dtype=np.float64,
        )
        stale_threshold = max(
            4,
            self._config.replay.label_delay_steps * 2
            if self._config.replay.label_delay_steps > 0
            else 8,
        )
        policy.update_pending_feedback_summary(
            pending_count=count,
            mean_age=float(np.mean(ages)),
            max_age=float(np.max(ages)),
            stale_fraction=float(np.mean(ages >= stale_threshold)),
        )

    def find_index(self, *, step: int, batch_id: str | None) -> int | None:
        return self._find_index(step=step, batch_id=batch_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_index(self, *, step: int, batch_id: str | None) -> int | None:
        if batch_id is not None:
            return next(
                (i for i, item in enumerate(self._pending) if item.batch_id == batch_id),
                None,
            )
        return next(
            (i for i, item in enumerate(self._pending) if item.step == step),
            None,
        )

    @staticmethod
    def _capture_feedback_state(
        *,
        policy: Any,
        policy_model: Any,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        decision: TabularDecision,
    ) -> object | None:
        capture = policy.capture_feedback_state
        params = list(inspect.signature(capture).parameters)
        if not params:
            return capture()
        return capture(
            model=policy_model,
            batch=batch,
            signal=signal,
            risk_state=risk_state,
            decision=decision,
        )
