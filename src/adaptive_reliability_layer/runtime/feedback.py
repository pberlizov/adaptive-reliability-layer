from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tabular_benchmark import (
    BanditFeedbackState,
    TabularBatch,
    TabularDecision,
    TabularShiftSignal,
    _compute_batch_utility,
)
from ..risk import RiskState
from .types import DeploymentSurface, InterventionDecision


@dataclass
class PendingDelayedOutcome:
    """Batch decision state waiting for delayed labels."""

    step: int
    batch: TabularBatch
    signal: TabularShiftSignal
    risk_state: RiskState
    decision: TabularDecision
    recommended: InterventionDecision
    action_taken: str
    predictions: list[int]
    probabilities: list[float]
    parameter_drift: float
    feedback_state: BanditFeedbackState | None = None
    abstained: bool = False
    batch_id: str | None = None


def compute_batch_utility(
    *,
    batch_accuracy: float,
    risk_alert: bool,
    parameter_drift: float,
    abstained: bool,
    action_taken: str,
    risk_capital: float = 1.0,
) -> float:
    effective_action = "abstain" if abstained else action_taken
    decision = TabularDecision(action=effective_action, reason="runtime")
    risk_state = RiskState(
        raw_score=0.0,
        p_value=1.0,
        e_value=1.0,
        capital=risk_capital,
        alert=risk_alert,
    )
    return _compute_batch_utility(
        batch_accuracy=batch_accuracy,
        risk_state=risk_state,
        decision=decision,
        parameter_drift=parameter_drift,
    )


def apply_policy_feedback(
    policy: Any,
    *,
    policy_model: Any,
    pending: PendingDelayedOutcome,
    labels: np.ndarray,
    frozen_baseline_accuracy: float | None = None,
    reveal_step: int | None = None,
    pending_outstanding_count: int = 0,
) -> dict[str, float]:
    """Update learnable policies when labels are revealed."""

    predictions = np.asarray(pending.predictions, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    batch = pending.batch
    if batch.labels is None:
        batch = TabularBatch(
            features=batch.features,
            labels=labels_array,
            regime=batch.regime,
        )
    if pending.abstained:
        served_mask = np.zeros(len(labels_array), dtype=bool)
    else:
        served_mask = np.ones(len(labels_array), dtype=bool)

    if served_mask.any():
        batch_accuracy = float((predictions[served_mask] == labels_array[served_mask]).mean())
    else:
        batch_accuracy = 0.0
    coverage = float(served_mask.mean())
    utility = compute_batch_utility(
        batch_accuracy=batch_accuracy,
        risk_alert=pending.risk_state.alert,
        parameter_drift=pending.parameter_drift,
        abstained=pending.abstained,
        action_taken=pending.action_taken,
        risk_capital=pending.risk_state.capital,
    )
    reliability = max(0.0, min(1.0, utility))
    mean_probability = float(np.mean(pending.probabilities)) if pending.probabilities else 0.5
    revealed_positive_rate = float(np.mean(labels_array))
    revealed_mean_residual = float(np.mean(labels_array - np.asarray(pending.probabilities, dtype=np.float64)))
    pending_delay_steps = 0 if reveal_step is None else max(0, int(reveal_step - pending.step))

    if hasattr(policy, "observe_delayed_outcome") and pending.feedback_state is not None:
        policy.observe_delayed_outcome(
            feedback_state=pending.feedback_state,
            model=policy_model,
            batch=batch,
            signal=pending.signal,
            risk_state=pending.risk_state,
            decision=pending.decision,
            batch_accuracy=batch_accuracy,
            reliability=reliability,
            utility=utility,
            retrospective_reward=None,  # let policy compute reward + lift from utility
            revealed_accuracy=batch_accuracy,
            revealed_coverage=coverage,
            revealed_baseline_accuracy=frozen_baseline_accuracy,
            pending_delay_steps=pending_delay_steps,
            pending_outstanding_count=pending_outstanding_count,
            revealed_mean_residual=revealed_mean_residual,
            predicted_positive_rate=mean_probability,
            revealed_positive_rate=revealed_positive_rate,
        )
    elif hasattr(policy, "observe_outcome"):
        policy.observe_outcome(
            model=policy_model,
            batch=batch,
            signal=pending.signal,
            risk_state=pending.risk_state,
            decision=pending.decision,
            batch_accuracy=batch_accuracy,
            reliability=reliability,
            utility=utility,
        )

    return {
        "batch_accuracy": batch_accuracy,
        "utility": utility,
        "coverage": coverage,
    }
