from __future__ import annotations

from ..tabular_benchmark import TabularBatch, TabularDecision, TabularReferenceProfile, TabularShiftSignal
from ..risk import RiskState
from .model_adapter import ModelAdapter, TorchTabularModelAdapter
from .types import InterventionDecision, OperatingMode


def decision_from_tabular(decision: TabularDecision) -> InterventionDecision:
    return InterventionDecision(
        action=decision.action,
        reason=decision.reason,
        selected_fraction=decision.selected_fraction,
    )


def apply_operating_mode(
    *,
    mode: OperatingMode,
    bounded_auto_actions: frozenset[str],
    adapter: ModelAdapter,
    decision: InterventionDecision,
    snapshot_before: object,
) -> tuple[str, str]:
    """Return (action_taken, reason_suffix) after gating policy output."""

    recommended = decision.action
    if mode == OperatingMode.SHADOW:
        if isinstance(adapter, TorchTabularModelAdapter):
            adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
        elif hasattr(adapter, "load_snapshot"):
            adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
        return "none", "shadow_mode_no_mutation"

    if mode == OperatingMode.RECOMMEND:
        if isinstance(adapter, TorchTabularModelAdapter):
            adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
        elif hasattr(adapter, "load_snapshot"):
            adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
        return "none", "recommend_mode_pending_approval"

    if mode == OperatingMode.BOUNDED_AUTO:
        if recommended in bounded_auto_actions:
            return recommended, decision.reason
        if isinstance(adapter, TorchTabularModelAdapter):
            adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
        elif hasattr(adapter, "load_snapshot"):
            adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
        return "none", f"bounded_auto_blocked:{recommended}"

    return recommended, decision.reason


def trust_state_from_signal(
    signal: TabularShiftSignal,
    decision: InterventionDecision,
) -> str:
    if signal.collapse_risk >= 0.5 or decision.action == "reset":
        return "escalate"
    if signal.alert and decision.action in {"adapt", "reset"}:
        return "caution"
    if signal.alert:
        return "monitor"
    return "normal"


def reliability_score(
    signal: TabularShiftSignal,
    risk_state: RiskState,
    decision: InterventionDecision,
) -> float:
    base = 1.0 - 0.20 * signal.feature_score - 0.28 * signal.output_score - 0.32 * signal.collapse_risk
    risk_penalty = min(0.45, 0.04 * max(0.0, risk_state.capital - 1.0))
    action_penalty = 0.08 if decision.action == "reset" else 0.0
    action_penalty += 0.05 if decision.action == "abstain" else 0.0
    return max(0.0, min(1.0, base - risk_penalty - action_penalty))


def runtime_batch_to_tabular(batch: object) -> TabularBatch:
    from .types import RuntimeBatch

    if isinstance(batch, TabularBatch):
        return batch
    if isinstance(batch, RuntimeBatch):
        return TabularBatch(features=batch.features, labels=batch.labels, regime=batch.regime)
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def build_runtime_policy(name: str, reference: TabularReferenceProfile, policy_config: object):
    from ..tabular_benchmark import (
        BanditTabularPolicy,
        ControllerTabularPolicy,
        DelayedBanditTabularPolicy,
        DelayedHybridBanditSpecialistPolicy,
        FraudContextDelayedBanditTabularPolicy,
        FraudRankDelayedBanditTabularPolicy,
        FrozenTabularPolicy,
        HybridBanditSpecialistPolicy,
        MultiActionTabularPolicy,
        NaiveTabularPolicy,
        RegimeAwareDelayedBanditTabularPolicy,
        ScheduledRetrainTabularPolicy,
    )

    if name == "frozen":
        return FrozenTabularPolicy()
    if name == "naive":
        return NaiveTabularPolicy()
    if name == "scheduled_retrain":
        return ScheduledRetrainTabularPolicy(
            retrain_interval=getattr(policy_config, "scheduled_retrain_interval", 6),
        )
    if name == "controller":
        return ControllerTabularPolicy()
    allowed_actions = getattr(policy_config, "allowed_actions", None)
    use_behavior_signals = bool(getattr(policy_config, "use_behavior_signals", True))
    bandit_kwargs = {
        "alpha": getattr(policy_config, "bandit_alpha", 0.75),
    }
    if allowed_actions is not None:
        bandit_kwargs["allowed_actions"] = allowed_actions
    delayed_bandit_kwargs = {**bandit_kwargs}
    regime_aware_kwargs = {**bandit_kwargs, "use_behavior_signals": use_behavior_signals}
    if name == "bandit":
        return BanditTabularPolicy(reference, **bandit_kwargs)
    if name == "delayed_bandit":
        return DelayedBanditTabularPolicy(reference, **delayed_bandit_kwargs)
    if name in {"regime_aware_delayed_bandit", "regime_aware_bandit"}:
        return RegimeAwareDelayedBanditTabularPolicy(reference, **regime_aware_kwargs)
    if name == "fraud_rank_delayed_bandit":
        return FraudRankDelayedBanditTabularPolicy(reference, **regime_aware_kwargs)
    if name == "fraud_context_delayed_bandit":
        return FraudContextDelayedBanditTabularPolicy(reference, **regime_aware_kwargs)
    if name == "delayed_hybrid":
        return DelayedHybridBanditSpecialistPolicy(
            reference,
            max_specialists=getattr(policy_config, "max_specialists", 4),
            distance_threshold=getattr(policy_config, "distance_threshold", 1.35),
            controller_kwargs=regime_aware_kwargs,
            use_behavior_signals=use_behavior_signals,
        )
    if name == "hybrid":
        return HybridBanditSpecialistPolicy(reference)
    return MultiActionTabularPolicy(
        reference,
        mild_threshold=getattr(policy_config, "mild_threshold", 0.95),
        severe_threshold=getattr(policy_config, "severe_threshold", 1.55),
        cooldown_steps=getattr(policy_config, "cooldown_steps", 2),
    )
