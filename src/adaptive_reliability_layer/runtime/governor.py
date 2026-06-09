from __future__ import annotations

from collections import deque

from ..tabular_benchmark import TabularShiftSignal
from ..risk import RiskState
from .config import RuntimeConfig
from .types import InterventionDecision, OperatingMode


class InterventionGovernor:
    """Tracks safety budgets and decides when the controller may act autonomously."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._recent_auto_action_steps: deque[int] = deque()
        self._recent_reset_steps: deque[int] = deque()
        self._recent_budget_block_steps: deque[int] = deque()
        # Rolling audit of the last 100 governor decisions for operator inspection.
        self._decision_log: deque[dict] = deque(maxlen=100)

    @property
    def decision_log(self) -> list[dict]:
        """Recent governor decisions (up to 100), newest last."""
        return list(self._decision_log)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_safety_budget(
        self,
        *,
        recommended: InterventionDecision,
        action_taken: str,
        reason_suffix: str,
        snapshot_before: object,
        adapter: object,
        step: int,
    ) -> tuple[str, str, OperatingMode, bool, str | None]:
        """Gate bounded_auto actions against per-window caps.

        Returns (action_taken, reason_suffix, effective_mode, budget_limited, budget_reason).
        """
        if action_taken in {"none", "hold"}:
            return action_taken, reason_suffix, OperatingMode.BOUNDED_AUTO, False, None

        budget = self._config.safety_budget
        if not budget.enabled:
            return action_taken, reason_suffix, OperatingMode.BOUNDED_AUTO, False, None

        auto_actions = self._trim_window(self._recent_auto_action_steps, step)
        resets = self._trim_window(self._recent_reset_steps, step)

        budget_reason: str | None = None
        if budget.max_auto_actions_per_window >= 0 and len(auto_actions) >= budget.max_auto_actions_per_window:
            budget_reason = "max_auto_actions_per_window"
        if (
            budget_reason is None
            and action_taken == "reset"
            and budget.max_resets_per_window >= 0
            and len(resets) >= budget.max_resets_per_window
        ):
            budget_reason = "max_resets_per_window"

        if budget_reason is None:
            self._decision_log.append({
                "step": step,
                "verdict": "allowed",
                "action": action_taken,
                "auto_actions_in_window": len(auto_actions),
                "resets_in_window": len(resets),
            })
            return action_taken, reason_suffix, OperatingMode.BOUNDED_AUTO, False, None

        self._recent_budget_block_steps.append(step)
        _load_snapshot(adapter, snapshot_before)
        self._decision_log.append({
            "step": step,
            "verdict": "blocked",
            "action": recommended.action,
            "budget_reason": budget_reason,
            "auto_actions_in_window": len(auto_actions),
            "resets_in_window": len(resets),
        })

        if budget.downgrade_to_recommend:
            return (
                "none",
                f"bounded_auto_budget_downgraded:{budget_reason}:{recommended.action}",
                OperatingMode.RECOMMEND,
                True,
                budget_reason,
            )
        return (
            "none",
            f"bounded_auto_budget_blocked:{budget_reason}:{recommended.action}",
            OperatingMode.BOUNDED_AUTO,
            True,
            budget_reason,
        )

    def record_action(self, action_taken: str, step: int) -> None:
        """Record that an action was executed, for budget tracking."""
        if action_taken in {"none", "hold", "abstain"}:
            return
        self._trim_window(self._recent_auto_action_steps, step)
        self._recent_auto_action_steps.append(step)
        if action_taken == "reset":
            self._trim_window(self._recent_reset_steps, step)
            self._recent_reset_steps.append(step)

    def should_retrain(
        self,
        *,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        recommended: InterventionDecision,
        action_taken: str,
        budget_limited: bool,
        monitor_saturated: bool = False,
        policy_name: str,
        step: int,
    ) -> bool:
        if policy_name == "frozen":
            return bool(risk_state.alert or signal.alert or signal.severe)
        severe_and_unmitigated = (
            (
                signal.severe
                or signal.collapse_risk >= 0.55
                or risk_state.capital >= self._config.monitor.risk_alert_threshold * 1.5
            )
            and action_taken in {"none", "hold", "abstain"}
        )
        blocked_under_pressure = budget_limited and (signal.alert or risk_state.alert)
        blocked_count = len(self._trim_window(self._recent_budget_block_steps, step))
        controller_stalled = (
            recommended.action in {"adapt", "reset"}
            and action_taken in {"none", "hold"}
        )
        return bool(
            severe_and_unmitigated
            or monitor_saturated
            or (blocked_under_pressure and blocked_count >= 2)
            or (controller_stalled and (risk_state.alert or signal.severe))
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim_window(self, values: deque[int], step: int) -> deque[int]:
        budget = self._config.safety_budget
        while values and step - values[0] >= budget.window_steps:
            values.popleft()
        return values


def _load_snapshot(adapter: object, snapshot: object) -> None:
    from .model_adapter import TorchTabularModelAdapter
    if isinstance(adapter, TorchTabularModelAdapter):
        adapter.load_snapshot(snapshot)  # type: ignore[arg-type]
    elif hasattr(adapter, "load_snapshot"):
        adapter.load_snapshot(snapshot)  # type: ignore[arg-type]
