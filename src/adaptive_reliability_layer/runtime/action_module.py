"""ActionModule protocol — pluggable action interface for ARL.

Any callable that satisfies the ActionModule protocol can be registered as a
first-class action and used by policies and the runtime layer.

Usage
-----
    from adaptive_reliability_layer.runtime.action_module import ActionModule, ActionResult, register_action

    class MyCustomAction:
        name = "my_action"
        risk_tier = "low"

        def apply(self, adapter, batch, signal, risk_state, reference, *, human_approved=False):
            # ... mutate adapter ...
            return ActionResult(action="my_action", reason="custom_logic", selected_fraction=0.0)

    register_action(MyCustomAction())
    # Now "my_action" can be added to bounded_auto_actions in RuntimeConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionResult:
    """Outcome of applying an action module."""

    action: str
    reason: str
    selected_fraction: float = 0.0
    metadata: dict[str, Any] | None = None


@runtime_checkable
class ActionModule(Protocol):
    """Protocol any pluggable action must satisfy.

    Attributes
    ----------
    name : str
        Unique action identifier used in audit records and config.
    risk_tier : str
        One of ``"low"`` or ``"high"``.  Low-risk actions may be used in
        ``bounded_auto``; high-risk actions require explicit allow-listing or
        human approval.

    Methods
    -------
    apply(adapter, batch, signal, risk_state, reference, *, human_approved)
        Mutate the adapter in-place and return an ActionResult.
        MUST be idempotent when applied on an already-snapshotted adapter
        (the caller will have taken a pre-snapshot before calling apply).
    """

    name: str
    risk_tier: str  # "low" or "high"

    def apply(
        self,
        adapter: Any,
        batch: Any,
        signal: Any,
        risk_state: Any,
        reference: Any,
        *,
        human_approved: bool = False,
    ) -> ActionResult:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ActionModule] = {}


def register_action(module: ActionModule) -> None:
    """Register a custom action module.

    After registration, the action name can be added to
    ``RuntimeConfig.bounded_auto_actions`` (if ``risk_tier=="low"``) or
    used in ``approve_and_apply`` (any tier).

    Raises
    ------
    TypeError
        If ``module`` does not satisfy the ActionModule protocol.
    ValueError
        If an action with the same name is already registered.
    """
    if not isinstance(module, ActionModule):
        raise TypeError(
            f"{module!r} does not satisfy the ActionModule protocol. "
            "It must have a 'name', 'risk_tier', and an 'apply' method."
        )
    if module.name in _REGISTRY:
        raise ValueError(
            f"Action '{module.name}' is already registered. "
            "Use deregister_action first to replace it."
        )
    _REGISTRY[module.name] = module


def deregister_action(name: str) -> bool:
    """Remove a registered custom action. Returns True if it existed."""
    return _REGISTRY.pop(name, None) is not None


def get_action(name: str) -> ActionModule | None:
    """Return a registered action module by name, or None if not found."""
    return _REGISTRY.get(name)


def registered_action_names() -> list[str]:
    """Return the names of all registered custom action modules."""
    return list(_REGISTRY)


def apply_registered_action(
    name: str,
    adapter: Any,
    batch: Any,
    signal: Any,
    risk_state: Any,
    reference: Any,
    *,
    human_approved: bool = False,
) -> ActionResult | None:
    """Apply a registered action by name.  Returns None if not registered."""
    module = _REGISTRY.get(name)
    if module is None:
        return None
    return module.apply(adapter, batch, signal, risk_state, reference, human_approved=human_approved)


# ---------------------------------------------------------------------------
# Built-in action modules (thin wrappers over existing primitives)
# ---------------------------------------------------------------------------

class _BnRefreshAction:
    name = "bn_refresh"
    risk_tier = "low"

    def apply(self, adapter, batch, signal, risk_state, reference, *, human_approved=False):
        del signal, risk_state, reference, human_approved
        import numpy as np
        adapter.refresh_batch_norm(np.asarray(batch.features, dtype=np.float32), passes=2)
        return ActionResult(action=self.name, reason="bn_refresh_module")


class _RecalibrateAction:
    name = "recalibrate"
    risk_tier = "low"

    def apply(self, adapter, batch, signal, risk_state, reference, *, human_approved=False):
        del batch, risk_state, human_approved
        adapter.recalibrate_temperature(
            reference_confidence=reference.mean_confidence,
            observed_confidence=signal.mean_confidence,
            momentum=0.25,
        )
        return ActionResult(action=self.name, reason="recalibrate_module")


class _CoolConfidenceAction:
    name = "cool_confidence"
    risk_tier = "low"

    def apply(self, adapter, batch, signal, risk_state, reference, *, human_approved=False):
        del batch, risk_state, human_approved
        overconfidence_gap = signal.mean_confidence - reference.mean_confidence
        if not human_approved and overconfidence_gap < 0.04:
            return ActionResult(action="hold", reason="cool_confidence_not_overconfident")
        adapter.recalibrate_temperature(
            reference_confidence=reference.mean_confidence,
            observed_confidence=signal.mean_confidence,
            momentum=0.15,
        )
        return ActionResult(action=self.name, reason="cool_confidence_module")


class _LatentRecenterAction:
    name = "latent_recenter"
    risk_tier = "low"

    def apply(self, adapter, batch, signal, risk_state, reference, *, human_approved=False):
        del signal, risk_state, reference, human_approved
        import numpy as np
        adapter.apply_latent_recenter(np.asarray(batch.features, dtype=np.float32), momentum=0.12)
        return ActionResult(action=self.name, reason="latent_recenter_module")


# Register built-ins (idempotent — skip if already registered from a prior import)
for _module in [_BnRefreshAction(), _RecalibrateAction(), _CoolConfidenceAction(), _LatentRecenterAction()]:
    if _module.name not in _REGISTRY:
        _REGISTRY[_module.name] = _module  # type: ignore[assignment]
