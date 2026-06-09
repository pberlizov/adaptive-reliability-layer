from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OperatingMode(str, Enum):
    """How the reliability layer may affect the underlying model."""

    SHADOW = "shadow"
    RECOMMEND = "recommend"
    BOUNDED_AUTO = "bounded_auto"


class TrustState(str, Enum):
    NORMAL = "normal"
    MONITOR = "monitor"
    CAUTION = "caution"
    ESCALATE = "escalate"


# Actions considered low-risk for bounded_auto by default.
DEFAULT_BOUNDED_AUTO_ACTIONS: frozenset[str] = frozenset(
    {"none", "hold", "bn_refresh", "bn_refresh_only", "covariate_refresh", "recalibrate",
     "cool_confidence", "label_shift", "bbse_label_shift", "latent_recenter"}
)

HIGH_RISK_ACTIONS: frozenset[str] = frozenset({"adapt", "reset", "abstain"})

PROFILE_BOUNDED_ACTIONS: dict[str, dict[str, frozenset[str]]] = {
    "general": {
        "stable": DEFAULT_BOUNDED_AUTO_ACTIONS,
        "covariate_drift": frozenset({"none", "hold", "bn_refresh", "covariate_refresh", "recalibrate", "cool_confidence", "latent_recenter"}),
        "label_drift": frozenset({"none", "hold", "recalibrate", "cool_confidence", "label_shift", "bbse_label_shift"}),
        "mixed_drift": frozenset({"none", "hold", "bn_refresh", "covariate_refresh", "recalibrate", "cool_confidence", "label_shift", "bbse_label_shift"}),
        "collapse_risk": frozenset({"none", "hold", "recalibrate"}),
        "reference_break": frozenset({"none", "hold"}),
    },
    "fraud": {
        "stable": DEFAULT_BOUNDED_AUTO_ACTIONS,
        "covariate_drift": frozenset({"none", "hold", "recalibrate", "cool_confidence", "label_shift"}),
        "label_drift": frozenset({"none", "hold", "recalibrate", "cool_confidence", "label_shift", "bbse_label_shift"}),
        "mixed_drift": frozenset({"none", "hold", "recalibrate", "cool_confidence", "label_shift", "bbse_label_shift"}),
        "collapse_risk": frozenset({"none", "hold", "recalibrate", "label_shift"}),
        "reference_break": frozenset({"none", "hold"}),
    },
    "sensor": {
        "stable": DEFAULT_BOUNDED_AUTO_ACTIONS,
        "covariate_drift": frozenset({"none", "hold", "bn_refresh", "covariate_refresh", "recalibrate", "cool_confidence", "latent_recenter"}),
        "label_drift": frozenset({"none", "hold", "recalibrate", "cool_confidence"}),
        "mixed_drift": frozenset({"none", "hold", "bn_refresh", "covariate_refresh", "recalibrate", "cool_confidence"}),
        "collapse_risk": frozenset({"none", "hold", "recalibrate"}),
        "reference_break": frozenset({"none", "hold"}),
    },
    "sensor_safe": {
        "stable": frozenset({"none", "hold"}),
        "covariate_drift": frozenset({"none", "hold"}),
        "label_drift": frozenset({"none", "hold", "recalibrate", "cool_confidence"}),
        "mixed_drift": frozenset({"none", "hold"}),
        "collapse_risk": frozenset({"none", "hold"}),
        "reference_break": frozenset({"none", "hold"}),
    },
}


@dataclass(frozen=True)
class RuntimeBatch:
    features: Any
    labels: Any | None = None
    regime: str = "live"
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterventionDecision:
    action: str
    reason: str
    selected_fraction: float = 0.0


@dataclass(frozen=True)
class DeploymentSurface:
    """Stable commercial API contract emitted on every batch."""

    step: int
    predictions: list[int]
    probabilities: list[float]
    confidence: float
    shift_score: float
    feature_shift_score: float
    output_shift_score: float
    collapse_risk: float
    risk_capital: float
    risk_alert: bool
    regime_hint: str
    recommended_action: str
    action_taken: str
    intervention_reason: str
    trust_state: str
    reliability_score: float
    parameter_drift: float
    operating_mode: str
    model_version: str
    specialist_id: str | None
    rollback_available: bool
    snapshot_id: str | None
    abstained: bool
    decision_record_version: str = "1.1"
    effective_operating_mode: str | None = None
    regime_id: str | None = None
    regime_confidence: float = 0.0
    regime_novelty: float = 0.0
    risk_score: float = 0.0
    batch_accuracy: float | None = None
    why_this_action: str = ""
    recommended_action_requires_approval: bool = False
    rollback_eligible: bool = False
    retrain_recommended: bool = False
    budget_limited: bool = False
    budget_reason: str | None = None
    batch_id: str | None = None
    shift_signature: str = "stable"
    controller_profile: str = "general"
    adaptation_opportunity_score: float = 0.0
    monitor_saturated: bool = False
    asr_class_concentration: float = 0.0
    drift_detector_score: float = 0.0
    timescale_expert: str = "medium"
    uncertainty_action: str = "hold_threshold"
    conformal_alpha: float = 0.10
    conformal_half_width: float = 0.0
    adaptation_safety_ok: bool = True
    proactive_hold: bool = False
    deferred_adaptation: bool = False
    correction_mean_abs_delta: float = 0.0
    correction_max_abs_delta: float = 0.0
    correction_flipped_predictions: int = 0
    correction_applied: bool = False
    explicit_action_executed: bool = False
    decision_threshold: float = 0.5
    threshold_shift: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def decision_record(self) -> dict[str, Any]:
        return {
            "version": self.decision_record_version,
            "step": self.step,
            "operating_mode": self.operating_mode,
            "effective_operating_mode": self.effective_operating_mode or self.operating_mode,
            "regime_id": self.regime_id or self.regime_hint,
            "regime_confidence": self.regime_confidence,
            "regime_novelty": self.regime_novelty,
            "recommended_action": self.recommended_action,
            "action_taken": self.action_taken,
            "why_this_action": self.why_this_action or self.intervention_reason,
            "risk_score": self.risk_score,
            "risk_capital": self.risk_capital,
            "risk_alert": self.risk_alert,
            "trust_state": self.trust_state,
            "reliability_score": self.reliability_score,
            "rollback_eligible": self.rollback_eligible or self.rollback_available,
            "retrain_recommended": self.retrain_recommended,
            "budget_limited": self.budget_limited,
            "budget_reason": self.budget_reason,
            "batch_id": self.batch_id,
            "shift_signature": self.shift_signature,
            "controller_profile": self.controller_profile,
            "adaptation_opportunity_score": self.adaptation_opportunity_score,
            "monitor_saturated": self.monitor_saturated,
            "asr_class_concentration": self.asr_class_concentration,
            "drift_detector_score": self.drift_detector_score,
            "timescale_expert": self.timescale_expert,
            "uncertainty_action": self.uncertainty_action,
            "conformal_alpha": self.conformal_alpha,
            "adaptation_safety_ok": self.adaptation_safety_ok,
            "proactive_hold": self.proactive_hold,
            "deferred_adaptation": self.deferred_adaptation,
            "correction_mean_abs_delta": self.correction_mean_abs_delta,
            "correction_max_abs_delta": self.correction_max_abs_delta,
            "correction_flipped_predictions": self.correction_flipped_predictions,
            "correction_applied": self.correction_applied,
            "explicit_action_executed": self.explicit_action_executed,
            "decision_threshold": self.decision_threshold,
            "threshold_shift": self.threshold_shift,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class AuditRecord:
    record_id: str
    step: int
    timestamp: str
    operating_mode: str
    model_version: str
    recommended_action: str
    action_taken: str
    intervention_reason: str
    shift_score: float
    risk_capital: float
    risk_alert: bool
    trust_state: str
    snapshot_id_before: str | None
    snapshot_id_after: str | None
    approved_by: str | None
    metadata_json: str
