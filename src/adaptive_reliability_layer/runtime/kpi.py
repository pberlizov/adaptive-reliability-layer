from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .types import DeploymentSurface


@dataclass(frozen=True)
class KpiConfig:
    """Business-weighted scoring for pilot and customer reports."""

    accuracy_weight: float = 1.0
    false_alert_cost: float = 0.06
    drift_cost: float = 0.03
    abstention_cost: float = 0.10
    reset_cost: float = 0.04
    retrain_recommendation_cost: float = 0.08
    harmful_alert_reduction_value: float = 1.0
    risk_capital_reduction_value: float = 0.02

    @classmethod
    def from_mapping(cls, data: dict) -> "KpiConfig":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


def score_surface_utility(surface: DeploymentSurface, config: KpiConfig) -> float:
    # Accuracy component: use actual measured batch accuracy if available, default to neutral 0.5
    batch_accuracy = (
        surface.batch_accuracy if surface.batch_accuracy is not None else 0.5
    )
    accuracy_component = batch_accuracy * config.accuracy_weight
    penalties = (
        config.false_alert_cost * float(surface.risk_alert)
        + config.drift_cost * min(1.0, surface.parameter_drift)
        + config.abstention_cost * float(surface.abstained)
        + config.reset_cost * float(surface.action_taken == "reset")
        + config.retrain_recommendation_cost * float(surface.retrain_recommended)
    )
    return accuracy_component - penalties


def summarize_business_kpis(
    surfaces: Sequence[DeploymentSurface],
    *,
    config: KpiConfig,
    baseline_alert_rate: float | None = None,
) -> dict[str, float | int | None]:
    if not surfaces:
        return {
            "steps": 0,
            "mean_business_score": 0.0,
            "risk_alert_rate": 0.0,
            "retrain_recommendation_rate": 0.0,
            "intervention_rate": 0.0,
            "harmful_alert_reduction_pct": None,
        }

    scores = [score_surface_utility(surface, config) for surface in surfaces]
    alerts = sum(1 for surface in surfaces if surface.risk_alert)
    retrains = sum(1 for surface in surfaces if surface.retrain_recommended)
    interventions = sum(
        1 for surface in surfaces if surface.action_taken not in {"none", "hold"}
    )
    alert_rate = alerts / len(surfaces)
    reduction = None
    if baseline_alert_rate is not None and baseline_alert_rate > 0:
        reduction = (1.0 - alert_rate / baseline_alert_rate) * 100.0

    return {
        "steps": len(surfaces),
        "mean_business_score": float(sum(scores) / len(scores)),
        "risk_alert_rate": alert_rate,
        "retrain_recommendation_rate": retrains / len(surfaces),
        "intervention_rate": interventions / len(surfaces),
        "harmful_alert_reduction_pct": reduction,
        "mean_risk_capital": float(sum(surface.risk_capital for surface in surfaces) / len(surfaces)),
    }
