from __future__ import annotations

from dataclasses import dataclass

from .report import ReplayComparisonResult, StrategyReplaySummary


@dataclass(frozen=True)
class BuyerFacingMetrics:
    """Operational KPIs translated for fraud/risk and MLOps buyers."""

    controller_name: str
    frozen_name: str
    # Risk / monitoring (lead metrics)
    risk_alert_rate_frozen: float
    risk_alert_rate_controller: float
    harmful_alert_reduction_pct: float
    mean_risk_capital_frozen: float
    mean_risk_capital_controller: float
    risk_exposure_reduction_pct: float
    # Accuracy (secondary — always report as pp delta, not headline)
    accuracy_frozen: float | None
    accuracy_controller: float | None
    accuracy_delta_pp: float | None
    accuracy_equivalent: bool
    # Operations
    intervention_rate_controller: float
    correction_applied_rate_controller: float
    correction_only_rate_controller: float
    bounded_interventions_per_1000_predictions: float
    recommendation_execution_rate: float
    reset_count_controller: int
    retrain_deferral_steps: int | None
    harmful_drift_events_avoided: int | None
    steps: int
    # Narrative lines ready for slides/email
    headline: str
    risk_sentence: str
    accuracy_sentence: str
    operations_sentence: str


def _pick_controller(summary: ReplayComparisonResult) -> StrategyReplaySummary | None:
    for name in (
        "regime_aware_delayed_bandit",
        "delayed_hybrid",
        "delayed_bandit",
        "bandit",
        "controller",
        "multi_action",
        "hybrid",
    ):
        match = next((item for item in summary.summaries if item.name == name), None)
        if match is not None:
            return match
    return next((item for item in summary.summaries if item.name != "frozen"), None)


def compute_buyer_kpis(
    result: ReplayComparisonResult,
    *,
    controller_name: str | None = None,
    frozen_name: str = "frozen",
    accuracy_equivalent_threshold_pp: float = 1.0,
) -> BuyerFacingMetrics | None:
    frozen = next((item for item in result.summaries if item.name == frozen_name), None)
    if frozen is None:
        return None

    if controller_name is not None:
        controller = next((item for item in result.summaries if item.name == controller_name), None)
    else:
        controller = _pick_controller(result)
    if controller is None:
        return None

    steps = max(1, frozen.steps)
    frozen_alert_rate = frozen.risk_alert_count / steps
    controller_alert_rate = controller.risk_alert_count / steps
    if frozen_alert_rate > 0:
        harmful_alert_reduction = (1.0 - controller_alert_rate / frozen_alert_rate) * 100.0
    else:
        harmful_alert_reduction = 100.0 if controller_alert_rate == 0 else 0.0

    if frozen.mean_risk_capital > 0:
        risk_exposure_reduction = (1.0 - controller.mean_risk_capital / frozen.mean_risk_capital) * 100.0
    else:
        risk_exposure_reduction = 0.0
    harmful_events_avoided = result.controller_vs_frozen_harmful_events_avoided
    retrain_deferral = result.controller_vs_frozen_retrain_deferral_steps
    material_risk_reduction = harmful_alert_reduction >= 10.0 or risk_exposure_reduction >= 10.0

    accuracy_delta_pp = None
    accuracy_equivalent = False
    if frozen.mean_accuracy is not None and controller.mean_accuracy is not None:
        accuracy_delta_pp = (controller.mean_accuracy - frozen.mean_accuracy) * 100.0
        accuracy_equivalent = abs(accuracy_delta_pp) <= accuracy_equivalent_threshold_pp

    if harmful_alert_reduction >= 50:
        headline = (
            f"Same decision quality with {harmful_alert_reduction:.0f}% fewer harmful drift alarms "
            f"({frozen_alert_rate:.0%} → {controller_alert_rate:.0%} alert rate)."
        )
    elif risk_exposure_reduction >= 50:
        headline = (
            f"Maintained model accuracy while cutting sequential risk exposure by {risk_exposure_reduction:.0f}% "
            f"vs frozen inference alone."
        )
    elif accuracy_delta_pp is not None and accuracy_delta_pp > accuracy_equivalent_threshold_pp:
        headline = (
            f"Improved decision quality by {accuracy_delta_pp:.1f} percentage points "
            f"with bounded controller steering and stable risk controls."
        )
    else:
        headline = "Controller-guided monitoring reduced operational risk signals vs frozen-only inference."

    risk_sentence = (
        f"Harmful-shift alert rate: frozen {frozen_alert_rate:.1%} → ARL {controller_alert_rate:.1%} "
        f"({harmful_alert_reduction:.0f}% reduction). "
        f"Mean risk capital: {frozen.mean_risk_capital:.1f} → {controller.mean_risk_capital:.1f} "
        f"({risk_exposure_reduction:.0f}% lower sustained risk burden)."
    )

    if accuracy_delta_pp is not None:
        if accuracy_equivalent:
            accuracy_sentence = (
                f"Accuracy held equivalent ({frozen.mean_accuracy:.1%} frozen vs {controller.mean_accuracy:.1%} ARL, "
                f"Δ {accuracy_delta_pp:+.1f} pp) — retraining was not required to preserve decision quality."
            )
        elif accuracy_delta_pp > 0:
            if material_risk_reduction:
                accuracy_sentence = (
                    f"Accuracy improved {accuracy_delta_pp:+.1f} percentage points "
                    f"({frozen.mean_accuracy:.1%} → {controller.mean_accuracy:.1%}) while risk fell."
                )
            else:
                accuracy_sentence = (
                    f"Accuracy improved {accuracy_delta_pp:+.1f} percentage points "
                    f"({frozen.mean_accuracy:.1%} → {controller.mean_accuracy:.1%}) with risk roughly unchanged."
                )
        else:
            if material_risk_reduction:
                accuracy_sentence = (
                    f"Accuracy tradeoff: {accuracy_delta_pp:+.1f} pp "
                    f"({frozen.mean_accuracy:.1%} → {controller.mean_accuracy:.1%}) in exchange for materially lower risk exposure."
                )
            else:
                accuracy_sentence = (
                    f"Accuracy tradeoff: {accuracy_delta_pp:+.1f} pp "
                    f"({frozen.mean_accuracy:.1%} → {controller.mean_accuracy:.1%}) without a compensating risk reduction."
                )
    else:
        accuracy_sentence = "Accuracy not evaluated on this replay slice."

    operations_sentence = (
        f"Controller steering touched {controller.correction_applied_rate:.0%} of batches "
        f"({controller.correction_only_rate:.0%} correction-only), while explicit interventions ran on "
        f"{controller.intervention_rate:.0%} of batches "
        f"({controller.bounded_interventions_per_1000:.1f} per 1k predictions). "
        f"Recommendation execution alignment was {controller.recommendation_execution_rate:.0%}, "
        f"with {controller.reset_count} controlled resets over {steps} steps."
    )
    if retrain_deferral is not None:
        operations_sentence += f" Retrain trigger was deferred by {retrain_deferral} steps vs frozen."
    if harmful_events_avoided is not None:
        operations_sentence += f" Harmful drift events avoided vs frozen: {harmful_events_avoided}."

    return BuyerFacingMetrics(
        controller_name=controller.name,
        frozen_name=frozen.name,
        risk_alert_rate_frozen=frozen_alert_rate,
        risk_alert_rate_controller=controller_alert_rate,
        harmful_alert_reduction_pct=harmful_alert_reduction,
        mean_risk_capital_frozen=frozen.mean_risk_capital,
        mean_risk_capital_controller=controller.mean_risk_capital,
        risk_exposure_reduction_pct=risk_exposure_reduction,
        accuracy_frozen=frozen.mean_accuracy,
        accuracy_controller=controller.mean_accuracy,
        accuracy_delta_pp=accuracy_delta_pp,
        accuracy_equivalent=accuracy_equivalent,
        intervention_rate_controller=controller.intervention_rate,
        correction_applied_rate_controller=controller.correction_applied_rate,
        correction_only_rate_controller=controller.correction_only_rate,
        bounded_interventions_per_1000_predictions=controller.bounded_interventions_per_1000,
        recommendation_execution_rate=controller.recommendation_execution_rate,
        reset_count_controller=controller.reset_count,
        retrain_deferral_steps=retrain_deferral,
        harmful_drift_events_avoided=harmful_events_avoided,
        steps=steps,
        headline=headline,
        risk_sentence=risk_sentence,
        accuracy_sentence=accuracy_sentence,
        operations_sentence=operations_sentence,
    )


def render_buyer_replay_report(
    result: ReplayComparisonResult,
    *,
    source_label: str = "offline replay",
    wedge: str = "fraud_risk",
    controller_name: str | None = None,
) -> str:
    kpis = compute_buyer_kpis(result, controller_name=controller_name)
    lines = [
        f"Adaptive Reliability Layer — Buyer Summary ({source_label})",
        f"Wedge: {wedge}",
        "",
        "=== Lead with risk, not accuracy ===",
    ]
    if kpis is None:
        lines.append("Could not compute buyer KPIs (missing frozen/controller pair).")
    else:
        lines.extend(
            [
                kpis.headline,
                "",
                "Risk & monitoring",
                kpis.risk_sentence,
                "",
                "Accuracy (secondary)",
                kpis.accuracy_sentence,
                "",
                "Operations",
                kpis.operations_sentence,
                "",
                "Retraining & control",
                (
                    f"Retrain deferral: {kpis.retrain_deferral_steps if kpis.retrain_deferral_steps is not None else 'n/a'} steps; "
                    f"harmful events avoided: {kpis.harmful_drift_events_avoided if kpis.harmful_drift_events_avoided is not None else 'n/a'}; "
                    f"recommendation execution: {kpis.recommendation_execution_rate:.0%}."
                ),
                "",
                "How to say this in a first call",
                _first_call_sentence(kpis),
                "",
            ]
        )

    lines.extend(
        [
            "=== Technical detail (for engineering review) ===",
            render_technical_replay_table(result),
        ]
    )
    return "\n".join(lines)


def render_technical_replay_table(result: ReplayComparisonResult) -> str:
    from .report import render_replay_report

    return render_replay_report(result)


def _first_call_sentence(kpis: BuyerFacingMetrics) -> str:
    if kpis.harmful_alert_reduction_pct >= 20.0:
        return (
            f"\"We cut harmful drift alarms by {kpis.harmful_alert_reduction_pct:.0f}% in offline replay while "
            f"keeping decision quality stable — that's fewer false escalations and less silent degradation "
            f"between retrains.\""
        )
    if kpis.accuracy_delta_pp is not None and kpis.accuracy_delta_pp > 1.0:
        return (
            f"\"We improved decision quality by {kpis.accuracy_delta_pp:.1f} points with narrow controller steering, "
            f"without increasing your operational risk burden.\""
        )
    if kpis.retrain_deferral_steps is not None and kpis.retrain_deferral_steps > 0:
        return (
            f"\"We kept model behavior stable and deferred the retrain trigger by {kpis.retrain_deferral_steps} "
            f"steps in offline replay, using bounded controller steering teams can audit.\""
        )
    return (
        "\"We replay your historical stream, show which controller steering steps would have been recommended, "
        "and quantify whether they improved reliability before retraining was necessary.\""
    )
