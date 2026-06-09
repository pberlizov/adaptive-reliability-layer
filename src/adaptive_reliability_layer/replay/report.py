from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from ..runtime.types import DeploymentSurface
from .types import ReplayRunState


@dataclass(frozen=True)
class StrategyReplaySummary:
    name: str
    steps: int
    mean_accuracy: float | None
    mean_utility: float
    mean_risk_capital: float
    mean_shift_score: float
    mean_reliability: float
    intervention_rate: float
    reset_count: int
    risk_alert_count: int
    first_drift_step: int | None
    recommendation_rate: float
    recommendation_execution_rate: float
    bounded_interventions_per_1000: float
    retrain_recommendation_count: int
    first_retrain_recommendation_step: int | None
    budget_limited_count: int
    correction_applied_rate: float = 0.0
    correction_only_rate: float = 0.0
    mean_correction_flipped_predictions: float = 0.0
    mean_abs_threshold_shift: float = 0.0


@dataclass(frozen=True)
class ReplayTimeline:
    name: str
    surfaces: tuple[DeploymentSurface, ...]


@dataclass(frozen=True)
class ReplayComparisonResult:
    summaries: tuple[StrategyReplaySummary, ...]
    controller_vs_frozen_utility_delta: float | None
    controller_vs_frozen_risk_reduction: float | None
    controller_vs_frozen_harmful_events_avoided: int | None
    controller_vs_frozen_retrain_deferral_steps: int | None
    timelines: tuple[ReplayTimeline, ...] = ()


@dataclass(frozen=True)
class DriftEpisode:
    start_step: int
    end_step: int
    peak_shift_score: float
    peak_risk_capital: float
    action_summary: str
    retrain_recommended: bool


_CONTROLLER_NAMES = frozenset(
    {
        "controller",
        "multi_action",
        "bandit",
        "delayed_bandit",
        "regime_aware_delayed_bandit",
        "delayed_hybrid",
        "naive",
        "scheduled_retrain",
    }
)


def utility_delta_vs_baseline(
    result: ReplayComparisonResult,
    *,
    controller_name: str,
    baseline_name: str,
) -> float | None:
    controller = next((item for item in result.summaries if item.name == controller_name), None)
    baseline = next((item for item in result.summaries if item.name == baseline_name), None)
    if controller is None or baseline is None:
        return None
    return controller.mean_utility - baseline.mean_utility


def adaptation_safety_rate(timeline: ReplayTimeline | None) -> float | None:
    if timeline is None or not timeline.surfaces:
        return None
    mutating = {"adapt", "reset", "bn_refresh", "covariate_refresh", "recalibrate", "label_shift"}
    checked = 0
    safe = 0
    for surface in timeline.surfaces:
        if surface.action_taken not in mutating:
            continue
        checked += 1
        if surface.adaptation_safety_ok:
            safe += 1
    if checked == 0:
        return None
    return safe / checked


def summarize_replay_runs(
    runs: Sequence[ReplayRunState],
    *,
    controller_name: str | None = None,
) -> ReplayComparisonResult:
    summaries = tuple(_summarize_run(run) for run in runs)
    timelines = tuple(ReplayTimeline(name=run.name, surfaces=tuple(run.surfaces)) for run in runs)
    frozen = next((summary for summary in summaries if summary.name == "frozen"), None)
    if controller_name is not None:
        controller = next((summary for summary in summaries if summary.name == controller_name), None)
        controller_timeline = next((timeline for timeline in timelines if timeline.name == controller_name), None)
    else:
        controller = next(
            (summary for summary in reversed(summaries) if summary.name in _CONTROLLER_NAMES),
            None,
        )
        controller_timeline = next(
            (timeline for timeline in reversed(timelines) if timeline.name in _CONTROLLER_NAMES),
            None,
        )
    frozen_timeline = next((timeline for timeline in timelines if timeline.name == "frozen"), None)
    utility_delta = None
    risk_reduction = None
    harmful_events_avoided = None
    retrain_deferral = None
    if frozen and controller:
        utility_delta = controller.mean_utility - frozen.mean_utility
        capital_reduction = None
        if frozen.mean_risk_capital > 0:
            capital_reduction = 1.0 - (controller.mean_risk_capital / frozen.mean_risk_capital)
        frozen_alert_rate = frozen.risk_alert_count / max(1, frozen.steps)
        controller_alert_rate = controller.risk_alert_count / max(1, controller.steps)
        alert_reduction = None
        if frozen_alert_rate > 0:
            alert_reduction = 1.0 - (controller_alert_rate / frozen_alert_rate)
        retrain_reduction = None
        if frozen.retrain_recommendation_count > 0:
            retrain_reduction = 1.0 - (
                controller.retrain_recommendation_count / frozen.retrain_recommendation_count
            )
        candidates = [
            value
            for value in (capital_reduction, alert_reduction, retrain_reduction)
            if value is not None
        ]
        risk_reduction = max(candidates) if candidates else None
    if frozen_timeline is not None and controller_timeline is not None:
        harmful_events_avoided = _harmful_events_avoided(frozen_timeline, controller_timeline)
        retrain_deferral = _retrain_deferral_steps(frozen_timeline, controller_timeline)
    return ReplayComparisonResult(
        summaries=summaries,
        controller_vs_frozen_utility_delta=utility_delta,
        controller_vs_frozen_risk_reduction=risk_reduction,
        controller_vs_frozen_harmful_events_avoided=harmful_events_avoided,
        controller_vs_frozen_retrain_deferral_steps=retrain_deferral,
        timelines=timelines,
    )


def _summarize_run(run: ReplayRunState) -> StrategyReplaySummary:
    mutating_actions = {"bn_refresh", "covariate_refresh", "recalibrate", "label_shift", "adapt", "reset"}
    correction_applied = sum(1 for surface in run.surfaces if surface.correction_applied)
    correction_only = sum(
        1 for surface in run.surfaces if surface.correction_applied and not surface.explicit_action_executed
    )
    interventions = sum(1 for surface in run.surfaces if surface.action_taken not in {"none", "hold"})
    recommendations = sum(1 for surface in run.surfaces if surface.recommended_action not in {"none", "hold"})
    executed_recommendations = sum(
        1 for surface in run.surfaces if surface.recommended_action == surface.action_taken and surface.action_taken not in {"none", "hold"}
    )
    first_drift = next((surface.step for surface in run.surfaces if surface.shift_score >= 1.0), None)
    retrain_steps = [surface.step for surface in run.surfaces if surface.retrain_recommended]
    return StrategyReplaySummary(
        name=run.name,
        steps=len(run.surfaces),
        mean_accuracy=mean(run.accuracies) if run.accuracies else None,
        mean_utility=mean(run.utilities) if run.utilities else 0.0,
        mean_risk_capital=mean(run.risk_capitals) if run.risk_capitals else 0.0,
        mean_shift_score=mean(run.shift_scores) if run.shift_scores else 0.0,
        mean_reliability=mean([surface.reliability_score for surface in run.surfaces]) if run.surfaces else 0.0,
        intervention_rate=interventions / max(1, len(run.surfaces)),
        reset_count=sum(1 for surface in run.surfaces if surface.action_taken == "reset"),
        risk_alert_count=sum(1 for surface in run.surfaces if surface.risk_alert),
        first_drift_step=first_drift,
        recommendation_rate=recommendations / max(1, len(run.surfaces)),
        recommendation_execution_rate=executed_recommendations / max(1, recommendations),
        bounded_interventions_per_1000=1000.0 * sum(1 for surface in run.surfaces if surface.action_taken in mutating_actions) / max(1, len(run.surfaces)),
        retrain_recommendation_count=len(retrain_steps),
        first_retrain_recommendation_step=retrain_steps[0] if retrain_steps else None,
        budget_limited_count=sum(1 for surface in run.surfaces if surface.budget_limited),
        correction_applied_rate=correction_applied / max(1, len(run.surfaces)),
        correction_only_rate=correction_only / max(1, len(run.surfaces)),
        mean_correction_flipped_predictions=mean(
            [surface.correction_flipped_predictions for surface in run.surfaces]
        ) if run.surfaces else 0.0,
        mean_abs_threshold_shift=mean(
            [abs(surface.threshold_shift) for surface in run.surfaces]
        ) if run.surfaces else 0.0,
    )


def _harmful_events_avoided(frozen: ReplayTimeline, controller: ReplayTimeline) -> int:
    paired = zip(frozen.surfaces, controller.surfaces)
    return sum(
        1
        for frozen_surface, controller_surface in paired
        if (frozen_surface.risk_alert or frozen_surface.retrain_recommended)
        and not (controller_surface.risk_alert or controller_surface.retrain_recommended)
    )


def _retrain_deferral_steps(frozen: ReplayTimeline, controller: ReplayTimeline) -> int | None:
    frozen_step = next((surface.step for surface in frozen.surfaces if surface.retrain_recommended), None)
    controller_step = next((surface.step for surface in controller.surfaces if surface.retrain_recommended), None)
    if frozen_step is None and controller_step is None:
        return 0
    if frozen_step is None:
        return None
    if controller_step is None:
        return len(controller.surfaces) - frozen_step
    return controller_step - frozen_step


def render_replay_report(result: ReplayComparisonResult) -> str:
    lines = [
        "Adaptive Reliability Layer Offline Replay Report",
        "",
        "strategy                 steps   accuracy   utility   risk_capital   reliability   shift_score   recommendation_rate   exec_rate   corr_rate   corr_only   retrain_flags   budget_limited",
    ]
    for summary in result.summaries:
        accuracy = f"{summary.mean_accuracy:.3f}" if summary.mean_accuracy is not None else "n/a"
        first_retrain = summary.first_retrain_recommendation_step if summary.first_retrain_recommendation_step is not None else "n/a"
        lines.append(
            f"{summary.name:<24} {summary.steps:>5}   {accuracy:>8}   "
            f"{summary.mean_utility:>7.3f}   {summary.mean_risk_capital:>12.3f}   "
            f"{summary.mean_reliability:>10.3f}   {summary.mean_shift_score:>11.3f}   "
            f"{summary.recommendation_rate:>18.3f}   {summary.recommendation_execution_rate:>8.3f}   "
            f"{summary.correction_applied_rate:>9.3f}   {summary.correction_only_rate:>9.3f}   "
            f"{summary.retrain_recommendation_count:>7}/{first_retrain!s:<5}   "
            f"{summary.budget_limited_count:>14}"
        )
    lines.extend([""])
    if result.controller_vs_frozen_utility_delta is not None:
        lines.append(f"controller_utility_delta_vs_frozen: {result.controller_vs_frozen_utility_delta:+.3f}")
    if result.controller_vs_frozen_risk_reduction is not None:
        lines.append(f"controller_risk_reduction_vs_frozen: {result.controller_vs_frozen_risk_reduction:.1%}")
    if result.controller_vs_frozen_harmful_events_avoided is not None:
        lines.append(f"controller_harmful_events_avoided_vs_frozen: {result.controller_vs_frozen_harmful_events_avoided}")
    if result.controller_vs_frozen_retrain_deferral_steps is not None:
        lines.append(f"controller_retrain_deferral_steps_vs_frozen: {result.controller_vs_frozen_retrain_deferral_steps}")
    return "\n".join(lines)


def render_operator_replay_report(
    result: ReplayComparisonResult,
    *,
    controller_name: str | None = None,
    top_events: int = 8,
) -> str:
    timeline = _pick_timeline(result, controller_name)
    lines = [
        "Adaptive Reliability Layer Operator Replay Report",
        "",
        render_replay_report(result),
        "",
    ]
    if timeline is None:
        lines.append("No controller timeline available.")
        return "\n".join(lines)

    lines.append("Top intervention timeline")
    lines.append(
        "step   regime   profile     signature        opp   sat   rec_action -> action_taken   shift   risk   trust   retrain   budget   why"
    )
    for surface in _interesting_surfaces(timeline)[:top_events]:
        lines.append(
            f"{surface.step:>4}   "
            f"{(surface.regime_id or surface.regime_hint):<18}   "
            f"{surface.controller_profile:<10} "
            f"{surface.shift_signature:<15} "
            f"{surface.adaptation_opportunity_score:>4.2f}   "
            f"{str(surface.monitor_saturated):<5} "
            f"{surface.recommended_action:<12} -> {surface.action_taken:<12}   "
            f"{surface.shift_score:>5.2f}   {surface.risk_capital:>5.2f}   "
            f"{surface.trust_state:<8}   {str(surface.retrain_recommended):<7}   "
            f"{str(surface.budget_limited):<6}   {surface.why_this_action or surface.intervention_reason}"
        )
    lines.extend(["", "Top drift episodes"])
    for episode in top_drift_episodes(timeline):
        lines.append(
            f"- steps {episode.start_step}-{episode.end_step}: peak_shift={episode.peak_shift_score:.2f}, "
            f"peak_risk={episode.peak_risk_capital:.2f}, retrain={episode.retrain_recommended}, "
            f"actions={episode.action_summary}"
        )
    return "\n".join(lines)


def top_drift_episodes(timeline: ReplayTimeline) -> list[DriftEpisode]:
    episodes: list[DriftEpisode] = []
    active: list[DeploymentSurface] = []
    for surface in timeline.surfaces:
        eventful = surface.risk_alert or surface.retrain_recommended or surface.shift_score >= 1.0
        if eventful:
            active.append(surface)
            continue
        if active:
            episodes.append(_build_episode(active))
            active = []
    if active:
        episodes.append(_build_episode(active))
    episodes.sort(key=lambda item: (item.peak_risk_capital, item.peak_shift_score), reverse=True)
    return episodes[:5]


def _build_episode(surfaces: list[DeploymentSurface]) -> DriftEpisode:
    action_counts: dict[str, int] = {}
    for surface in surfaces:
        action_counts[surface.action_taken] = action_counts.get(surface.action_taken, 0) + 1
    action_summary = ", ".join(f"{action}:{count}" for action, count in sorted(action_counts.items()))
    return DriftEpisode(
        start_step=surfaces[0].step,
        end_step=surfaces[-1].step,
        peak_shift_score=max(surface.shift_score for surface in surfaces),
        peak_risk_capital=max(surface.risk_capital for surface in surfaces),
        action_summary=action_summary,
        retrain_recommended=any(surface.retrain_recommended for surface in surfaces),
    )


def _interesting_surfaces(timeline: ReplayTimeline) -> list[DeploymentSurface]:
    surfaces = [
        surface
        for surface in timeline.surfaces
        if (
            surface.action_taken not in {"none", "hold"}
            or surface.recommended_action not in {"none", "hold"}
            or surface.risk_alert
            or surface.retrain_recommended
            or surface.budget_limited
        )
    ]
    surfaces.sort(key=lambda surface: surface.step)
    return surfaces


def _pick_timeline(result: ReplayComparisonResult, controller_name: str | None) -> ReplayTimeline | None:
    if controller_name is not None:
        return next((timeline for timeline in result.timelines if timeline.name == controller_name), None)
    for name in ("bandit", "controller", "multi_action", "hybrid"):
        timeline = next((item for item in result.timelines if item.name == name), None)
        if timeline is not None:
            return timeline
    return None
