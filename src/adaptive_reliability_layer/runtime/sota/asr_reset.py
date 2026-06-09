from __future__ import annotations

from dataclasses import dataclass

from ...tabular_benchmark import TabularDecision, TabularShiftSignal


@dataclass(frozen=True)
class ASRResetAdvice:
    action: str
    reason: str
    concentration: float
    selective: bool


def advise_asr_reset(
    predictions: list[int],
    *,
    concentration: float,
    signal: TabularShiftSignal,
    recent_reset_steps: int,
) -> ASRResetAdvice | None:
    """When/where to reset (ASR): concentration-driven, not fixed cooldown."""

    if recent_reset_steps > 0:
        return None
    high_shift = signal.score >= 1.75
    reset_concentration = 0.92 if high_shift else 0.88
    reset_collapse = 0.42 if high_shift else 0.35
    recalibrate_concentration = 0.78 if high_shift else 0.72
    recalibrate_collapse = 0.28 if high_shift else 0.22
    if concentration >= reset_concentration and signal.collapse_risk >= reset_collapse:
        return ASRResetAdvice(
            action="reset",
            reason="asr_full_reset_class_concentration",
            concentration=concentration,
            selective=False,
        )
    if concentration >= recalibrate_concentration and signal.collapse_risk >= recalibrate_collapse:
        return ASRResetAdvice(
            action="recalibrate",
            reason="asr_selective_recalibrate_concentration",
            concentration=concentration,
            selective=True,
        )
    return None
