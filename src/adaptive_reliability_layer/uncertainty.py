from __future__ import annotations

from dataclasses import dataclass

from .adaptation import AdaptationDecision
from .monitoring import ShiftSignal


@dataclass(frozen=True)
class ReliabilityOutput:
    confidence: float
    reliability_score: float
    trust_state: str


class UncertaintyWrapper:
    """Combines model confidence with shift and adaptation state."""

    def summarize(self, probabilities: list[float], signal: ShiftSignal, decision: AdaptationDecision) -> ReliabilityOutput:
        confidence = sum(max(p, 1.0 - p) for p in probabilities) / max(1, len(probabilities))
        reliability_score = max(
            0.0,
            min(
                1.0,
                1.0 - 0.18 * signal.feature_score - 0.22 * signal.output_score - 0.35 * signal.collapse_risk,
            ),
        )

        if signal.collapse_risk >= 0.5 or decision.action == "reset":
            trust_state = "escalate"
        elif signal.alert and decision.action in {"adapt", "reset"}:
            trust_state = "caution"
        elif signal.alert:
            trust_state = "monitor"
        else:
            trust_state = "normal"

        return ReliabilityOutput(
            confidence=confidence,
            reliability_score=reliability_score,
            trust_state=trust_state,
        )
