from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from ...tabular_benchmark import TabularShiftSignal
from .adaptation_safety import AdaptationSafetyTracker
from .asr_reset import advise_asr_reset
from .collapse_asr import combined_asr_collapse_risk
from .deferred_adaptation import DeferredAdaptationJob, DeferredAdaptationQueue
from .drift_detector import DriftDetectorState
from .online_conformal import OnlineConformalController
from .proactive_drift import ProactiveDriftMonitor
from .rccda_budget import RCCDABudgetGate
from .timescale import MultiTimescaleController


@dataclass(frozen=True)
class SotaExtensionsConfig:
    enabled: bool = True
    asr_reset_enabled: bool = True
    online_conformal_enabled: bool = True
    target_coverage: float = 0.90
    timescale_enabled: bool = True
    drift_detector_enabled: bool = True
    proactive_drift_enabled: bool = True
    rccda_budget_enabled: bool = True
    deferred_adaptation_enabled: bool = False
    adaptation_safety_enabled: bool = True
    maintenance_latent_recenter: bool = True
    max_unsafe_adaptation_rate: float = 0.15

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> SotaExtensionsConfig:
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in data if key in fields})


@dataclass
class SotaBatchContext:
    asr_concentration: float = 0.0
    drift_detector_score: float = 0.0
    timescale_expert: str = "medium"
    timescale_gain: float = 0.0
    proactive_hold: bool = False
    proactive_slope: float = 0.0
    uncertainty_action: str = "hold_threshold"
    conformal_alpha: float = 0.10
    conformal_half_width: float = 0.0
    adaptation_safety_ok: bool = True
    deferred_adaptation: bool = False
    rccda_blocked: bool = False
    rccda_reason: str | None = None


@dataclass
class SotaRuntimeExtensions:
    config: SotaExtensionsConfig = field(default_factory=SotaExtensionsConfig)
    conformal: OnlineConformalController = field(default_factory=OnlineConformalController)
    timescale: MultiTimescaleController = field(default_factory=MultiTimescaleController)
    drift_detector: DriftDetectorState = field(default_factory=DriftDetectorState)
    proactive: ProactiveDriftMonitor = field(default_factory=ProactiveDriftMonitor)
    rccda: RCCDABudgetGate = field(default_factory=RCCDABudgetGate)
    deferred: DeferredAdaptationQueue = field(default_factory=DeferredAdaptationQueue)
    safety: AdaptationSafetyTracker = field(default_factory=AdaptationSafetyTracker)
    _recent_asr_reset: int = 0

    @classmethod
    def from_runtime_config(cls, runtime_config: object) -> SotaRuntimeExtensions:
        spec = getattr(runtime_config, "sota", None)
        if spec is None:
            return cls()
        config = SotaExtensionsConfig(
            enabled=getattr(spec, "enabled", True),
            asr_reset_enabled=getattr(spec, "asr_reset_enabled", True),
            online_conformal_enabled=getattr(spec, "online_conformal_enabled", True),
            target_coverage=getattr(spec, "target_coverage", 0.90),
            timescale_enabled=getattr(spec, "timescale_enabled", True),
            drift_detector_enabled=getattr(spec, "drift_detector_enabled", True),
            proactive_drift_enabled=getattr(spec, "proactive_drift_enabled", True),
            rccda_budget_enabled=getattr(spec, "rccda_budget_enabled", True),
            deferred_adaptation_enabled=getattr(spec, "deferred_adaptation_enabled", False),
            adaptation_safety_enabled=getattr(spec, "adaptation_safety_enabled", True),
            maintenance_latent_recenter=getattr(spec, "maintenance_latent_recenter", True),
            max_unsafe_adaptation_rate=getattr(spec, "max_unsafe_adaptation_rate", 0.15),
        )
        return cls(
            config=config,
            conformal=OnlineConformalController(target_coverage=config.target_coverage),
        )

    def enrich_signal(
        self,
        signal: TabularShiftSignal,
        *,
        predictions: list[int],
        probabilities: list[float],
    ) -> TabularShiftSignal:
        if not self.config.enabled or not self.config.asr_reset_enabled:
            return signal
        enhanced, concentration = combined_asr_collapse_risk(
            predictions,
            probabilities,
            base_collapse_risk=signal.collapse_risk,
        )
        self._last_concentration = concentration
        severe = signal.severe or enhanced >= 0.30
        alert = signal.alert or enhanced >= 0.18 or signal.score >= 1.1
        return replace(
            signal,
            collapse_risk=enhanced,
            severe=severe,
            alert=alert,
            score=float(signal.feature_score + 0.75 * signal.output_score + 0.65 * enhanced),
        )

    def observe_batch(
        self,
        *,
        signal: TabularShiftSignal,
        predictions: list[int],
        probabilities: list[float],
        controller_profile: str,
        label: int | None = None,
    ) -> SotaBatchContext:
        if not self.config.enabled:
            return SotaBatchContext()

        concentration = getattr(self, "_last_concentration", 0.0)
        if self.config.asr_reset_enabled:
            _, concentration = combined_asr_collapse_risk(
                predictions,
                probabilities,
                base_collapse_risk=signal.collapse_risk,
            )

        drift_score = 0.0
        if self.config.drift_detector_enabled:
            drift_score = self.drift_detector.observe(
                positive_rate=signal.positive_rate,
                mean_confidence=signal.mean_confidence,
                output_score=signal.output_score,
            )

        timescale_expert = "medium"
        timescale_gain = 0.0
        if self.config.timescale_enabled:
            timescale_expert = self.timescale.update(
                shift_score=signal.score,
                output_score=signal.output_score,
            )
            timescale_gain = self.timescale.adaptation_gain(timescale_expert)

        proactive_hold = False
        proactive_slope = 0.0
        if self.config.proactive_drift_enabled:
            proactive_hold, proactive_slope = self.proactive.observe(signal.score)

        uncertainty_action = "hold_threshold"
        conformal_alpha = self.conformal.alpha
        conformal_half_width = 0.0
        if self.config.online_conformal_enabled:
            mean_confidence = float(sum(max(p, 1.0 - p) for p in probabilities) / max(1, len(probabilities)))
            uncertainty_action = self.conformal.issue_action(
                mean_confidence=mean_confidence,
                collapse_risk=signal.collapse_risk,
                shift_score=signal.score,
            )
            conformal_half_width = self.conformal.interval_half_width(mean_confidence)
            if label is not None and probabilities:
                score = self.conformal.nonconformity(probabilities[0], label)
                predicted = 1 if probabilities[0] >= 0.5 else 0
                self.conformal.observe(score, hit=predicted == int(label))

        if self._recent_asr_reset > 0:
            self._recent_asr_reset -= 1

        return SotaBatchContext(
            asr_concentration=concentration,
            drift_detector_score=drift_score,
            timescale_expert=timescale_expert,
            timescale_gain=timescale_gain,
            proactive_hold=proactive_hold,
            proactive_slope=proactive_slope,
            uncertainty_action=uncertainty_action,
            conformal_alpha=conformal_alpha,
            conformal_half_width=conformal_half_width,
        )

    def adjust_adaptation_opportunity(self, base: float, ctx: SotaBatchContext) -> float:
        if not self.config.enabled or base <= 0.0:
            return base
        adjusted = base
        if self.config.timescale_enabled:
            adjusted = float(min(1.0, max(0.0, 0.7 * adjusted + 0.3 * ctx.timescale_gain)))
        if ctx.proactive_hold:
            adjusted = float(min(adjusted, 0.15))
        if ctx.uncertainty_action == "tighten_abstention":
            adjusted = float(min(adjusted, 0.25))
        return adjusted

    def should_retrain(self, ctx: SotaBatchContext) -> bool:
        if not self.config.enabled or not self.config.drift_detector_enabled:
            return False
        return self.drift_detector.should_retrain() or ctx.drift_detector_score >= 0.75

    def apply_asr_override(
        self,
        *,
        recommended_action: str,
        recommended_reason: str,
        predictions: list[int],
        signal: TabularShiftSignal,
        ctx: SotaBatchContext,
    ) -> tuple[str, str]:
        if not self.config.enabled or not self.config.asr_reset_enabled:
            return recommended_action, recommended_reason
        advice = advise_asr_reset(
            predictions,
            concentration=ctx.asr_concentration,
            signal=signal,
            recent_reset_steps=self._recent_asr_reset,
        )
        if advice is None:
            return recommended_action, recommended_reason
        if advice.action == "reset":
            self._recent_asr_reset = 3
        return advice.action, f"{recommended_reason};{advice.reason}"

    def apply_conformal_override(
        self,
        *,
        recommended_action: str,
        recommended_reason: str,
        ctx: SotaBatchContext,
        collapse_risk: float,
        shift_score: float = 0.0,
    ) -> tuple[str, str]:
        if not self.config.enabled or not self.config.online_conformal_enabled:
            return recommended_action, recommended_reason
        structural_shift = shift_score >= 1.75
        light_actions = {"recalibrate", "label_shift", "hold", "none"}
        if ctx.uncertainty_action == "tighten_abstention":
            if recommended_action in {"adapt", "reset", "bn_refresh", "covariate_refresh"}:
                if structural_shift and recommended_action == "bn_refresh":
                    return recommended_action, recommended_reason
                return "hold", f"{recommended_reason};conformal_tighten_hold"
            if recommended_action == "label_shift" and structural_shift:
                return recommended_action, recommended_reason
            if recommended_action in light_actions and structural_shift:
                return recommended_action, recommended_reason
            if recommended_action in {"none", "hold"} and collapse_risk >= (0.52 if structural_shift else 0.45):
                return "abstain", f"{recommended_reason};conformal_tighten_abstain"
        if ctx.uncertainty_action == "relax_abstention" and recommended_action in {"none", "hold"}:
            collapse_floor = 0.30 if structural_shift else 0.25
            width_ceiling = 0.20 if structural_shift else 0.15
            if collapse_risk >= collapse_floor and ctx.conformal_half_width <= width_ceiling:
                return "recalibrate", f"{recommended_reason};conformal_relax_recalibrate"
        return recommended_action, recommended_reason

    def rccda_block(self, ctx: SotaBatchContext, *, signal: TabularShiftSignal) -> tuple[bool, str | None]:
        if not self.config.enabled or not self.config.rccda_budget_enabled:
            return False, None
        self.rccda.observe_proxy_loss(
            shift_score=signal.score,
            collapse_risk=signal.collapse_risk,
            miscoverage=None,
        )
        blocked, reason = self.rccda.should_block_update()
        return blocked, reason

    def maybe_defer(
        self,
        *,
        step: int,
        action: str,
        snapshot_before: object,
    ) -> bool:
        if not self.config.enabled or not self.config.deferred_adaptation_enabled:
            return False
        if action in {"none", "hold"}:
            return False
        return self.deferred.enqueue(
            DeferredAdaptationJob(step=step, action=action, snapshot_before=snapshot_before, metadata={})
        )

    def maintenance_action_override(
        self,
        *,
        controller_profile: str,
        shift_signature: str,
        recommended_action: str,
    ) -> str | None:
        if not self.config.enabled or not self.config.maintenance_latent_recenter:
            return None
        if controller_profile not in {"sensor", "sensor_safe"}:
            return None
        if shift_signature not in {"covariate_drift", "mixed_drift"}:
            return None
        if recommended_action in {"adapt", "bn_refresh"}:
            return "latent_recenter"
        return None

    def record_safety(
        self,
        *,
        step: int,
        operating_mode: str,
        action_taken: str,
        collapse_risk: float,
        parameter_drift: float,
        shift_score: float = 0.0,
    ) -> bool:
        if not self.config.enabled or not self.config.adaptation_safety_enabled:
            return True
        force_shadow = os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"}
        return self.safety.record(
            step=step,
            operating_mode=operating_mode,
            action_taken=action_taken,
            collapse_risk=collapse_risk,
            parameter_drift=parameter_drift,
            force_shadow=force_shadow,
            shift_score=shift_score,
        )

    def safety_summary(self) -> dict[str, float]:
        return self.safety.summary()

    def passes_verification(self) -> bool:
        return self.safety.passes_deployment_gate(
            max_unsafe_rate=self.config.max_unsafe_adaptation_rate,
        )
