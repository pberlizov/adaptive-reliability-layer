from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any

import numpy as np

from ..risk import MartingaleRiskMonitor, RiskState
from ..tabular_benchmark import (
    TabularBatch,
    TabularDecision,
    TabularReferenceProfile,
    TabularShiftMonitor,
    TabularShiftSignal,
)
from .action_gating import (
    apply_operating_mode,
    build_runtime_policy,
    decision_from_tabular,
    reliability_score,
    runtime_batch_to_tabular,
    trust_state_from_signal,
)
from .audit import AuditStore, GovernanceService, SnapshotStore
from .config import RuntimeConfig
from .correction import DelayedCorrectionEngine
from .governor import InterventionGovernor, _load_snapshot
from .logging_config import log_event
from .metrics import RuntimeMetrics
from .model_adapter import ModelAdapter, TorchTabularModelAdapter
from .policy_state import load_policy_state_from_file, save_policy_state
from .policy_state_store import build_policy_state_store
from .sota import SotaRuntimeExtensions
from .types import (
    DeploymentSurface,
    InterventionDecision,
    OperatingMode,
    PROFILE_BOUNDED_ACTIONS,
    RuntimeBatch,
)


@dataclass(frozen=True)
class PendingApprovalState:
    step: int
    batch: TabularBatch
    signal: TabularShiftSignal
    risk_state: RiskState
    pre_probabilities: list[float]
    recommended: InterventionDecision
    snapshot_before: object
    snapshot_id_before: str | None
    metadata: dict[str, Any]


class ReliabilityLayer:
    """Production runtime orchestrator for monitor + controller + governance."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        reference: TabularReferenceProfile,
        reference_scores: list[float],
        config: RuntimeConfig | None = None,
        governance: GovernanceService | None = None,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self._adapter = adapter
        self._reference = reference
        self._config = config or RuntimeConfig()
        self._monitor = TabularShiftMonitor(
            reference,
            alert_threshold=self._config.monitor.alert_threshold,
            severe_threshold=self._config.monitor.severe_threshold,
        )
        self._risk_monitor = MartingaleRiskMonitor(
            reference_scores,
            alert_threshold=self._config.monitor.risk_alert_threshold,
            decay=self._config.monitor.risk_decay,
        )
        self._policy = build_runtime_policy(
            self._config.policy.name,
            reference,
            self._config.policy,
        )
        self._governance = governance or GovernanceService(
            SnapshotStore(
                self._config.governance.snapshot_dir,
                max_snapshots=self._config.governance.max_snapshots,
            ),
            AuditStore(self._config.governance.audit_db_path),
        )
        self._metrics = metrics or RuntimeMetrics(
            namespace=self._config.metrics.namespace,
            enabled=self._config.metrics.enabled,
        )
        self._logger = logging.getLogger("adaptive_reliability_layer.runtime")
        self._step = 0
        self._specialist_id: str | None = None
        self._pending_approval: PendingApprovalState | None = None
        self._correction = DelayedCorrectionEngine(self._config)
        self._governor = InterventionGovernor(self._config)
        self._policy_store = build_policy_state_store(self._config)
        if self._policy_store is not None:
            self._policy_store.load(self._policy)
        elif self._config.policy_state_path:
            load_policy_state_from_file(self._policy, self._config.policy_state_path)
        self._sota = SotaRuntimeExtensions.from_runtime_config(self._config)

    @property
    def governance(self) -> GovernanceService:
        return self._governance

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def revealed_metrics(self) -> tuple[dict[str, float], ...]:
        return self._correction.revealed_metrics

    def save_policy_state(self, path: str | None = None) -> str:
        if path is not None:
            return str(save_policy_state(self._policy, path))
        if self._policy_store is not None:
            self._policy_store.save(self._policy)
            return self._config.policy_state_save_path or self._config.policy_state_redis_key or "policy_store"
        target = self._config.policy_state_save_path
        if not target:
            raise ValueError("policy_state_save_path is not configured")
        return str(save_policy_state(self._policy, target))

    def reveal_labels(
        self,
        step: int,
        labels: np.ndarray | list[int],
        *,
        frozen_baseline_accuracy: float | None = None,
        batch_id: str | None = None,
    ) -> dict[str, float]:
        """Apply delayed feedback for a prior batch once labels are available."""

        metrics = self._correction.reveal(
            step,
            labels,
            frozen_baseline_accuracy=frozen_baseline_accuracy,
            batch_id=batch_id,
            reveal_step=self._step,
            policy=self._policy,
            policy_model=self._policy_model(),
        )
        if self._policy_store is not None:
            self._policy_store.save(self._policy)
        elif self._config.policy_state_save_path:
            save_policy_state(self._policy, self._config.policy_state_save_path)
        return metrics

    def reveal_labels_by_batch_id(
        self,
        batch_id: str,
        labels: np.ndarray | list[int],
        *,
        frozen_baseline_accuracy: float | None = None,
    ) -> dict[str, float]:
        return self.reveal_labels(
            step=-1,
            labels=labels,
            frozen_baseline_accuracy=frozen_baseline_accuracy,
            batch_id=batch_id,
        )

    def is_ready(self) -> bool:
        return self._adapter is not None and self._reference is not None

    @property
    def pending_delayed_count(self) -> int:
        return self._correction.pending_count

    def process_batch(
        self,
        batch: RuntimeBatch | TabularBatch,
        *,
        operating_mode_override: OperatingMode | None = None,
        bounded_auto_actions_override: frozenset[str] | None = None,
    ) -> DeploymentSurface:
        tabular_batch = runtime_batch_to_tabular(batch)
        batch_id = self._extract_batch_id(batch)
        features = np.asarray(tabular_batch.features, dtype=np.float32)
        labels = tabular_batch.labels
        label_delay = self._config.replay.label_delay_steps
        immediate_labels = None
        if labels is not None and label_delay <= 0:
            immediate_labels = np.asarray(labels, dtype=np.int64)
        elif labels is not None and label_delay > 0:
            tabular_batch = TabularBatch(
                features=tabular_batch.features,
                labels=None,
                regime=tabular_batch.regime,
            )

        operating_mode = operating_mode_override or self._config.operating_mode
        controller_profile = self._controller_profile(batch)
        self._pending_approval = None
        self._correction.publish_summary(self._policy, self._step)
        self._prepare_policy_model(tabular_batch)

        pre_probabilities = self._adapter.predict_proba(features)
        predictions_pre = self._adapter.predict(features)
        signal = self._monitor.evaluate(features, pre_probabilities)
        signal = self._sota.enrich_signal(signal, predictions=predictions_pre, probabilities=pre_probabilities)
        shift_signature = self._shift_signature(signal, controller_profile=controller_profile)
        monitor_saturated = self._monitor_saturated(signal, controller_profile=controller_profile)
        adaptation_opportunity_score = self._adaptation_opportunity_score(
            signal,
            controller_profile=controller_profile,
            shift_signature=shift_signature,
            monitor_saturated=monitor_saturated,
        )
        sota_ctx = self._sota.observe_batch(
            signal=signal,
            predictions=predictions_pre,
            probabilities=pre_probabilities,
            controller_profile=controller_profile,
            label=int(immediate_labels[0]) if immediate_labels is not None and len(immediate_labels) else None,
        )
        adaptation_opportunity_score = self._sota.adjust_adaptation_opportunity(
            adaptation_opportunity_score,
            sota_ctx,
        )
        bounded_actions = self._resolve_bounded_actions(
            controller_profile=controller_profile,
            shift_signature=shift_signature,
            monitor_saturated=monitor_saturated,
            adaptation_opportunity_score=adaptation_opportunity_score,
            bounded_auto_actions_override=bounded_auto_actions_override,
        )
        raw_risk_score = signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk
        risk_state = self._risk_monitor.update(raw_risk_score)

        snapshot_before = self._adapter.export_snapshot()
        snapshot_id_before = self._maybe_save_snapshot(
            operating_mode=operating_mode,
            reason="pre_intervention",
            force=False,
        )

        policy_model = self._policy_model()
        tabular_decision = self._policy.apply(
            policy_model,
            signal,
            risk_state,
            tabular_batch,
            pre_probabilities,
        )
        original_recommended = decision_from_tabular(tabular_decision)
        recommended = self._profile_adjust_decision(
            recommended=original_recommended,
            controller_profile=controller_profile,
            shift_signature=shift_signature,
            adaptation_opportunity_score=adaptation_opportunity_score,
            monitor_saturated=monitor_saturated,
        )
        maintenance_action = self._sota.maintenance_action_override(
            controller_profile=controller_profile,
            shift_signature=shift_signature,
            recommended_action=recommended.action,
        )
        if maintenance_action is not None:
            recommended = InterventionDecision(
                action=maintenance_action,
                reason=f"{recommended.reason};maintenance_latent_recenter",
                selected_fraction=recommended.selected_fraction,
            )
        recommended_action, recommended_reason = self._sota.apply_asr_override(
            recommended_action=recommended.action,
            recommended_reason=recommended.reason,
            predictions=predictions_pre,
            signal=signal,
            ctx=sota_ctx,
        )
        if recommended_action != recommended.action:
            recommended = InterventionDecision(
                action=recommended_action,
                reason=recommended_reason,
                selected_fraction=recommended.selected_fraction,
            )
        conformal_action, conformal_reason = self._sota.apply_conformal_override(
            recommended_action=recommended.action,
            recommended_reason=recommended.reason,
            ctx=sota_ctx,
            collapse_risk=signal.collapse_risk,
            shift_score=signal.score,
        )
        if conformal_action != recommended.action:
            recommended = InterventionDecision(
                action=conformal_action,
                reason=conformal_reason,
                selected_fraction=recommended.selected_fraction,
            )
        if sota_ctx.proactive_hold and recommended.action not in {"none", "hold", "abstain"}:
            recommended = InterventionDecision(
                action="hold",
                reason=f"{recommended.reason};proactive_drift_hold",
                selected_fraction=0.0,
            )
        if recommended.action != original_recommended.action:
            self._adapter.load_snapshot(snapshot_before)  # type: ignore[arg-type]
            remapped_action, remap_reason, remap_fraction = self._apply_explicit_action(
                approved_action=recommended.action,
                batch=tabular_batch,
                signal=signal,
                risk_state=risk_state,
                pre_probabilities=pre_probabilities,
            )
            recommended = InterventionDecision(
                action=remapped_action,
                reason=f"{recommended.reason};{remap_reason}",
                selected_fraction=remap_fraction,
            )
        policy_diagnostics = self._policy.get_diagnostics() if hasattr(self._policy, "get_diagnostics") else {}
        policy_diagnostics = dict(policy_diagnostics)

        action_taken, reason_suffix = apply_operating_mode(
            mode=operating_mode,
            bounded_auto_actions=bounded_actions,
            adapter=self._adapter,
            decision=recommended,
            snapshot_before=snapshot_before,
        )
        effective_operating_mode = operating_mode
        budget_limited = False
        budget_reason: str | None = None
        if operating_mode == OperatingMode.BOUNDED_AUTO:
            (
                action_taken,
                reason_suffix,
                effective_operating_mode,
                budget_limited,
                budget_reason,
            ) = self._governor.apply_safety_budget(
                recommended=recommended,
                action_taken=action_taken,
                reason_suffix=reason_suffix,
                snapshot_before=snapshot_before,
                adapter=self._adapter,
                step=self._step,
            )
        rccda_blocked, rccda_reason = self._sota.rccda_block(sota_ctx, signal=signal)
        if (
            rccda_blocked
            and self._config.safety_budget.rccda_loss_slope_block
            and action_taken not in {"none", "hold"}
        ):
            _load_snapshot(self._adapter, snapshot_before)
            action_taken = "none"
            reason_suffix = f"{reason_suffix};rccda_blocked:{rccda_reason}"
            budget_limited = True
            budget_reason = rccda_reason
        deferred = self._sota.maybe_defer(
            step=self._step,
            action=action_taken,
            snapshot_before=snapshot_before,
        )
        if action_taken == "reset":
            self._risk_monitor.reset()
            risk_state = RiskState(
                raw_score=risk_state.raw_score,
                p_value=risk_state.p_value,
                e_value=risk_state.e_value,
                capital=1.0,
                alert=False,
            )
        elif self._config.policy.name != "frozen" and action_taken in {"hold", "none"}:
            mitigated_escalation = recommended.action not in {"none", "hold", "abstain"}
            if mitigated_escalation and not signal.severe:
                risk_state = self._risk_monitor.apply_mitigation(decay_factor=0.55)
            elif action_taken == "hold" and sota_ctx.proactive_hold:
                risk_state = self._risk_monitor.apply_mitigation(decay_factor=0.4)

        snapshot_id_after: str | None = None
        if action_taken not in {"none", "hold"}:
            snapshot_id_after = self._maybe_save_snapshot(
                operating_mode=operating_mode,
                reason=f"post_{action_taken}",
                force=True,
            )

        raw_post_probabilities = self._adapter.predict_proba(features)
        probabilities = self._postprocess_probabilities(
            raw_post_probabilities,
            signal=signal,
            risk_state=risk_state,
            batch=tabular_batch,
        )
        decision_threshold = self._decision_threshold(
            signal=signal,
            risk_state=risk_state,
            batch=tabular_batch,
        )
        correction_mean_abs_delta, correction_max_abs_delta, correction_flipped_predictions, correction_applied = self._correction_stats(
            raw_post_probabilities,
            probabilities,
        )
        correction_applied = correction_applied or abs(decision_threshold - 0.5) > 1e-6
        predictions = [1 if probability >= decision_threshold else 0 for probability in probabilities]
        confidence = float(np.mean([max(p, 1.0 - p) for p in probabilities]))
        trust = trust_state_from_signal(signal, recommended)
        reliability = reliability_score(signal, risk_state, recommended)
        regime_id, regime_confidence, regime_novelty = self._regime_context(
            tabular_batch=tabular_batch,
            batch=batch,
            policy_diagnostics=policy_diagnostics,
        )
        retrain_recommended = self._governor.should_retrain(
            signal=signal,
            risk_state=risk_state,
            recommended=recommended,
            action_taken=action_taken,
            budget_limited=budget_limited,
            policy_name=self._config.policy.name,
            step=self._step,
        ) or self._sota.should_retrain(sota_ctx)
        adaptation_safety_ok = self._sota.record_safety(
            step=self._step,
            operating_mode=effective_operating_mode.value,
            action_taken=action_taken,
            collapse_risk=signal.collapse_risk,
            parameter_drift=self._adapter.parameter_drift(),
            shift_score=signal.score,
        )
        requires_approval = (
            effective_operating_mode == OperatingMode.RECOMMEND
            and recommended.action not in {"none", "hold"}
        )
        batch_accuracy = None
        if immediate_labels is not None:
            batch_accuracy = float((np.array(predictions) == immediate_labels).mean())

        surface = DeploymentSurface(
            step=self._step,
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            shift_score=signal.score,
            feature_shift_score=signal.feature_score,
            output_shift_score=signal.output_score,
            collapse_risk=signal.collapse_risk,
            risk_capital=risk_state.capital,
            risk_alert=risk_state.alert,
            regime_hint=tabular_batch.regime,
            recommended_action=recommended.action,
            action_taken=action_taken,
            intervention_reason=f"{recommended.reason};{reason_suffix}",
            why_this_action=f"{recommended.reason};{reason_suffix}",
            trust_state=trust,
            reliability_score=reliability,
            parameter_drift=self._adapter.parameter_drift(),
            operating_mode=operating_mode.value,
            effective_operating_mode=effective_operating_mode.value,
            model_version=self._adapter.model_version,
            specialist_id=self._specialist_id,
            rollback_available=snapshot_id_before is not None,
            rollback_eligible=snapshot_id_before is not None,
            snapshot_id=snapshot_id_after or snapshot_id_before,
            abstained=action_taken == "abstain" or recommended.action == "abstain",
            regime_id=regime_id,
            regime_confidence=regime_confidence,
            regime_novelty=regime_novelty,
            risk_score=raw_risk_score,
            batch_accuracy=batch_accuracy,
            recommended_action_requires_approval=requires_approval,
            retrain_recommended=retrain_recommended,
            budget_limited=budget_limited,
            budget_reason=budget_reason,
            batch_id=batch_id,
            shift_signature=shift_signature,
            controller_profile=controller_profile,
            adaptation_opportunity_score=adaptation_opportunity_score,
            monitor_saturated=monitor_saturated,
            asr_class_concentration=sota_ctx.asr_concentration,
            drift_detector_score=sota_ctx.drift_detector_score,
            timescale_expert=sota_ctx.timescale_expert,
            uncertainty_action=sota_ctx.uncertainty_action,
            conformal_alpha=sota_ctx.conformal_alpha,
            conformal_half_width=sota_ctx.conformal_half_width,
            adaptation_safety_ok=adaptation_safety_ok,
            proactive_hold=sota_ctx.proactive_hold,
            deferred_adaptation=deferred,
            correction_mean_abs_delta=correction_mean_abs_delta,
            correction_max_abs_delta=correction_max_abs_delta,
            correction_flipped_predictions=correction_flipped_predictions,
            correction_applied=correction_applied,
            explicit_action_executed=action_taken not in {"none", "hold"},
            decision_threshold=decision_threshold,
            threshold_shift=decision_threshold - 0.5,
        )
        if effective_operating_mode == OperatingMode.RECOMMEND and recommended.action not in {"none", "hold"}:
            self._pending_approval = PendingApprovalState(
                step=surface.step,
                batch=tabular_batch,
                signal=signal,
                risk_state=risk_state,
                pre_probabilities=list(pre_probabilities),
                recommended=recommended,
                snapshot_before=snapshot_before,
                snapshot_id_before=snapshot_id_before,
                metadata={
                    "regime": tabular_batch.regime,
                    "regime_id": regime_id,
                    "regime_confidence": regime_confidence,
                    "regime_novelty": regime_novelty,
                    "batch_metadata": getattr(batch, "metadata", {}),
                    "controller_profile": controller_profile,
                    "shift_signature": shift_signature,
                    "adaptation_opportunity_score": adaptation_opportunity_score,
                    "monitor_saturated": monitor_saturated,
                },
            )

        self._governance.record_intervention(
            step=self._step,
            operating_mode=effective_operating_mode.value,
            model_version=self._adapter.model_version,
            recommended_action=recommended.action,
            action_taken=action_taken,
            intervention_reason=surface.intervention_reason,
            shift_score=signal.score,
            risk_capital=risk_state.capital,
            risk_alert=risk_state.alert,
            trust_state=trust,
            snapshot_id_before=snapshot_id_before,
            snapshot_id_after=snapshot_id_after,
            metadata={
                "regime": tabular_batch.regime,
                "regime_id": regime_id,
                "regime_confidence": regime_confidence,
                "regime_novelty": regime_novelty,
                "batch_metadata": getattr(batch, "metadata", {}),
                "policy_version": self._config.governance.policy_version,
                "environment": self._config.governance.environment,
                "effective_operating_mode": effective_operating_mode.value,
                "risk_score": raw_risk_score,
                "retrain_recommended": retrain_recommended,
                "budget_limited": budget_limited,
                "budget_reason": budget_reason,
                "decision_record": surface.decision_record(),
                "controller_profile": controller_profile,
                "shift_signature": shift_signature,
                "adaptation_opportunity_score": adaptation_opportunity_score,
                "monitor_saturated": monitor_saturated,
                "sota": {
                    "asr_concentration": sota_ctx.asr_concentration,
                    "drift_detector_score": sota_ctx.drift_detector_score,
                    "timescale_expert": sota_ctx.timescale_expert,
                    "uncertainty_action": sota_ctx.uncertainty_action,
                    "adaptation_safety_ok": adaptation_safety_ok,
                },
            },
        )

        if self._config.metrics.enabled:
            self._metrics.observe_batch(
                shift_score=signal.score,
                risk_capital=risk_state.capital,
                batch_accuracy=batch_accuracy,
                recommended_action=recommended.action,
                action_taken=action_taken,
                operating_mode=effective_operating_mode.value,
                risk_alert=risk_state.alert,
            )

        if self._config.log_json:
            log_event(
                self._logger,
                "batch_processed",
                step=self._step,
                recommended_action=recommended.action,
                action_taken=action_taken,
                shift_score=signal.score,
                risk_capital=risk_state.capital,
                trust_state=trust,
                retrain_recommended=retrain_recommended,
                regime_id=regime_id,
                controller_profile=controller_profile,
                shift_signature=shift_signature,
                adaptation_opportunity_score=adaptation_opportunity_score,
                monitor_saturated=monitor_saturated,
                effective_operating_mode=effective_operating_mode.value,
            )

        self._governor.record_action(action_taken, self._step)
        self._correction.enqueue(
            step=surface.step,
            batch_id=batch_id,
            batch=tabular_batch,
            signal=signal,
            risk_state=risk_state,
            decision=tabular_decision,
            recommended=recommended,
            action_taken=action_taken,
            predictions=predictions,
            probabilities=probabilities,
            parameter_drift=surface.parameter_drift,
            abstained=surface.abstained,
            policy=self._policy,
            policy_model=self._policy_model(),
        )
        if immediate_labels is not None:
            if batch_id is not None:
                self.reveal_labels(surface.step, immediate_labels, batch_id=batch_id)
            else:
                self.reveal_labels(surface.step, immediate_labels)
        self._step += 1
        return surface

    def _postprocess_probabilities(
        self,
        probabilities: list[float],
        *,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
    ) -> list[float]:
        if not hasattr(self._policy, "correct_probabilities"):
            return probabilities
        corrected = self._policy.correct_probabilities(  # type: ignore[attr-defined]
            probabilities,
            signal=signal,
            risk_state=risk_state,
            batch=batch,
        )
        return [float(np.clip(probability, 1e-5, 1.0 - 1e-5)) for probability in corrected]

    def _decision_threshold(
        self,
        *,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        batch: TabularBatch,
    ) -> float:
        if not hasattr(self._policy, "decision_threshold"):
            return 0.5
        threshold = self._policy.decision_threshold(  # type: ignore[attr-defined]
            signal=signal,
            risk_state=risk_state,
            batch=batch,
        )
        return float(np.clip(threshold, 0.05, 0.95))

    def _correction_stats(
        self,
        raw_probabilities: list[float],
        corrected_probabilities: list[float],
    ) -> tuple[float, float, int, bool]:
        if len(raw_probabilities) != len(corrected_probabilities) or not raw_probabilities:
            return 0.0, 0.0, 0, False
        raw = np.asarray(raw_probabilities, dtype=np.float64)
        corrected = np.asarray(corrected_probabilities, dtype=np.float64)
        delta = np.abs(corrected - raw)
        raw_predictions = raw >= 0.5
        corrected_predictions = corrected >= 0.5
        flips = int(np.sum(raw_predictions != corrected_predictions))
        max_delta = float(delta.max(initial=0.0))
        mean_delta = float(delta.mean())
        applied = bool(max_delta > 1e-6)
        return mean_delta, max_delta, flips, applied

    def _prepare_policy_model(self, tabular_batch: TabularBatch) -> None:
        if hasattr(self._policy, "prepare_model"):
            revealed_rate = self._correction.recent_revealed_positive_rate
            if revealed_rate is not None and hasattr(self._policy, "set_revealed_positive_rate"):
                self._policy.set_revealed_positive_rate(revealed_rate)  # type: ignore[attr-defined]
            self._policy.prepare_model(self._policy_model(), tabular_batch)  # type: ignore[attr-defined]

    @staticmethod
    def _extract_batch_id(batch: RuntimeBatch | TabularBatch) -> str | None:
        metadata = getattr(batch, "metadata", None) or {}
        batch_id = metadata.get("batch_id")
        return str(batch_id) if batch_id is not None else None

    def approve_and_apply(
        self,
        batch: RuntimeBatch | TabularBatch,
        *,
        approved_action: str,
        approver: str,
    ) -> DeploymentSurface:
        """Recommend-mode helper: human approves a previously recommended action."""

        if self._config.operating_mode != OperatingMode.RECOMMEND:
            raise RuntimeError("approve_and_apply is only valid in recommend operating mode")
        if self._pending_approval is None:
            raise RuntimeError("no pending recommendation is available for approval")

        pending = self._pending_approval
        tabular_batch = runtime_batch_to_tabular(batch)
        if not self._same_batch(tabular_batch, pending.batch):
            raise ValueError("approved batch does not match the pending recommendation")

        self._adapter.load_snapshot(pending.snapshot_before)  # type: ignore[arg-type]
        snapshot_id_before = pending.snapshot_id_before
        if approved_action not in {"none", "hold", "abstain"}:
            snapshot_id_before = self._governance.snapshots.save(
                self._adapter,
                reason="pre_approved_action",
                step=pending.step,
            )
        action_taken, approval_reason, selected_fraction = self._apply_explicit_action(
            approved_action=approved_action,
            batch=pending.batch,
            signal=pending.signal,
            risk_state=pending.risk_state,
            pre_probabilities=pending.pre_probabilities,
            human_approved=True,
        )

        post_risk_state = pending.risk_state
        if action_taken == "reset":
            self._risk_monitor.reset()
            post_risk_state = RiskState(
                raw_score=pending.risk_state.raw_score,
                p_value=pending.risk_state.p_value,
                e_value=pending.risk_state.e_value,
                capital=1.0,
                alert=False,
            )

        snapshot_id_after: str | None = None
        if action_taken not in {"none", "hold"}:
            snapshot_id_after = self._governance.snapshots.save(
                self._adapter,
                reason=f"approved_{action_taken}",
                step=pending.step,
            )

        features = np.asarray(pending.batch.features, dtype=np.float32)
        raw_post_probabilities = self._adapter.predict_proba(features)
        probabilities = self._postprocess_probabilities(
            raw_post_probabilities,
            signal=pending.signal,
            risk_state=post_risk_state,
            batch=pending.batch,
        )
        decision_threshold = self._decision_threshold(
            signal=pending.signal,
            risk_state=post_risk_state,
            batch=pending.batch,
        )
        correction_mean_abs_delta, correction_max_abs_delta, correction_flipped_predictions, correction_applied = self._correction_stats(
            raw_post_probabilities,
            probabilities,
        )
        correction_applied = correction_applied or abs(decision_threshold - 0.5) > 1e-6
        predictions = [1 if probability >= decision_threshold else 0 for probability in probabilities]
        confidence = float(np.mean([max(p, 1.0 - p) for p in probabilities]))
        approved_decision = InterventionDecision(
            action=action_taken,
            reason=approval_reason,
            selected_fraction=selected_fraction,
        )
        trust = trust_state_from_signal(pending.signal, approved_decision)
        reliability = reliability_score(pending.signal, post_risk_state, approved_decision)
        regime_id = str(pending.metadata.get("regime_id", pending.batch.regime))
        regime_confidence = float(pending.metadata.get("regime_confidence", 0.0))
        regime_novelty = float(pending.metadata.get("regime_novelty", 0.0))
        risk_score = pending.risk_state.raw_score
        controller_profile = str(pending.metadata.get("controller_profile", "general"))
        shift_signature = str(pending.metadata.get("shift_signature", self._shift_signature(pending.signal, controller_profile=controller_profile)))
        adaptation_opportunity_score = float(pending.metadata.get("adaptation_opportunity_score", 0.0))
        monitor_saturated = bool(pending.metadata.get("monitor_saturated", False))
        retrain_recommended = self._governor.should_retrain(
            signal=pending.signal,
            risk_state=post_risk_state,
            recommended=pending.recommended,
            action_taken=action_taken,
            budget_limited=False,
            monitor_saturated=monitor_saturated,
            policy_name=self._config.policy.name,
            step=self._step,
        )
        batch_accuracy = None
        if pending.batch.labels is not None:
            batch_accuracy = float((np.array(predictions) == np.asarray(pending.batch.labels)).mean())

        surface = DeploymentSurface(
            step=pending.step,
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            shift_score=pending.signal.score,
            feature_shift_score=pending.signal.feature_score,
            output_shift_score=pending.signal.output_score,
            collapse_risk=pending.signal.collapse_risk,
            risk_capital=post_risk_state.capital,
            risk_alert=post_risk_state.alert,
            regime_hint=pending.batch.regime,
            recommended_action=pending.recommended.action,
            action_taken=action_taken,
            intervention_reason=f"human_approved:{approved_action};{approval_reason}",
            why_this_action=f"human_approved:{approved_action};{approval_reason}",
            trust_state=trust,
            reliability_score=reliability,
            parameter_drift=self._adapter.parameter_drift(),
            operating_mode=OperatingMode.RECOMMEND.value,
            effective_operating_mode=OperatingMode.RECOMMEND.value,
            model_version=self._adapter.model_version,
            specialist_id=self._specialist_id,
            rollback_available=snapshot_id_before is not None,
            rollback_eligible=snapshot_id_before is not None,
            snapshot_id=snapshot_id_after or snapshot_id_before,
            abstained=action_taken == "abstain",
            regime_id=regime_id,
            regime_confidence=regime_confidence,
            regime_novelty=regime_novelty,
            risk_score=risk_score,
            recommended_action_requires_approval=False,
            retrain_recommended=retrain_recommended,
            shift_signature=shift_signature,
            controller_profile=controller_profile,
            adaptation_opportunity_score=adaptation_opportunity_score,
            monitor_saturated=monitor_saturated,
            correction_mean_abs_delta=correction_mean_abs_delta,
            correction_max_abs_delta=correction_max_abs_delta,
            correction_flipped_predictions=correction_flipped_predictions,
            correction_applied=correction_applied,
            explicit_action_executed=action_taken not in {"none", "hold"},
            decision_threshold=decision_threshold,
            threshold_shift=decision_threshold - 0.5,
        )

        self._governance.record_intervention(
            step=surface.step,
            operating_mode="recommend",
            model_version=self._adapter.model_version,
            recommended_action=pending.recommended.action,
            action_taken=surface.action_taken,
            intervention_reason=surface.intervention_reason,
            shift_score=surface.shift_score,
            risk_capital=surface.risk_capital,
            risk_alert=surface.risk_alert,
            trust_state=surface.trust_state,
            snapshot_id_before=snapshot_id_before,
            snapshot_id_after=snapshot_id_after,
            approved_by=approver,
            metadata={
                "approved_action": approved_action,
                "recommended_action": pending.recommended.action,
                "recommended_reason": pending.recommended.reason,
                "regime_id": regime_id,
                "regime_confidence": regime_confidence,
                "regime_novelty": regime_novelty,
                "risk_score": risk_score,
                "retrain_recommended": retrain_recommended,
                "decision_record": surface.decision_record(),
                "controller_profile": controller_profile,
                "shift_signature": shift_signature,
                "adaptation_opportunity_score": adaptation_opportunity_score,
                "monitor_saturated": monitor_saturated,
                **pending.metadata,
            },
        )
        self._governor.record_action(action_taken, self._step)
        self._pending_approval = None
        return surface

    def rollback(self, snapshot_id: str, *, actor: str = "operator") -> None:
        self._governance.rollback(self._adapter, snapshot_id, step=self._step, actor=actor)

    def set_operating_mode(self, mode: OperatingMode) -> dict[str, object]:
        """Switch operating mode at runtime without restart."""

        import os
        from dataclasses import replace

        force_shadow = os.environ.get("ARL_FORCE_SHADOW", "").lower() in {"1", "true", "yes"}
        requested = mode
        effective = OperatingMode.SHADOW if force_shadow else mode
        self._config = replace(self._config, operating_mode=effective)
        return {
            "requested_mode": requested.value,
            "operating_mode": effective.value,
            "force_shadow_active": force_shadow,
        }

    def pending_recommendation(self) -> dict[str, object] | None:
        pending = self._pending_approval
        if pending is None:
            return None
        return {
            "step": pending.step,
            "recommended_action": pending.recommended.action,
            "reason": pending.recommended.reason,
            "regime_id": pending.metadata.get("regime_id", pending.batch.regime),
            "controller_profile": pending.metadata.get("controller_profile", "general"),
            "shift_signature": pending.metadata.get("shift_signature", self._shift_signature(pending.signal, controller_profile=str(pending.metadata.get("controller_profile", "general")))),
            "adaptation_opportunity_score": pending.metadata.get("adaptation_opportunity_score", 0.0),
            "monitor_saturated": pending.metadata.get("monitor_saturated", False),
        }

    def export_audit_jsonl(self, path: str) -> None:
        self._governance.audit.export_jsonl(path)

    def _regime_context(
        self,
        *,
        tabular_batch: TabularBatch,
        batch: RuntimeBatch | TabularBatch,
        policy_diagnostics: dict[str, Any],
    ) -> tuple[str, float, float]:
        batch_metadata = getattr(batch, "metadata", {}) or {}
        regime_id = str(batch_metadata.get("regime_id", tabular_batch.regime))
        regime_confidence = float(
            policy_diagnostics.get(
                "specialist_last_regime_confidence",
                policy_diagnostics.get("regime_recurrence_confidence", 0.0),
            )
        )
        regime_novelty = float(
            policy_diagnostics.get(
                "specialist_last_regime_novelty",
                policy_diagnostics.get("regime_novelty_score", 0.0),
            )
        )
        return regime_id, regime_confidence, regime_novelty

    def _controller_profile(self, batch: RuntimeBatch | TabularBatch) -> str:
        metadata = getattr(batch, "metadata", None) or {}
        explicit = metadata.get("controller_profile")
        if explicit is not None:
            return str(explicit)
        wedge = metadata.get("wedge")
        if wedge == "fraud_risk":
            return "fraud"
        if wedge == "predictive_maintenance":
            return "sensor"
        return "general"

    def _shift_signature(self, signal: TabularShiftSignal, *, controller_profile: str = "general") -> str:
        if controller_profile in {"sensor", "sensor_safe"}:
            return self._maintenance_shift_signature(signal)
        if self._monitor_saturated(signal, controller_profile=controller_profile):
            return "collapse_risk"
        if signal.collapse_risk >= 0.30 or (signal.severe and signal.output_score >= 0.22):
            return "collapse_risk"
        if not signal.alert and signal.feature_score < 0.9 and signal.output_score < 0.14:
            return "stable"
        if signal.feature_score >= max(1.15, 1.45 * signal.output_score) and signal.collapse_risk < 0.22:
            return "covariate_drift"
        if signal.output_score >= max(0.18, 1.2 * signal.feature_score):
            return "label_drift"
        if signal.alert or signal.severe:
            return "mixed_drift"
        return "stable"

    def _maintenance_shift_signature(self, signal: TabularShiftSignal) -> str:
        feature_dominant = signal.feature_score >= max(1.10, 1.60 * signal.output_score)
        extreme_feature_mismatch = (
            signal.feature_score >= 4.0
            or (signal.score >= 6.0 and signal.feature_score >= 2.5)
            or (signal.severe and signal.feature_score >= 2.0 and signal.output_score <= 0.30)
        )
        if extreme_feature_mismatch and signal.collapse_risk < 0.30:
            return "reference_break"
        if signal.collapse_risk >= 0.30 or (signal.severe and signal.output_score >= 0.22):
            return "collapse_risk"
        if not signal.alert and signal.feature_score < 0.9 and signal.output_score < 0.14:
            return "stable"
        if feature_dominant and signal.collapse_risk < 0.22:
            return "covariate_drift"
        if signal.output_score >= max(0.18, 1.3 * signal.feature_score):
            return "mixed_drift"
        if signal.alert or signal.severe:
            return "mixed_drift"
        return "stable"

    def _monitor_saturated(self, signal: TabularShiftSignal, *, controller_profile: str) -> bool:
        if controller_profile not in {"sensor", "sensor_safe"}:
            return False
        return bool(
            signal.score >= max(8.0, self._config.monitor.severe_threshold * 4.0)
            or signal.feature_score >= 6.0
            or (signal.severe and signal.feature_score >= 3.0 and signal.output_score >= 0.6)
        )

    def _adaptation_opportunity_score(
        self,
        signal: TabularShiftSignal,
        *,
        controller_profile: str,
        shift_signature: str,
        monitor_saturated: bool,
    ) -> float:
        if monitor_saturated:
            return 0.0
        if controller_profile == "sensor_safe":
            if shift_signature == "covariate_drift":
                return max(
                    0.0,
                    min(1.0, 0.35 * signal.feature_score - 0.45 * signal.output_score - 0.55 * signal.collapse_risk),
                )
            if shift_signature == "mixed_drift":
                return max(0.0, min(1.0, 0.10 * signal.feature_score - 0.35 * signal.output_score))
            return max(0.0, min(1.0, 0.08 * signal.output_score))
        if controller_profile == "sensor":
            if shift_signature == "covariate_drift":
                return max(0.0, min(1.0, 0.55 * signal.feature_score - 0.35 * signal.output_score - 0.45 * signal.collapse_risk))
            if shift_signature == "mixed_drift":
                return max(0.0, min(1.0, 0.25 * signal.feature_score - 0.30 * signal.output_score))
            return max(0.0, min(1.0, 0.15 * signal.output_score))
        if controller_profile == "fraud":
            if shift_signature == "label_drift":
                return max(0.0, min(1.0, 0.75 * signal.output_score + 0.25 * signal.collapse_risk))
            return max(0.0, min(1.0, 0.25 * signal.feature_score + 0.20 * signal.output_score))
        return max(0.0, min(1.0, 0.30 * signal.feature_score + 0.30 * signal.output_score))

    def _resolve_bounded_actions(
        self,
        *,
        controller_profile: str,
        shift_signature: str,
        monitor_saturated: bool,
        adaptation_opportunity_score: float,
        bounded_auto_actions_override: frozenset[str] | None,
    ) -> frozenset[str]:
        base_actions = bounded_auto_actions_override or self._config.bounded_auto_actions
        if controller_profile == "sensor_safe":
            if monitor_saturated or adaptation_opportunity_score < 0.35:
                return frozenset(action for action in base_actions if action in {"none", "hold"})
        if controller_profile == "sensor" and (monitor_saturated or adaptation_opportunity_score < 0.20):
            return frozenset(action for action in base_actions if action in {"none", "hold"})
        # Combined accuracy + positive-rate gate: block adaptation when BOTH
        # revealed accuracy is high AND the positive rate hasn't drifted from
        # the reference.  This isolates benign operating-condition switches
        # (high acc, stable positive rate) from genuine degradation (acc drops
        # OR positive rate rises).  Either condition failing → gate opens.
        revealed_acc = self._revealed_accuracy_signal()
        revealed_pos = self._correction.recent_revealed_positive_rate
        if (
            revealed_acc is not None
            and revealed_pos is not None
            and revealed_acc > 0.92
            # Directional: only open the gate when positive rate has RISEN above
            # reference + margin.  Falling below reference (e.g. early test with
            # 0% failures vs 12-15% training rate) is not genuine degradation.
            and revealed_pos <= self._reference.mean_probability + 0.05
        ):
            return frozenset(action for action in base_actions if action in {"none", "hold"})
        profile_actions = PROFILE_BOUNDED_ACTIONS.get(controller_profile, PROFILE_BOUNDED_ACTIONS["general"])
        signature_actions = profile_actions.get(shift_signature, profile_actions["stable"])
        return frozenset(action for action in base_actions if action in signature_actions)

    def _profile_adjust_decision(
        self,
        *,
        recommended: InterventionDecision,
        controller_profile: str,
        shift_signature: str,
        adaptation_opportunity_score: float,
        monitor_saturated: bool,
    ) -> InterventionDecision:
        if (
            controller_profile == "sensor"
            and shift_signature == "covariate_drift"
            and not monitor_saturated
            and adaptation_opportunity_score >= 0.35
            and recommended.action in {"bn_refresh", "adapt"}
        ):
            return InterventionDecision(
                action="covariate_refresh",
                reason=f"{recommended.reason};profile_covariate_refresh",
                selected_fraction=recommended.selected_fraction,
            )
        return recommended

    def _policy_model(self) -> Any:
        if isinstance(self._adapter, TorchTabularModelAdapter):
            return self._adapter.inner
        return self._adapter

    def _same_batch(self, lhs: TabularBatch, rhs: TabularBatch) -> bool:
        if lhs.regime != rhs.regime:
            return False
        if lhs.features.shape != rhs.features.shape:
            return False
        if not np.array_equal(lhs.features, rhs.features):
            return False
        if lhs.labels is None and rhs.labels is None:
            return True
        if lhs.labels is None or rhs.labels is None:
            return False
        return bool(np.array_equal(lhs.labels, rhs.labels))

    def _apply_explicit_action(
        self,
        *,
        approved_action: str,
        batch: TabularBatch,
        signal: TabularShiftSignal,
        risk_state: RiskState,
        pre_probabilities: list[float],
        human_approved: bool = False,
    ) -> tuple[str, str, float]:
        features = np.asarray(batch.features, dtype=np.float32)
        if approved_action in {"none", "hold", "abstain"}:
            return approved_action, "human_selected_no_mutation", 0.0
        if approved_action == "bn_refresh":
            self._adapter.refresh_batch_norm(features, passes=1)
            return approved_action, "human_approved_bn_refresh", 0.0
        if approved_action == "bn_refresh_only":
            # BN statistics update only — no temperature recalibration, no bias change.
            # Decoupled from covariate_refresh for ablation studies.
            self._adapter.refresh_batch_norm(features, passes=2)
            return approved_action, "bn_refresh_only_stats_update", 0.0
        if approved_action == "covariate_refresh":
            self._adapter.apply_covariate_refresh(
                features=features,
                reference_confidence=self._reference.mean_confidence,
                observed_confidence=signal.mean_confidence,
                intensity=2 if signal.severe or risk_state.alert else 1,
            )
            return approved_action, "human_approved_covariate_refresh", 0.0
        if approved_action == "recalibrate":
            # Guard (auto only): only recalibrate when revealed accuracy confirms degradation.
            # Skipped for human-approved actions — human override takes precedence.
            if not human_approved:
                recent = list(self._correction.revealed_metrics)[-5:]
                if recent:
                    recent_accuracy = float(np.mean([m.get("batch_accuracy", 1.0) for m in recent]))
                    if recent_accuracy > 0.85:
                        return "hold", "recalibrate_accuracy_not_degraded", 0.0
            self._adapter.recalibrate_temperature(
                reference_confidence=self._reference.mean_confidence,
                observed_confidence=signal.mean_confidence,
                momentum=0.25,
            )
            return approved_action, "recalibration", 0.0
        if approved_action == "cool_confidence":
            # Overconfidence-specific recalibration: fires when observed_confidence
            # exceeds the reference.  No accuracy-degradation guard — overconfident
            # models can still have high accuracy, yet produce poorly-calibrated
            # probabilities that mislead downstream decision logic.
            # Auto guard: only fire when the model is genuinely overconfident.
            overconfidence_gap = signal.mean_confidence - self._reference.mean_confidence
            if not human_approved and overconfidence_gap < 0.04:
                return "hold", "cool_confidence_not_overconfident", 0.0
            self._adapter.recalibrate_temperature(
                reference_confidence=self._reference.mean_confidence,
                observed_confidence=signal.mean_confidence,
                momentum=0.15,  # gentler than recalibrate (0.25) — confidence can bounce
            )
            return approved_action, "overconfidence_temperature_cooling", 0.0
        if approved_action == "label_shift":
            # Guard (auto only): require meaningful positive-rate deviation.
            # Also directional: don't apply if rate has only fallen (early test
            # period) — that would over-correct in the wrong direction.
            if not human_approved:
                revealed_rate = self._revealed_positive_rate()
                if abs(revealed_rate - self._reference.mean_probability) < 0.07:
                    return "hold", "label_shift_below_threshold", 0.0
                if revealed_rate < self._reference.mean_probability:
                    return "hold", "label_shift_rate_fallen_below_ref", 0.0
            self._adapter.apply_label_shift_correction(
                source_positive_rate=self._reference.mean_probability,
                target_positive_rate=signal.positive_rate,
                momentum=0.35,
            )
            return approved_action, "label_shift_correction", 0.0
        if approved_action == "bbse_label_shift":
            if not human_approved:
                revealed_positive_rate = self._revealed_positive_rate()
                if revealed_positive_rate < self._reference.mean_probability + 0.07:
                    return "hold", "bbse_label_shift_rate_not_risen", 0.0
            revealed_positive_rate = self._revealed_positive_rate()
            self._adapter.apply_label_shift_correction(
                source_positive_rate=self._reference.mean_probability,
                target_positive_rate=revealed_positive_rate,
                momentum=0.45,
            )
            return approved_action, "bbse_label_shift_from_revealed_labels", 0.0
        if approved_action == "adapt":
            selected_fraction = self._adapter.adapt(
                features,
                pre_probabilities,
                learning_rate=0.025,
                confidence_threshold=0.90 if signal.severe or risk_state.alert else 0.80,
                anchor_strength=0.16,
                entropy_weight=0.08,
                max_parameter_drift=0.70,
                steps=2,
            )
            if selected_fraction == 0.0:
                return "hold", "human_approved_adapt_no_confident_samples", 0.0
            return approved_action, "human_approved_adapt", selected_fraction
        if approved_action == "reset":
            self._adapter.reset()
            return approved_action, "human_approved_reset", 0.0
        if approved_action == "latent_recenter":
            self._adapter.apply_latent_recenter(
                features=features,
                momentum=0.12,
            )
            return approved_action, "human_approved_latent_recenter", 0.0
        return "none", f"unsupported_approved_action:{approved_action}", 0.0

    def _revealed_accuracy_signal(self) -> float | None:
        """Mean batch accuracy from recent revealed labels, or None if < 2 batches revealed.

        Used to distinguish benign operating-condition switches (accuracy stays
        high) from genuine model degradation (accuracy drops).  Returns None when
        insufficient data, so callers fall through to default behaviour.
        """
        recent = list(self._correction.revealed_metrics)[-5:]
        if len(recent) < 2:
            return None
        return float(np.mean([m.get("batch_accuracy", 1.0) for m in recent]))

    def _revealed_positive_rate(self) -> float:
        """Return the mean positive rate from recent revealed label batches.

        Falls back to the source (training) positive rate when fewer than 2
        batches have been revealed so far.
        """
        rate = self._correction.recent_revealed_positive_rate
        if rate is None:
            return self._reference.mean_probability
        return rate

    def _maybe_save_snapshot(
        self,
        *,
        operating_mode: OperatingMode,
        reason: str,
        force: bool,
    ) -> str | None:
        governance = self._config.governance
        if operating_mode == OperatingMode.SHADOW and not governance.persist_snapshots_in_shadow:
            return self._governance.snapshots.latest_snapshot_id()
        if operating_mode == OperatingMode.RECOMMEND and not governance.persist_snapshots_on_recommend:
            return self._governance.snapshots.latest_snapshot_id()
        if not force and not governance.persist_snapshots_on_mutation:
            return self._governance.snapshots.latest_snapshot_id()
        return self._governance.snapshots.save(self._adapter, reason=reason, step=self._step)


def build_reliability_layer_from_reference_batches(
    adapter: ModelAdapter,
    reference_batches: list[TabularBatch],
    *,
    config: RuntimeConfig | None = None,
) -> ReliabilityLayer:
    from .reference import build_reference_profile_from_adapter

    profile, reference_scores = build_reference_profile_from_adapter(adapter, reference_batches)
    return ReliabilityLayer(
        adapter=adapter,
        reference=profile,
        reference_scores=reference_scores,
        config=config,
    )
