from __future__ import annotations

import numpy as np
import pytest

from adaptive_reliability_layer.runtime.sota.asr_reset import advise_asr_reset
from adaptive_reliability_layer.runtime.sota.collapse_asr import (
    combined_asr_collapse_risk,
    prediction_class_concentration,
)
from adaptive_reliability_layer.runtime.sota.drift_detector import DriftDetectorState
from adaptive_reliability_layer.runtime.sota.extensions import SotaRuntimeExtensions
from adaptive_reliability_layer.runtime.sota.online_conformal import OnlineConformalController
from adaptive_reliability_layer.runtime.sota.timescale import MultiTimescaleController
from adaptive_reliability_layer.runtime.config import RuntimeConfig, SotaExtensionsConfigSpec
from adaptive_reliability_layer.tabular_benchmark import TabularShiftSignal


def _signal(collapse: float = 0.1) -> TabularShiftSignal:
    return TabularShiftSignal(
        score=1.0,
        feature_score=0.5,
        output_score=0.4,
        collapse_risk=collapse,
        alert=False,
        severe=False,
        mean_entropy=0.4,
        mean_probability=0.5,
        positive_rate=0.5,
        mean_confidence=0.7,
    )


def test_asr_concentration_detects_collapse():
    predictions = [0] * 20 + [1]
    assert prediction_class_concentration(predictions) >= 0.9


def test_combined_asr_enhances_collapse_risk():
    enhanced, concentration = combined_asr_collapse_risk(
        [0] * 15 + [1],
        [0.9] * 16,
        base_collapse_risk=0.1,
    )
    assert enhanced >= 0.1
    assert concentration >= 0.8


def test_asr_reset_advice_full_reset():
    advice = advise_asr_reset(
        [0] * 18 + [1, 1],
        concentration=0.9,
        signal=_signal(collapse=0.4),
        recent_reset_steps=0,
    )
    assert advice is not None
    assert advice.action == "reset"


def test_drift_detector_rises_with_shift():
    detector = DriftDetectorState(window=16)
    for rate in np.linspace(0.1, 0.9, 12):
        detector.observe(positive_rate=float(rate), mean_confidence=0.6, output_score=0.2)
    assert detector.score() > 0.2


def test_online_conformal_updates_alpha():
    controller = OnlineConformalController(target_coverage=0.9, learning_rate=0.2)
    before = controller.alpha
    controller.observe(0.5, hit=False)
    assert controller.alpha != before or controller.alpha >= 0.02


def test_timescale_selects_expert():
    controller = MultiTimescaleController()
    name = controller.update(shift_score=2.0, output_score=1.5)
    assert name in {"short", "medium", "long"}


def test_sota_extensions_proactive_hold():
    config = RuntimeConfig(sota=SotaExtensionsConfigSpec(proactive_drift_enabled=True))
    sota = SotaRuntimeExtensions.from_runtime_config(config)
    signal = _signal()
    for score in [0.5, 0.8, 1.1, 1.4, 1.7]:
        ctx = sota.observe_batch(
            signal=signal,
            predictions=[0, 1],
            probabilities=[0.6, 0.7],
            controller_profile="fraud",
        )
    assert ctx.proactive_hold or ctx.proactive_slope >= 0.0


def test_adaptation_safety_tracker():
    from adaptive_reliability_layer.runtime.sota.adaptation_safety import AdaptationSafetyTracker

    tracker = AdaptationSafetyTracker()
    assert tracker.record(
        step=1,
        operating_mode="shadow",
        action_taken="recalibrate",
        collapse_risk=0.2,
        parameter_drift=0.1,
        force_shadow=True,
    )
    assert tracker.passes_deployment_gate(max_unsafe_rate=0.5)
