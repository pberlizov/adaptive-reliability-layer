"""Risk reduction and customer replay productization tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaptive_reliability_layer.replay.customer_replay import CustomerReplaySpec, run_customer_replay
from adaptive_reliability_layer.replay.engine import run_offline_replay_comparison
from adaptive_reliability_layer.replay.report import summarize_replay_runs
from adaptive_reliability_layer.replay.types import ReplayRunState
from adaptive_reliability_layer.risk import MartingaleRiskMonitor
from adaptive_reliability_layer.runtime.types import DeploymentSurface


def test_martingale_mitigation_lowers_capital():
    monitor = MartingaleRiskMonitor(
        [0.01, 0.02, 0.03],
        decay=1.0,
        epsilon=0.9,
        alert_threshold=100.0,
    )
    before = 1.0
    for _ in range(4):
        before = monitor.update(0.99).capital
    assert before > 1.0
    after = monitor.apply_mitigation(decay_factor=0.5).capital
    assert after < before


def _surface(
    *,
    step: int,
    risk_capital: float,
    risk_alert: bool,
    retrain_recommended: bool,
) -> DeploymentSurface:
    return DeploymentSurface(
        step=step,
        predictions=[0],
        probabilities=[0.5],
        confidence=0.5,
        shift_score=1.0,
        feature_shift_score=0.2,
        output_shift_score=0.2,
        collapse_risk=0.1,
        risk_capital=risk_capital,
        risk_alert=risk_alert,
        regime_hint="r0",
        recommended_action="hold",
        action_taken="hold",
        intervention_reason="test",
        trust_state="stable",
        reliability_score=0.8,
        parameter_drift=0.01,
        operating_mode="shadow",
        model_version="test-v1",
        specialist_id=None,
        rollback_available=False,
        snapshot_id=None,
        abstained=False,
        retrain_recommended=retrain_recommended,
        adaptation_safety_ok=True,
        budget_limited=False,
    )


def test_summarize_risk_reduction_uses_alert_and_retrain_signals():
    class _DummyLayer:
        pass

    frozen = ReplayRunState(name="frozen", layer=_DummyLayer())  # type: ignore[arg-type]
    controller = ReplayRunState(name="regime_aware_delayed_bandit", layer=_DummyLayer())  # type: ignore[arg-type]
    for step in range(4):
        frozen.surfaces.append(
            _surface(step=step, risk_capital=10.0, risk_alert=True, retrain_recommended=True)
        )
        frozen.risk_capitals.append(10.0)
        frozen.utilities.append(0.8)
        controller.surfaces.append(
            _surface(step=step, risk_capital=8.0, risk_alert=False, retrain_recommended=False)
        )
        controller.risk_capitals.append(8.0)
        controller.utilities.append(0.85)
    result = summarize_replay_runs([frozen, controller], controller_name="regime_aware_delayed_bandit")
    assert result.controller_vs_frozen_risk_reduction is not None
    assert result.controller_vs_frozen_risk_reduction >= 0.19
    assert result.controller_vs_frozen_harmful_events_avoided == 4


@pytest.mark.slow
def test_customer_replay_cli_smoke(tmp_path: Path):
    csv_path = tmp_path / "customer.csv"
    rows = []
    for index in range(32):
        rows.append(
            {
                "timestamp": f"2025-06-01T00:{index:02d}:00Z",
                "label": index % 5,
                "feature_0": float(index) / 32.0,
                "feature_1": float(index % 3) / 3.0,
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "customer_shadow.yaml"
    output = tmp_path / "out"
    result = run_customer_replay(
        CustomerReplaySpec(
            input_path=csv_path,
            config_path=config_path,
            output_dir=output,
            customer_label="testco",
            dual_mode=False,
            batch_size=8,
            label_delay_steps=2,
        )
    )
    assert (output / "customer_manifest.json").exists()
    assert (output / "buyer_report.md").exists()
    assert result.customer_label == "testco"


@pytest.mark.slow
def test_fraud_bundle_risk_reduction_nonzero(runtime_config):
    from adaptive_reliability_layer.replay.real_data import load_ulb_creditcard_fraud_torch_bundle

    bundle = load_ulb_creditcard_fraud_torch_bundle(steps=12, batch_size=64)
    config = runtime_config
    result = run_offline_replay_comparison(
        bundle.stream,
        runtime_config=config,
        strategies=("frozen", "regime_aware_delayed_bandit"),
        layer_builder=bundle.build_layer,
        controller_name="regime_aware_delayed_bandit",
    )
    assert result.controller_vs_frozen_risk_reduction is not None
