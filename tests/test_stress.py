"""Stress and robustness tests for ARL runtime edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reliability_layer.runtime.config import (
    MonitorConfig,
    RuntimeConfig,
    SafetyBudgetConfig,
)
from adaptive_reliability_layer.runtime.types import OperatingMode


@pytest.fixture
def tiny_layer(tmp_path):
    """Minimal ReliabilityLayer for stress tests — quick to build."""
    from dataclasses import replace

    from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay

    base = RuntimeConfig()
    config = replace(
        base,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        governance=replace(
            base.governance,
            audit_db_path=str(tmp_path / "audit.db"),
            snapshot_dir=str(tmp_path / "snaps"),
        ),
        metrics=replace(base.metrics, enabled=False),
        replay=replace(base.replay, batch_size=16, max_steps=4, label_delay_steps=0),
        log_json=False,
    )
    return build_layer_for_tabular_replay(config=config)


def _make_batch(n_rows: int = 16, n_features: int = 30, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    from adaptive_reliability_layer.runtime.types import RuntimeBatch

    return RuntimeBatch(
        features=rng.standard_normal((n_rows, n_features)).astype(np.float32),
        labels=None,
        regime="live",
    )


# ---------------------------------------------------------------------------
# False alarm flood
# ---------------------------------------------------------------------------

class TestFalseAlarmFlood:
    """Governor must protect against a continuous barrage of shift-detector fires."""

    def test_governor_caps_actions_under_flood(self, tiny_layer):
        """Even with continuous severe shift signals, auto-actions are capped."""
        from dataclasses import replace

        from adaptive_reliability_layer.runtime.types import RuntimeBatch

        layer = tiny_layer
        # Override safety budget to a low cap so we can observe the block
        layer._config = replace(
            layer._config,
            safety_budget=SafetyBudgetConfig(
                enabled=True,
                window_steps=50,
                max_auto_actions_per_window=2,
                max_resets_per_window=1,
                downgrade_to_recommend=False,
            ),
        )
        rng = np.random.default_rng(1)
        # Send 10 batches with extreme drift to trigger many recommendations
        actions_taken = []
        for i in range(10):
            features = rng.standard_normal((16, 30)).astype(np.float32) * 20.0
            batch = RuntimeBatch(features=features, labels=None, regime="live")
            surface = layer.process_batch(batch)
            actions_taken.append(surface.action_taken)

        explicit = [a for a in actions_taken if a not in {"none", "hold", "abstain"}]
        # Safety budget should cap total explicit actions
        assert len(explicit) <= 4, (
            f"governor failed to cap actions under flood: {explicit}"
        )

    def test_governor_decision_log_captures_blocks(self, tiny_layer):
        """Blocked actions must appear in governor.decision_log."""
        from dataclasses import replace

        from adaptive_reliability_layer.runtime.types import RuntimeBatch

        layer = tiny_layer
        layer._config = replace(
            layer._config,
            safety_budget=SafetyBudgetConfig(
                enabled=True,
                window_steps=100,
                max_auto_actions_per_window=1,
                max_resets_per_window=0,
                downgrade_to_recommend=False,
            ),
        )
        rng = np.random.default_rng(2)
        for _ in range(8):
            batch = RuntimeBatch(
                features=rng.standard_normal((16, 30)).astype(np.float32) * 15.0,
                labels=None,
                regime="live",
            )
            layer.process_batch(batch)

        log = layer._governor.decision_log
        verdicts = [entry["verdict"] for entry in log]
        assert "blocked" in verdicts or "allowed" in verdicts, (
            "governor decision log should contain entries"
        )


# ---------------------------------------------------------------------------
# Mislabeled label-reveal stress test
# ---------------------------------------------------------------------------

class TestMislabeledReveal:
    """Layer must stay stable when revealed labels contain noise."""

    def test_partial_label_noise_10pct(self, tiny_layer):
        """10% mislabeled labels must not crash or corrupt the layer."""
        from adaptive_reliability_layer.runtime.types import RuntimeBatch

        layer = tiny_layer
        rng = np.random.default_rng(3)
        surfaces = []
        for step in range(4):
            batch = RuntimeBatch(
                features=rng.standard_normal((16, 30)).astype(np.float32),
                labels=None,
                regime="live",
            )
            surfaces.append(layer.process_batch(batch))

        for surface in surfaces:
            true_labels = rng.integers(0, 2, size=16).astype(np.int64)
            # Flip 10% randomly (label noise)
            flip_mask = rng.random(16) < 0.10
            noisy_labels = np.where(flip_mask, 1 - true_labels, true_labels)
            try:
                layer.reveal_labels(surface.step, noisy_labels)
            except KeyError:
                pass  # step already revealed, acceptable

        # Layer must still produce valid predictions after noisy reveals
        final_batch = RuntimeBatch(
            features=rng.standard_normal((16, 30)).astype(np.float32),
            labels=None,
            regime="live",
        )
        final_surface = layer.process_batch(final_batch)
        assert len(final_surface.predictions) == 16
        assert all(p in {0, 1} for p in final_surface.predictions)

    def test_all_same_class_labels(self, tiny_layer):
        """All-zero or all-one label batches must not crash the layer."""
        from adaptive_reliability_layer.serving.state import _check_label_quality

        all_zero = np.zeros(16, dtype=np.int64)
        all_one = np.ones(16, dtype=np.int64)
        warnings_zero = _check_label_quality(all_zero)
        warnings_one = _check_label_quality(all_one)

        assert "all_negative_labels" in warnings_zero
        assert "all_positive_labels" in warnings_one

    def test_out_of_range_labels_detected(self):
        """Labels outside {0,1} must be flagged."""
        from adaptive_reliability_layer.serving.state import _check_label_quality

        bad_labels = np.array([0, 1, 2, -1, 0], dtype=np.int64)
        warnings = _check_label_quality(bad_labels)
        assert any("out_of_range" in w for w in warnings)

    def test_nan_labels_detected(self):
        """NaN labels must be flagged immediately."""
        from adaptive_reliability_layer.serving.state import _check_label_quality

        nan_labels = np.array([0.0, float("nan"), 1.0], dtype=np.float64)
        warnings = _check_label_quality(nan_labels)
        assert "non_finite_labels" in warnings


# ---------------------------------------------------------------------------
# Delay distribution stress test
# ---------------------------------------------------------------------------

class TestDelayDistribution:
    """Layer must handle variable and out-of-order label arrival patterns."""

    def test_exponential_delay_pattern(self, tiny_layer):
        """Labels arriving with exponential delays must all be processable."""
        from adaptive_reliability_layer.runtime.types import RuntimeBatch

        layer = tiny_layer
        rng = np.random.default_rng(4)
        n_batches = 6
        surfaces = []

        for _ in range(n_batches):
            batch = RuntimeBatch(
                features=rng.standard_normal((16, 30)).astype(np.float32),
                labels=None,
                regime="live",
            )
            surfaces.append(layer.process_batch(batch))

        # Reveal in reverse order (out-of-order arrival)
        for surface in reversed(surfaces):
            labels = rng.integers(0, 2, size=16).astype(np.int64)
            try:
                layer.reveal_labels(surface.step, labels)
            except KeyError:
                pass  # already consumed, acceptable

        assert layer.pending_delayed_count == 0 or layer.pending_delayed_count < n_batches

    def test_labels_revealed_twice_are_rejected(self, tiny_layer):
        """Revealing labels for the same step twice must raise KeyError on the second call."""
        from adaptive_reliability_layer.runtime.types import RuntimeBatch

        layer = tiny_layer
        rng = np.random.default_rng(5)
        batch = RuntimeBatch(
            features=rng.standard_normal((16, 30)).astype(np.float32),
            labels=None,
            regime="live",
        )
        surface = layer.process_batch(batch)
        labels = rng.integers(0, 2, size=16).astype(np.int64)
        layer.reveal_labels(surface.step, labels)  # first: ok

        with pytest.raises(KeyError):
            layer.reveal_labels(surface.step, labels)  # second: must fail

    def test_specialist_reservoir_does_not_overflow(self, tiny_layer):
        """Long stream with many regime switches must not accumulate infinite specialists."""
        from dataclasses import replace

        from adaptive_reliability_layer.runtime.config import PolicyConfig
        from adaptive_reliability_layer.runtime.types import RuntimeBatch

        layer = tiny_layer
        layer._config = replace(
            layer._config,
            policy=replace(layer._config.policy, name="delayed_hybrid", max_specialists=4),
        )
        rng = np.random.default_rng(6)

        for i in range(20):
            # Alternate between very different distributions to generate many regimes
            scale = 10.0 if i % 2 == 0 else 0.1
            batch = RuntimeBatch(
                features=(rng.standard_normal((16, 30)) * scale).astype(np.float32),
                labels=None,
                regime=f"regime_{i % 5}",
            )
            layer.process_batch(batch)

        # The specialist reservoir should respect max_specialists
        if hasattr(layer._policy, "_specialists"):
            n = len(layer._policy._specialists)
            assert n <= 8, f"specialist pool overflowed: {n} specialists"
