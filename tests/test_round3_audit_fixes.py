"""Regression tests for Round 3 audit fixes."""

from __future__ import annotations

import numpy as np

from adaptive_reliability_layer.runtime.feedback import compute_batch_utility
from adaptive_reliability_layer.torch_model import TorchTabularAdapterModel


def test_fit_source_tolerates_tiny_training_sets():
    rng = np.random.default_rng(7)
    x_train = rng.standard_normal((3, 8)).astype(np.float32)
    y_train = np.array([0, 1, 0], dtype=np.int64)
    x_val = rng.standard_normal((4, 8)).astype(np.float32)
    y_val = np.array([0, 1, 0, 1], dtype=np.int64)

    model = TorchTabularAdapterModel(x_train.shape[1], seed=7)
    summary = model.fit_source(x_train, y_train, x_val, y_val, epochs=2, batch_size=64)
    assert summary.best_validation_accuracy >= 0.0


def test_refresh_batch_norm_single_row_does_not_raise():
    rng = np.random.default_rng(11)
    x_train = rng.standard_normal((8, 6)).astype(np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    x_val = x_train[:4]
    y_val = y_train[:4]

    model = TorchTabularAdapterModel(x_train.shape[1], seed=11)
    model.fit_source(x_train, y_train, x_val, y_val, epochs=1)
    model.refresh_batch_norm(x_val[:1])


def test_compute_batch_utility_penalizes_abstention_when_action_is_none():
    utility_abstain = compute_batch_utility(
        batch_accuracy=0.8,
        risk_alert=False,
        parameter_drift=0.0,
        abstained=True,
        action_taken="none",
    )
    utility_served = compute_batch_utility(
        batch_accuracy=0.8,
        risk_alert=False,
        parameter_drift=0.0,
        abstained=False,
        action_taken="none",
    )
    assert utility_abstain < utility_served
    assert utility_abstain == utility_served - 0.10
