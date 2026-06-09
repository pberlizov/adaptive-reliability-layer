import numpy as np

from adaptive_reliability_layer.tabular_benchmark import (
    TabularBatch,
    TentTabularPolicy,
    run_tabular_benchmark_with_factories,
)
from adaptive_reliability_layer.tabular_benchmark import _tta_policy_factories
from adaptive_reliability_layer.torch_model import TorchTabularAdapterModel


def test_tent_adapt_runs_without_error():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(16, 8)).astype(np.float32)
    labels = (rng.random(16) > 0.5).astype(np.int64)
    x_val = rng.normal(size=(32, 8)).astype(np.float32)
    y_val = (rng.random(32) > 0.5).astype(np.int64)
    model = TorchTabularAdapterModel(8, seed=1)
    model.fit_source(features, labels, x_val, y_val, epochs=2)
    model.tent_adapt(features, steps=1)
    policy = TentTabularPolicy()
    batch = TabularBatch(features=features, labels=labels, regime="test")
    decision = policy.apply(model, None, None, batch, [0.5] * 16)
    assert decision.action == "none"


def test_tta_factories_include_tent_and_bandit():
    names = [name for name, _ in _tta_policy_factories()]
    assert "tent" in names
    assert "eata_style" in names
    assert "bandit" in names


def test_tta_tabular_comparison_smoke():
    result = run_tabular_benchmark_with_factories(
        policy_factories=_tta_policy_factories(),
        steps=6,
        batch_size=24,
        seed=3,
    )
    strategy_names = {strategy.name for strategy in result.strategies}
    assert {"frozen", "tent", "bandit"}.issubset(strategy_names)
