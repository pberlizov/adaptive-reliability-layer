from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fashion_mnist_shift_benchmark import (
    FashionMnistSourceData,
    _build_fashion_mnist_source,
    _sample_indices,
    _transform_batch,
)
from .tabular_benchmark import (
    ControllerTabularPolicy,
    DelayedHybridBanditSpecialistPolicy,
    DelayedBanditTabularPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    PolicyFactory,
    RegimeAwareDelayedBanditTabularPolicy,
    RoutedDelayedBanditSpecialistPolicy,
    SpecialistMemoryTabularPolicy,
    TabularBatch,
)
from .temporal_fashion_mnist_benchmark import (
    TemporalFashionBenchmarkResult,
    TemporalRewardConfig,
    render_temporal_fashion_mnist_report,
    run_temporal_fashion_mnist_benchmark,
)


@dataclass(frozen=True)
class RecurrenceSegment:
    label: str
    transform_regime: str
    positive_rate: float
    effective_step: int


def _recurrence_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("delayed_bandit", lambda reference: DelayedBanditTabularPolicy(reference)),
        (
            "regime_aware_delayed_bandit",
            lambda reference: RegimeAwareDelayedBanditTabularPolicy(reference),
        ),
        (
            "routed_delayed_bandit",
            lambda reference: RoutedDelayedBanditSpecialistPolicy(reference, distance_threshold=0.45),
        ),
        (
            "specialist_memory",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=0.70),
        ),
        ("hybrid", lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=0.70)),
        (
            "delayed_hybrid",
            lambda reference: DelayedHybridBanditSpecialistPolicy(reference, distance_threshold=0.45),
        ),
    ]


def _recurrence_schedule(severity: str) -> tuple[RecurrenceSegment, ...]:
    if severity == "extreme":
        return (
            RecurrenceSegment("stable_intro", "stable", 0.50, 0),
            RecurrenceSegment("pattern_a_noise", "brightness_noise", 0.50, 18),
            RecurrenceSegment("pattern_b_occlusion", "inverted_occlusion", 0.26, 46),
            RecurrenceSegment("pattern_c_translation", "translated_blur", 0.76, 78),
            RecurrenceSegment("pattern_a_noise_return", "brightness_noise", 0.52, 20),
            RecurrenceSegment("pattern_b_occlusion_return", "inverted_occlusion", 0.28, 48),
        )
    return (
        RecurrenceSegment("stable_intro", "stable", 0.50, 0),
        RecurrenceSegment("pattern_a_noise", "brightness_noise", 0.50, 18),
        RecurrenceSegment("pattern_b_occlusion", "inverted_occlusion", 0.32, 46),
        RecurrenceSegment("pattern_c_translation", "translated_blur", 0.68, 78),
        RecurrenceSegment("pattern_a_noise_return", "brightness_noise", 0.52, 20),
        RecurrenceSegment("pattern_b_occlusion_return", "inverted_occlusion", 0.34, 48),
    )


def build_recurrence_fashion_mnist_stream(
    source: FashionMnistSourceData,
    *,
    steps: int = 72,
    batch_size: int = 64,
    seed: int = 7,
    severity: str = "standard",
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(source.y_test == 1)
    negative_indices = np.flatnonzero(source.y_test == 0)
    schedule = _recurrence_schedule(severity)
    segment_length = max(1, steps // len(schedule))

    batches: list[TabularBatch] = []
    for step in range(steps):
        segment = schedule[min(len(schedule) - 1, step // segment_length)]
        indices = _sample_indices(
            rng,
            positive_indices,
            negative_indices,
            batch_size,
            segment.positive_rate,
        )
        raw_batch = source.raw_test_images[indices]
        labels = source.y_test[indices]
        transformed = _transform_batch(
            raw_batch,
            segment.transform_regime,
            segment.effective_step,
            rng,
            severity=severity,
        )
        standardized = ((transformed - source.image_mean) / source.image_std).reshape(len(transformed), -1).astype(
            np.float32
        )
        batches.append(TabularBatch(features=standardized, labels=labels, regime=segment.label))
    return batches


def run_recurrence_temporal_benchmark(
    *,
    steps: int = 72,
    batch_size: int = 64,
    seed: int = 7,
    severity: str = "standard",
    reveal_delay_steps: int = 8,
    source_train_size: int = 2000,
    source_epochs: int = 2,
    reward_config: TemporalRewardConfig | None = None,
    policy_factories: list[tuple[str, PolicyFactory]] | None = None,
) -> TemporalFashionBenchmarkResult:
    source = _build_fashion_mnist_source(seed=seed, source_train_size=source_train_size)
    stream_batches = build_recurrence_fashion_mnist_stream(
        source,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 31,
        severity=severity,
    )
    factories = policy_factories if policy_factories is not None else _recurrence_policy_factories()
    return run_temporal_fashion_mnist_benchmark(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        severity=severity,
        reveal_delay_steps=reveal_delay_steps,
        source_train_size=source_train_size,
        source_epochs=source_epochs,
        reward_config=reward_config,
        stream_batches=stream_batches,
        policy_factories=factories,
    )


def recurrence_temporal_benchmark_to_dict(result: TemporalFashionBenchmarkResult) -> dict:
    return {
        "steps": result.steps,
        "batch_size": result.batch_size,
        "reveal_delay_steps": result.reveal_delay_steps,
        "source_validation_accuracy": result.source_summary.best_validation_accuracy,
        "strategies": [
            {
                "name": strategy.base.name,
                "overall_accuracy": strategy.base.overall_accuracy,
                "mean_utility": strategy.base.mean_utility,
                "mean_risk_capital": strategy.base.mean_risk_capital,
                "revealed_accuracy": strategy.revealed_accuracy,
                "revealed_coverage": strategy.revealed_coverage,
                "eventual_revealed_accuracy": strategy.eventual_revealed_accuracy,
                "mean_retro_gap": strategy.mean_retro_gap,
                "regime_accuracy": strategy.base.regime_accuracy,
                "diagnostics": strategy.base.diagnostics,
            }
            for strategy in result.strategies
        ],
    }


def render_recurrence_temporal_benchmark_report(result: TemporalFashionBenchmarkResult) -> str:
    lines = [
        "Adaptive Reliability Layer Recurrence-First Temporal Benchmark",
        (
            f"steps={result.steps} batch_size={result.batch_size} reveal_delay={result.reveal_delay_steps} "
            f"source_val_acc={result.source_summary.best_validation_accuracy:.3f}"
        ),
        "",
    ]
    lines.extend(render_temporal_fashion_mnist_report(result).splitlines()[3:])
    lines.append("")
    lines.append("specialist_diagnostics")
    for strategy in result.strategies:
        specialist_count = strategy.base.diagnostics.get("specialist_count")
        specialist_reuse = strategy.base.diagnostics.get("specialist_reuse_ratio")
        shadow_wins = strategy.base.diagnostics.get("specialist_shadow_wins")
        route_advantage = strategy.base.diagnostics.get("specialist_mean_route_advantage_ema")
        route_fallbacks = strategy.base.diagnostics.get("specialist_route_fallbacks")
        warm_starts = strategy.base.diagnostics.get("specialist_warm_starts_applied")
        reuse_confidence = strategy.base.diagnostics.get("specialist_last_reuse_confidence")
        if (
            specialist_count is None
            and specialist_reuse is None
            and shadow_wins is None
            and route_advantage is None
            and route_fallbacks is None
            and warm_starts is None
            and reuse_confidence is None
        ):
            continue
        lines.append(
            f"{strategy.base.name:<24}"
            f"specialist_count={specialist_count if specialist_count is not None else 0.0:.3f} "
            f"reuse_ratio={specialist_reuse if specialist_reuse is not None else 0.0:.3f} "
            f"shadow_wins={shadow_wins if shadow_wins is not None else 0.0:.3f} "
            f"route_adv_ema={route_advantage if route_advantage is not None else 0.0:.3f} "
            f"route_fallbacks={route_fallbacks if route_fallbacks is not None else 0.0:.3f} "
            f"warm_starts={warm_starts if warm_starts is not None else 0.0:.3f} "
            f"reuse_conf={reuse_confidence if reuse_confidence is not None else 0.0:.3f}"
        )
    return "\n".join(lines)
