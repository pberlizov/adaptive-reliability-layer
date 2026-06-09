from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .tabular_benchmark import (
    BanditTabularPolicy,
    ControllerTabularPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    MultiActionTabularPolicy,
    NaiveTabularPolicy,
    PolicyFactory,
    SpecialistMemoryTabularPolicy,
    TabularBatch,
    TabularBenchmarkResult,
    TabularReferenceProfile,
    TabularShiftMonitor,
    _build_reference_profile,
    _evaluate_strategy,
)
from .torch_model import SourceFitSummary, TorchTabularAdapterModel


@dataclass(frozen=True)
class DigitsSourceData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    raw_test_images: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def _shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    src_x_start = max(0, -dx)
    src_x_end = image.shape[1] - max(0, dx)
    dst_x_start = max(0, dx)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    src_y_start = max(0, -dy)
    src_y_end = image.shape[0] - max(0, dy)
    dst_y_start = max(0, dy)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]
    return shifted


def _blur_image(image: np.ndarray) -> np.ndarray:
    return (
        image
        + _shift_image(image, 1, 0)
        + _shift_image(image, -1, 0)
        + _shift_image(image, 0, 1)
        + _shift_image(image, 0, -1)
    ) / 5.0


def _build_digits_source(seed: int = 7) -> DigitsSourceData:
    digits = load_digits()
    images = digits.images.astype(np.float32)
    labels = (digits.target >= 5).astype(np.int64)

    images_source_pool, images_test, y_source_pool, y_test = train_test_split(
        images,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    images_source_small, _, y_source_small, _ = train_test_split(
        images_source_pool,
        y_source_pool,
        train_size=0.22,
        random_state=seed + 1,
        stratify=y_source_pool,
    )
    images_train, images_validation, y_train, y_validation = train_test_split(
        images_source_small,
        y_source_small,
        test_size=0.28,
        random_state=seed + 2,
        stratify=y_source_small,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(images_train.reshape(len(images_train), -1)).astype(np.float32)
    x_validation = scaler.transform(images_validation.reshape(len(images_validation), -1)).astype(np.float32)

    return DigitsSourceData(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        raw_test_images=images_test,
        y_test=y_test,
        scaler=scaler,
    )


def _regime_for_step(step: int) -> str:
    if step < 15:
        return "stable"
    if step < 30:
        return "contrast_noise"
    if step < 45:
        return "label_shift"
    if step < 60:
        return "inverted_occlusion"
    if step < 75:
        return "contrast_recurrence"
    return "translated_blur"


def _sample_indices(
    rng: np.random.Generator,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    batch_size: int,
    positive_rate: float,
) -> np.ndarray:
    positive_count = int(round(batch_size * positive_rate))
    negative_count = batch_size - positive_count
    chosen_positive = rng.choice(positive_indices, size=positive_count, replace=True)
    chosen_negative = rng.choice(negative_indices, size=negative_count, replace=True)
    indices = np.concatenate([chosen_positive, chosen_negative])
    rng.shuffle(indices)
    return indices


def _transform_batch(images: np.ndarray, regime: str, step: int, rng: np.random.Generator) -> np.ndarray:
    transformed = images.copy()

    if regime == "stable":
        return transformed

    if regime == "contrast_noise":
        noise_scale = 1.2 + 0.1 * max(0, step - 15)
        transformed = 1.45 * transformed + 2.2
        transformed += rng.normal(0.0, noise_scale, size=transformed.shape)
        return np.clip(transformed, 0.0, 16.0)

    if regime == "label_shift":
        transformed = 1.10 * transformed + 0.8
        transformed += rng.normal(0.0, 0.7, size=transformed.shape)
        return np.clip(transformed, 0.0, 16.0)

    if regime == "inverted_occlusion":
        transformed = 16.0 - transformed
        transformed[:, 2:6, 2:6] *= 0.15
        transformed += rng.normal(0.0, 1.0, size=transformed.shape)
        return np.clip(transformed, 0.0, 16.0)

    if regime == "contrast_recurrence":
        transformed = 1.35 * transformed + 1.6
        transformed += rng.normal(0.0, 0.8, size=transformed.shape)
        return np.clip(transformed, 0.0, 16.0)

    shifted_images = []
    for image in transformed:
        shifted = _shift_image(image, dx=1, dy=1)
        blurred = _blur_image(shifted)
        blurred[:, :2] *= 0.25
        shifted_images.append(np.clip(blurred + rng.normal(0.0, 0.6, size=blurred.shape), 0.0, 16.0))
    return np.stack(shifted_images, axis=0).astype(np.float32)


def build_digits_stream(
    source: DigitsSourceData,
    *,
    steps: int = 90,
    batch_size: int = 48,
    seed: int = 7,
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(source.y_test == 1)
    negative_indices = np.flatnonzero(source.y_test == 0)

    batches: list[TabularBatch] = []
    for step in range(steps):
        regime = _regime_for_step(step)
        if regime == "stable":
            positive_rate = 0.50
        elif regime == "contrast_noise":
            positive_rate = 0.50
        elif regime == "label_shift":
            positive_rate = 0.82
        elif regime == "inverted_occlusion":
            positive_rate = 0.35
        elif regime == "contrast_recurrence":
            positive_rate = 0.50
        else:
            positive_rate = 0.62

        indices = _sample_indices(rng, positive_indices, negative_indices, batch_size, positive_rate)
        raw_batch = source.raw_test_images[indices]
        labels = source.y_test[indices]
        transformed = _transform_batch(raw_batch, regime, step, rng)
        flattened = transformed.reshape(len(transformed), -1)
        standardized = source.scaler.transform(flattened).astype(np.float32)
        batches.append(TabularBatch(features=standardized, labels=labels, regime=regime))
    return batches


def _build_reference_batches(
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    count: int = 10,
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_validation))
    batches: list[TabularBatch] = []
    for _ in range(count):
        chosen = rng.choice(indices, size=batch_size, replace=True)
        batches.append(TabularBatch(features=x_validation[chosen], labels=y_validation[chosen], regime="reference"))
    return batches


def _build_digits_reference_profile(
    model: TorchTabularAdapterModel,
    reference_batches: list[TabularBatch],
) -> tuple[TabularReferenceProfile, list[float]]:
    reference, _ = _build_reference_profile(model, reference_batches)
    adjusted_reference = replace(
        reference,
        feature_variance=np.maximum(reference.feature_variance, 0.25),
    )
    monitor = TabularShiftMonitor(adjusted_reference)
    reference_scores: list[float] = []
    for batch in reference_batches:
        batch_probabilities = model.predict_proba(batch.features)
        signal = monitor.evaluate(batch.features, batch_probabilities)
        reference_scores.append(signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk)
    return adjusted_reference, reference_scores


def _default_digits_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        (
            "specialist_memory",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.10),
        ),
        (
            "hybrid",
            lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.10),
        ),
    ]


def run_digits_shift_benchmark(steps: int = 90, batch_size: int = 48, seed: int = 7) -> TabularBenchmarkResult:
    source = _build_digits_source(seed=seed)
    model = TorchTabularAdapterModel(input_dim=source.x_train.shape[1], seed=seed)
    source_summary: SourceFitSummary = model.fit_source(
        source.x_train,
        source.y_train,
        source.x_validation,
        source.y_validation,
        epochs=30,
        batch_size=64,
        learning_rate=1e-3,
    )

    reference_batches = _build_reference_batches(
        source.x_validation,
        source.y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    reference, reference_scores = _build_digits_reference_profile(model, reference_batches)
    stream = build_digits_stream(source, steps=steps, batch_size=batch_size, seed=seed + 31)
    strategies = run_digits_shift_benchmark_with_factories(
        policy_factories=_default_digits_policy_factories(),
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        prepared=(
            source_summary,
            reference,
            reference_scores,
            stream,
            model,
        ),
    ).strategies
    return TabularBenchmarkResult(
        steps=steps,
        batch_size=batch_size,
        source_summary=source_summary,
        reference=reference,
        strategies=strategies,
    )

def run_digits_shift_benchmark_with_factories(
    *,
    policy_factories: list[tuple[str, PolicyFactory]],
    steps: int = 90,
    batch_size: int = 48,
    seed: int = 7,
    prepared: tuple[SourceFitSummary, TabularReferenceProfile, list[float], list[TabularBatch], TorchTabularAdapterModel]
    | None = None,
) -> TabularBenchmarkResult:
    if prepared is None:
        source = _build_digits_source(seed=seed)
        model = TorchTabularAdapterModel(input_dim=source.x_train.shape[1], seed=seed)
        source_summary = model.fit_source(
            source.x_train,
            source.y_train,
            source.x_validation,
            source.y_validation,
            epochs=30,
            batch_size=64,
            learning_rate=1e-3,
        )
        reference_batches = _build_reference_batches(
            source.x_validation,
            source.y_validation,
            batch_size=batch_size,
            seed=seed + 17,
        )
        reference, reference_scores = _build_digits_reference_profile(model, reference_batches)
        stream = build_digits_stream(source, steps=steps, batch_size=batch_size, seed=seed + 31)
    else:
        source_summary, reference, reference_scores, stream, model = prepared

    strategies = tuple(
        _evaluate_strategy(name, model.clone(), factory(reference), stream, reference, reference_scores)
        for name, factory in policy_factories
    )
    return TabularBenchmarkResult(
        steps=steps,
        batch_size=batch_size,
        source_summary=source_summary,
        reference=reference,
        strategies=strategies,
    )


def render_digits_shift_benchmark_report(result: TabularBenchmarkResult) -> str:
    frozen_accuracy = next(
        strategy.overall_accuracy for strategy in result.strategies if strategy.name == "frozen"
    )
    lines = [
        "Adaptive Reliability Layer Digits Shift Benchmark",
        (
            f"steps={result.steps} batch_size={result.batch_size} "
            f"source_val_acc={result.source_summary.best_validation_accuracy:.3f}"
        ),
        (
            "reference "
            f"entropy={result.reference.mean_entropy:.3f} "
            f"mean_probability={result.reference.mean_probability:.3f} "
            f"positive_rate={result.reference.positive_rate:.3f} "
            f"mean_confidence={result.reference.mean_confidence:.3f}"
        ),
        "",
        "strategy     accuracy   served_acc   coverage   utility   delta_vs_frozen   alerts   risk_alerts   adapts   resets   abstains   mean_shift   mean_capital   reliability   param_drift",
    ]
    for strategy in result.strategies:
        lines.append(
            f"{strategy.name:<16}"
            f"{strategy.overall_accuracy:>8.3f}"
            f"{strategy.served_accuracy:>13.3f}"
            f"{strategy.coverage:>11.3f}"
            f"{strategy.mean_utility:>10.3f}"
            f"{strategy.overall_accuracy - frozen_accuracy:>18.3f}"
            f"{strategy.alerts:>9}"
            f"{strategy.risk_alerts:>14}"
            f"{strategy.adaptations:>9}"
            f"{strategy.resets:>9}"
            f"{strategy.abstains:>11}"
            f"{strategy.mean_shift_score:>13.3f}"
            f"{strategy.mean_risk_capital:>14.3f}"
            f"{strategy.mean_reliability:>14.3f}"
            f"{strategy.mean_parameter_drift:>14.3f}"
        )
        regime_summary = ", ".join(
            f"{regime}={accuracy:.3f}" for regime, accuracy in strategy.regime_accuracy.items()
        )
        lines.append(f"  regimes: {regime_summary}")
        worst_traces = sorted(strategy.traces, key=lambda trace: trace.batch_accuracy)[:3]
        worst_summary = ", ".join(
            (
                f"step={trace.step}:{trace.regime}:acc={trace.batch_accuracy:.2f}:"
                f"action={trace.action}:capital={trace.martingale_capital:.2f}"
            )
            for trace in worst_traces
        )
        lines.append(f"  worst:   {worst_summary}")
        action_summary = ", ".join(
            f"{action}={count}" for action, count in sorted(strategy.action_counts.items())
        )
        lines.append(f"  actions: {action_summary}")
        if strategy.diagnostics:
            diagnostic_summary = ", ".join(
                f"{name}={value:.3f}" for name, value in sorted(strategy.diagnostics.items())
            )
            lines.append(f"  diag:    {diagnostic_summary}")
    return "\n".join(lines)
