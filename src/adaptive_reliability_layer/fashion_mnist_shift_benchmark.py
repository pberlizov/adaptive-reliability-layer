from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST

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
from .torch_image_model import TorchImageAdapterModel
from .torch_model import SourceFitSummary


@dataclass(frozen=True)
class FashionMnistSourceData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    raw_test_images: np.ndarray
    y_test: np.ndarray
    image_mean: float
    image_std: float


@dataclass(frozen=True)
class FashionBenchmarkConfig:
    backbone: str = "convnet"
    severity: str = "standard"
    source_train_size: int = 10000
    epochs: int = 8
    batch_size: int = 64


def _image_model_hparams(backbone: str) -> tuple[int, int, int, int]:
    if backbone == "convnet":
        return 64, 16, 8, 96
    if backbone == "resnet_small":
        return 80, 20, 6, 128
    raise ValueError(f"unsupported fashion benchmark backbone: {backbone}")


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


def _repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _build_fashion_mnist_source(seed: int = 7, *, source_train_size: int = 10000) -> FashionMnistSourceData:
    data_dir = _repo_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FashionMNIST(root=str(data_dir), train=True, download=True)
    test_dataset = FashionMNIST(root=str(data_dir), train=False, download=True)

    train_images = train_dataset.data.numpy().astype(np.float32) / 255.0
    test_images = test_dataset.data.numpy().astype(np.float32) / 255.0

    selected_classes = {0, 2, 4, 6}
    positive_classes = {4, 6}
    train_class_labels = train_dataset.targets.numpy()
    test_class_labels = test_dataset.targets.numpy()

    train_mask = np.isin(train_class_labels, list(selected_classes))
    test_mask = np.isin(test_class_labels, list(selected_classes))

    train_images = train_images[train_mask]
    test_images = test_images[test_mask]
    train_labels = np.array([1 if int(label) in positive_classes else 0 for label in train_class_labels[train_mask]])
    test_labels = np.array([1 if int(label) in positive_classes else 0 for label in test_class_labels[test_mask]])

    source_images, _, source_labels, _ = train_test_split(
        train_images,
        train_labels,
        train_size=source_train_size,
        random_state=seed,
        stratify=train_labels,
    )
    images_train, images_validation, y_train, y_validation = train_test_split(
        source_images,
        source_labels,
        test_size=0.25,
        random_state=seed + 1,
        stratify=source_labels,
    )

    image_mean = float(images_train.mean())
    image_std = float(images_train.std() + 1e-6)

    x_train = ((images_train - image_mean) / image_std).reshape(len(images_train), -1).astype(np.float32)
    x_validation = ((images_validation - image_mean) / image_std).reshape(len(images_validation), -1).astype(np.float32)

    return FashionMnistSourceData(
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_validation=x_validation,
        y_validation=y_validation.astype(np.int64),
        raw_test_images=test_images,
        y_test=test_labels.astype(np.int64),
        image_mean=image_mean,
        image_std=image_std,
    )


def _regime_for_step(step: int) -> str:
    if step < 15:
        return "stable"
    if step < 30:
        return "brightness_noise"
    if step < 45:
        return "label_shift"
    if step < 60:
        return "inverted_occlusion"
    if step < 75:
        return "brightness_recurrence"
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


def _transform_batch(
    images: np.ndarray,
    regime: str,
    step: int,
    rng: np.random.Generator,
    *,
    severity: str,
) -> np.ndarray:
    transformed = images.copy()
    severity_level = 0
    if severity == "harsh":
        severity_level = 1
    elif severity == "extreme":
        severity_level = 2
    harsh = severity_level >= 1
    extreme = severity_level >= 2

    if regime == "stable":
        return transformed

    if regime == "brightness_noise":
        if extreme:
            noise_scale = 0.20 + 0.015 * max(0, step - 15)
            transformed = 1.75 * transformed + 0.15
        else:
            noise_scale = (0.10 if not harsh else 0.16) + 0.01 * max(0, step - 15)
            transformed = (1.35 if not harsh else 1.55) * transformed + (0.08 if not harsh else 0.12)
        transformed += rng.normal(0.0, noise_scale, size=transformed.shape)
        if extreme:
            transformed[:, :4, :] *= 0.65
        return np.clip(transformed, 0.0, 1.0)

    if regime == "label_shift":
        if extreme:
            transformed = 1.25 * transformed + 0.10
            transformed += rng.normal(0.0, 0.10, size=transformed.shape)
        else:
            transformed = (1.10 if not harsh else 1.18) * transformed + (0.03 if not harsh else 0.06)
            transformed += rng.normal(0.0, 0.05 if not harsh else 0.08, size=transformed.shape)
        return np.clip(transformed, 0.0, 1.0)

    if regime == "inverted_occlusion":
        transformed = 1.0 - transformed
        if extreme:
            transformed[:, 4:24, 4:24] *= 0.04
            transformed[:, :7, :] *= 0.15
            transformed[:, :, 18:22] *= 0.25
        elif harsh:
            transformed[:, 6:22, 6:22] *= 0.10
            transformed[:, :5, :] *= 0.25
        else:
            transformed[:, 8:20, 8:20] *= 0.18
        transformed += rng.normal(0.0, 0.06 if not harsh else (0.09 if not extreme else 0.11), size=transformed.shape)
        return np.clip(transformed, 0.0, 1.0)

    if regime == "brightness_recurrence":
        if extreme:
            transformed = 1.60 * transformed + 0.12
            transformed += rng.normal(0.0, 0.10, size=transformed.shape)
        else:
            transformed = (1.28 if not harsh else 1.45) * transformed + (0.06 if not harsh else 0.09)
            transformed += rng.normal(0.0, 0.06 if not harsh else 0.08, size=transformed.shape)
        return np.clip(transformed, 0.0, 1.0)

    shifted_images = []
    for image in transformed:
        shifted = _shift_image(
            image,
            dx=2 if not harsh else (3 if not extreme else 4),
            dy=1 if not harsh else (2 if not extreme else 3),
        )
        blurred = _blur_image(shifted)
        if extreme:
            blurred[:7, :] *= 0.15
            blurred[:, :6] *= 0.30
            blurred[18:24, :] *= 0.55
        else:
            blurred[:4 if not harsh else 6, :] *= 0.25
            blurred[:, :3 if not harsh else 5] *= 0.40
        shifted_images.append(
            np.clip(
                blurred + rng.normal(0.0, 0.05 if not harsh else (0.08 if not extreme else 0.10), size=blurred.shape),
                0.0,
                1.0,
            )
        )
    return np.stack(shifted_images, axis=0).astype(np.float32)


def build_fashion_mnist_stream(
    source: FashionMnistSourceData,
    *,
    steps: int = 90,
    batch_size: int = 64,
    seed: int = 7,
    severity: str = "standard",
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(source.y_test == 1)
    negative_indices = np.flatnonzero(source.y_test == 0)
    severity_level = 0
    if severity == "harsh":
        severity_level = 1
    elif severity == "extreme":
        severity_level = 2
    harsh = severity_level >= 1
    extreme = severity_level >= 2

    batches: list[TabularBatch] = []
    for step in range(steps):
        regime = _regime_for_step(step)
        if regime == "stable":
            positive_rate = 0.50
        elif regime == "brightness_noise":
            positive_rate = 0.50
        elif regime == "label_shift":
            positive_rate = 0.82 if not harsh else (0.88 if not extreme else 0.92)
        elif regime == "inverted_occlusion":
            positive_rate = 0.34 if not harsh else (0.28 if not extreme else 0.24)
        elif regime == "brightness_recurrence":
            positive_rate = 0.50
        else:
            positive_rate = 0.64 if not harsh else (0.72 if not extreme else 0.78)

        indices = _sample_indices(rng, positive_indices, negative_indices, batch_size, positive_rate)
        raw_batch = source.raw_test_images[indices]
        labels = source.y_test[indices]
        transformed = _transform_batch(raw_batch, regime, step, rng, severity=severity)
        standardized = ((transformed - source.image_mean) / source.image_std).reshape(len(transformed), -1).astype(
            np.float32
        )
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


def _build_fashion_reference_profile(
    model: TorchImageAdapterModel,
    reference_batches: list[TabularBatch],
) -> tuple[TabularReferenceProfile, list[float]]:
    reference, _ = _build_reference_profile(model, reference_batches)
    adjusted_reference = replace(
        reference,
        feature_variance=np.maximum(reference.feature_variance, 0.10),
    )
    monitor = TabularShiftMonitor(adjusted_reference)
    reference_scores: list[float] = []
    for batch in reference_batches:
        batch_probabilities = model.predict_proba(batch.features)
        signal = monitor.evaluate(batch.features, batch_probabilities)
        reference_scores.append(signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk)
    return adjusted_reference, reference_scores


def _default_fashion_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        (
            "specialist_memory",
            lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.15),
        ),
        (
            "hybrid",
            lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.15),
        ),
    ]


def run_fashion_mnist_shift_benchmark(
    steps: int = 90,
    batch_size: int = 64,
    seed: int = 7,
    backbone: str = "convnet",
    severity: str = "standard",
) -> TabularBenchmarkResult:
    config = FashionBenchmarkConfig(backbone=backbone, severity=severity, batch_size=batch_size)
    source = _build_fashion_mnist_source(seed=seed, source_train_size=config.source_train_size)
    hidden_dim, adapter_dim, epochs, train_batch_size = _image_model_hparams(backbone)
    model = TorchImageAdapterModel(
        seed=seed,
        backbone=backbone,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
    )
    source_summary = model.fit_source(
        source.x_train,
        source.y_train,
        source.x_validation,
        source.y_validation,
        epochs=epochs,
        batch_size=train_batch_size,
        learning_rate=1e-3,
    )
    reference_batches = _build_reference_batches(
        source.x_validation,
        source.y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    reference, reference_scores = _build_fashion_reference_profile(model, reference_batches)
    stream = build_fashion_mnist_stream(
        source,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 31,
        severity=severity,
    )
    strategies = run_fashion_mnist_shift_benchmark_with_factories(
        policy_factories=_default_fashion_policy_factories(),
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        backbone=backbone,
        severity=severity,
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


def run_fashion_mnist_shift_benchmark_with_factories(
    *,
    policy_factories: list[tuple[str, PolicyFactory]],
    steps: int = 90,
    batch_size: int = 64,
    seed: int = 7,
    backbone: str = "convnet",
    severity: str = "standard",
    prepared: tuple[SourceFitSummary, TabularReferenceProfile, list[float], list[TabularBatch], TorchImageAdapterModel]
    | None = None,
) -> TabularBenchmarkResult:
    if prepared is None:
        config = FashionBenchmarkConfig(backbone=backbone, severity=severity, batch_size=batch_size)
        source = _build_fashion_mnist_source(seed=seed, source_train_size=config.source_train_size)
        hidden_dim, adapter_dim, epochs, train_batch_size = _image_model_hparams(backbone)
        model = TorchImageAdapterModel(
            seed=seed,
            backbone=backbone,
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
        )
        source_summary = model.fit_source(
            source.x_train,
            source.y_train,
            source.x_validation,
            source.y_validation,
            epochs=epochs,
            batch_size=train_batch_size,
            learning_rate=1e-3,
        )
        reference_batches = _build_reference_batches(
            source.x_validation,
            source.y_validation,
            batch_size=batch_size,
            seed=seed + 17,
        )
        reference, reference_scores = _build_fashion_reference_profile(model, reference_batches)
        stream = build_fashion_mnist_stream(
            source,
            steps=steps,
            batch_size=batch_size,
            seed=seed + 31,
            severity=severity,
        )
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


def render_fashion_mnist_shift_benchmark_report(result: TabularBenchmarkResult) -> str:
    frozen_accuracy = next(
        strategy.overall_accuracy for strategy in result.strategies if strategy.name == "frozen"
    )
    lines = [
        "Adaptive Reliability Layer Fashion-MNIST Shift Benchmark",
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
