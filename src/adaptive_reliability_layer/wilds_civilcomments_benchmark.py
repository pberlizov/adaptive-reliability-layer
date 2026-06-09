from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .tabular_benchmark import (
    PolicyFactory,
    TabularBatch,
    TabularBenchmarkResult,
    TabularReferenceProfile,
    _build_reference_batches,
    _build_reference_profile,
    _default_policy_factories,
    run_tabular_benchmark_with_factories,
)
from .benchmark_suite import AggregateStat
from .torch_model import SourceFitSummary, TorchTabularAdapterModel


@dataclass(frozen=True)
class WildsCivilCommentsConfig:
    root_dir: str = "data/wilds"
    download: bool = False
    seed: int = 7
    train_limit: int = 6000
    validation_limit: int = 2000
    test_limit: int = 9000
    batch_size: int = 64
    steps: int = 72
    max_tfidf_features: int = 5000
    svd_dim: int = 96
    min_group_count: int = 96
    suite_name: str = "custom"


@dataclass(frozen=True)
class WildsGroupSummary:
    role: str
    group_id: int
    group_name: str
    count: int
    source_accuracy: float


@dataclass(frozen=True)
class WildsCivilCommentsBenchmarkResult:
    config: WildsCivilCommentsConfig
    source_summary: SourceFitSummary
    reference: TabularReferenceProfile
    selected_groups: tuple[WildsGroupSummary, ...]
    benchmark: TabularBenchmarkResult


@dataclass(frozen=True)
class WildsStrategyAggregate:
    name: str
    metrics: dict[str, AggregateStat]
    regime_accuracy: dict[str, AggregateStat]
    diagnostics: dict[str, AggregateStat]


@dataclass(frozen=True)
class WildsBenchmarkAggregate:
    name: str
    seeds: tuple[int, ...]
    configs: tuple[WildsCivilCommentsConfig, ...]
    strategies: tuple[WildsStrategyAggregate, ...]


@dataclass(frozen=True)
class WildsStrategySummary:
    name: str
    utility_wins: int
    accuracy_wins: int
    mean_utility_margin_vs_frozen: float
    mean_accuracy_margin_vs_frozen: float


@dataclass(frozen=True)
class WildsCivilCommentsSuiteResult:
    seeds: tuple[int, ...]
    benchmarks: tuple[WildsBenchmarkAggregate, ...]
    summary: tuple[WildsStrategySummary, ...]


def _require_wilds() -> Any:
    try:
        from wilds import get_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by import-time smoke only
        raise ImportError(
            "WILDS support requires the `wilds` package. Install it with `pip install wilds`."
        ) from exc
    return get_dataset


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_subset(
    subset: Any,
    *,
    limit: int,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    available = np.arange(len(subset))
    chosen = available if limit >= len(available) else rng.choice(available, size=limit, replace=False)

    texts: list[str] = []
    labels: list[int] = []
    metadata_rows: list[np.ndarray] = []
    for index in chosen:
        item = subset[int(index)]
        if len(item) == 3:
            text, label, metadata = item
        elif len(item) == 2:
            text, label = item
            metadata = np.zeros(1, dtype=np.int64)
        else:  # pragma: no cover - defensive against future WILDS API changes
            raise ValueError(f"unexpected WILDS item structure of length {len(item)}")
        texts.append(text if isinstance(text, str) else str(text))
        label_value = int(_to_numpy(label).reshape(-1)[0])
        labels.append(label_value)
        metadata_rows.append(_to_numpy(metadata).reshape(-1))

    return texts, np.asarray(labels, dtype=np.int64), np.stack(metadata_rows, axis=0)


def _group_ids(dataset: Any, metadata: np.ndarray) -> np.ndarray:
    identity_columns = {
        "male": 0,
        "female": 1,
        "LGBTQ": 2,
        "christian": 3,
        "muslim": 4,
        "other_religions": 5,
        "black": 6,
        "white": 7,
    }
    active_identities = np.stack([metadata[:, index] for index in identity_columns.values()], axis=1)
    has_identity = active_identities.max(axis=1) > 0
    primary_identity = np.argmax(active_identities, axis=1).astype(np.int64)
    none_group = len(identity_columns)
    return np.where(has_identity, primary_identity, none_group).astype(np.int64)


def _group_name_map() -> dict[int, str]:
    return {
        0: "male",
        1: "female",
        2: "LGBTQ",
        3: "christian",
        4: "muslim",
        5: "other_religions",
        6: "black",
        7: "white",
        8: "none",
    }


def _vectorize_text(
    train_texts: list[str],
    validation_texts: list[str],
    test_texts: list[str],
    *,
    max_features: int,
    svd_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
    )
    x_train_sparse = vectorizer.fit_transform(train_texts)
    x_validation_sparse = vectorizer.transform(validation_texts)
    x_test_sparse = vectorizer.transform(test_texts)

    effective_dim = max(8, min(svd_dim, x_train_sparse.shape[1] - 1))
    svd = TruncatedSVD(n_components=effective_dim, random_state=13)
    x_train = svd.fit_transform(x_train_sparse).astype(np.float32)
    x_validation = svd.transform(x_validation_sparse).astype(np.float32)
    x_test = svd.transform(x_test_sparse).astype(np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return x_train, x_validation, x_test


def _group_statistics(
    model: TorchTabularAdapterModel,
    x_test: np.ndarray,
    y_test: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_group_count: int,
) -> list[WildsGroupSummary]:
    group_names = _group_name_map()
    probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float32)
    predictions = (probabilities >= 0.5).astype(np.int64)
    summaries: list[WildsGroupSummary] = []
    for group_id in sorted(np.unique(group_ids)):
        mask = group_ids == group_id
        count = int(mask.sum())
        if count < min_group_count:
            continue
        accuracy = float((predictions[mask] == y_test[mask]).mean())
        summaries.append(
            WildsGroupSummary(
                role="candidate",
                group_id=int(group_id),
                group_name=group_names.get(int(group_id), f"group_{int(group_id)}"),
                count=count,
                source_accuracy=accuracy,
            )
        )
    summaries.sort(key=lambda item: item.count, reverse=True)
    return summaries


def _selected_group_roles(candidates: list[WildsGroupSummary]) -> tuple[WildsGroupSummary, ...]:
    if len(candidates) < 3:
        raise ValueError("need at least three sufficiently large WILDS groups to build the benchmark stream")

    max_count = max(candidate.count for candidate in candidates)
    min_accuracy = min(candidate.source_accuracy for candidate in candidates)
    max_accuracy = max(candidate.source_accuracy for candidate in candidates)
    spread = max(1e-6, max_accuracy - min_accuracy)
    midpoint = 0.5 * (min_accuracy + max_accuracy)

    def count_score(candidate: WildsGroupSummary) -> float:
        return candidate.count / max_count

    def mid_accuracy_score(candidate: WildsGroupSummary) -> float:
        return 1.0 - abs(candidate.source_accuracy - midpoint) / spread

    easy = max(
        candidates,
        key=lambda candidate: (
            candidate.source_accuracy,
            count_score(candidate),
            candidate.count,
        ),
    )
    remaining = [candidate for candidate in candidates if candidate.group_id != easy.group_id]
    hard = min(
        remaining,
        key=lambda candidate: (
            candidate.source_accuracy,
            -count_score(candidate),
            -candidate.count,
        ),
    )
    recurring_candidates = [candidate for candidate in remaining if candidate.group_id != hard.group_id]
    recurring = max(
        recurring_candidates,
        key=lambda candidate: (
            0.65 * count_score(candidate) + 0.35 * mid_accuracy_score(candidate),
            candidate.count,
        ),
    )
    chosen = {
        "easy": easy,
        "hard": hard,
        "recurring": recurring,
    }
    return tuple(
        WildsGroupSummary(
            role=role,
            group_id=summary.group_id,
            group_name=summary.group_name,
            count=summary.count,
            source_accuracy=summary.source_accuracy,
        )
        for role, summary in chosen.items()
    )


def _sample_group_batch(
    rng: np.random.Generator,
    pool: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    return rng.choice(pool, size=batch_size, replace=len(pool) < batch_size)


def _build_wilds_stream(
    *,
    x_test: np.ndarray,
    y_test: np.ndarray,
    group_ids: np.ndarray,
    selected_groups: tuple[WildsGroupSummary, ...],
    steps: int,
    batch_size: int,
    seed: int,
) -> list[TabularBatch]:
    rng = np.random.default_rng(seed)
    role_to_group = {summary.role: summary.group_id for summary in selected_groups}
    easy_group = role_to_group["easy"]
    hard_group = role_to_group["hard"]
    recurring_group = role_to_group["recurring"]

    easy_indices = np.flatnonzero(group_ids == easy_group)
    hard_indices = np.flatnonzero(group_ids == hard_group)
    recurring_indices = np.flatnonzero(group_ids == recurring_group)
    mixed_indices = np.concatenate([hard_indices, recurring_indices], axis=0)

    schedule = (
        ("easy_stable", easy_indices),
        ("hard_shift", hard_indices),
        ("recurring_shift", recurring_indices),
        ("mixed_shift", mixed_indices),
        ("easy_return", easy_indices),
        ("recurring_return", recurring_indices),
        ("hard_recurrence", hard_indices),
    )
    segment_length = max(1, steps // len(schedule))

    batches: list[TabularBatch] = []
    for step in range(steps):
        regime, pool = schedule[min(len(schedule) - 1, step // segment_length)]
        chosen = _sample_group_batch(rng, pool, batch_size=batch_size)
        batches.append(
            TabularBatch(
                features=x_test[chosen],
                labels=y_test[chosen],
                regime=regime,
            )
        )
    return batches


def run_wilds_civilcomments_benchmark(
    *,
    config: WildsCivilCommentsConfig | None = None,
    policy_factories: list[tuple[str, PolicyFactory]] | None = None,
) -> WildsCivilCommentsBenchmarkResult:
    effective_config = WildsCivilCommentsConfig() if config is None else config
    get_dataset = _require_wilds()
    dataset = get_dataset(
        dataset="civilcomments",
        root_dir=str(Path(effective_config.root_dir)),
        download=effective_config.download,
    )

    train_subset = dataset.get_subset("train")
    validation_subset = dataset.get_subset("val")
    test_subset = dataset.get_subset("test")

    train_texts, y_train, _ = _extract_subset(
        train_subset,
        limit=effective_config.train_limit,
        seed=effective_config.seed,
    )
    validation_texts, y_validation, _ = _extract_subset(
        validation_subset,
        limit=effective_config.validation_limit,
        seed=effective_config.seed + 1,
    )
    test_texts, y_test, test_metadata = _extract_subset(
        test_subset,
        limit=effective_config.test_limit,
        seed=effective_config.seed + 2,
    )

    x_train, x_validation, x_test = _vectorize_text(
        train_texts,
        validation_texts,
        test_texts,
        max_features=effective_config.max_tfidf_features,
        svd_dim=effective_config.svd_dim,
    )
    group_ids = _group_ids(dataset, test_metadata)

    source_model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=effective_config.seed)
    source_summary = source_model.fit_source(x_train, y_train, x_validation, y_validation)
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=effective_config.batch_size,
        seed=effective_config.seed + 17,
    )
    reference, reference_scores = _build_reference_profile(source_model, reference_batches)
    selected_groups = _selected_group_roles(
        _group_statistics(
            source_model,
            x_test,
            y_test,
            group_ids,
            min_group_count=effective_config.min_group_count,
        )
    )
    stream = _build_wilds_stream(
        x_test=x_test,
        y_test=y_test,
        group_ids=group_ids,
        selected_groups=selected_groups,
        steps=effective_config.steps,
        batch_size=effective_config.batch_size,
        seed=effective_config.seed + 31,
    )
    factories = _default_policy_factories() if policy_factories is None else policy_factories
    benchmark = run_tabular_benchmark_with_factories(
        policy_factories=factories,
        steps=effective_config.steps,
        batch_size=effective_config.batch_size,
        seed=effective_config.seed,
        prepared=(
            source_summary,
            reference,
            reference_scores,
            stream,
            source_model,
        ),
    )
    return WildsCivilCommentsBenchmarkResult(
        config=effective_config,
        source_summary=source_summary,
        reference=reference,
        selected_groups=selected_groups,
        benchmark=benchmark,
    )


def wilds_civilcomments_benchmark_to_dict(result: WildsCivilCommentsBenchmarkResult) -> dict:
    payload = {
        "config": asdict(result.config),
        "source_summary": asdict(result.source_summary),
        "reference": {
            "feature_mean": result.reference.feature_mean.tolist(),
            "feature_variance": result.reference.feature_variance.tolist(),
            "mean_entropy": result.reference.mean_entropy,
            "mean_probability": result.reference.mean_probability,
            "positive_rate": result.reference.positive_rate,
            "mean_confidence": result.reference.mean_confidence,
        },
        "selected_groups": [asdict(group) for group in result.selected_groups],
    }
    payload["benchmark"] = {
        "steps": result.benchmark.steps,
        "batch_size": result.benchmark.batch_size,
        "strategies": [
            {
                "name": strategy.name,
                "overall_accuracy": strategy.overall_accuracy,
                "served_accuracy": strategy.served_accuracy,
                "coverage": strategy.coverage,
                "mean_utility": strategy.mean_utility,
                "risk_alerts": strategy.risk_alerts,
                "mean_risk_capital": strategy.mean_risk_capital,
                "mean_parameter_drift": strategy.mean_parameter_drift,
                "regime_accuracy": strategy.regime_accuracy,
                "diagnostics": strategy.diagnostics,
            }
            for strategy in result.benchmark.strategies
        ],
    }
    return payload


def render_wilds_civilcomments_report(result: WildsCivilCommentsBenchmarkResult) -> str:
    lines = [
        "Adaptive Reliability Layer WILDS CivilComments Benchmark",
        (
            f"steps={result.config.steps} batch_size={result.config.batch_size} "
            f"train_limit={result.config.train_limit} val_limit={result.config.validation_limit} "
            f"test_limit={result.config.test_limit}"
        ),
        f"source_val_acc={result.source_summary.best_validation_accuracy:.3f}",
        "",
        "selected_groups",
    ]
    for summary in result.selected_groups:
        lines.append(
            f"{summary.role:<12}"
            f"group_id={summary.group_id:<6}"
            f"group={summary.group_name:<18}"
            f"count={summary.count:<6}"
            f"source_acc={summary.source_accuracy:.3f}"
        )
    lines.append("")
    lines.append("strategy                      accuracy   utility   delta_vs_frozen   risk_alerts   mean_capital")
    frozen_accuracy = next(
        strategy.overall_accuracy
        for strategy in result.benchmark.strategies
        if strategy.name == "frozen"
    )
    for strategy in result.benchmark.strategies:
        lines.append(
            f"{strategy.name:<28}"
            f"{strategy.overall_accuracy:>8.3f}"
            f"{strategy.mean_utility:>10.3f}"
            f"{strategy.overall_accuracy - frozen_accuracy:>18.3f}"
            f"{strategy.risk_alerts:>14}"
            f"{strategy.mean_risk_capital:>15.3f}"
        )
    return "\n".join(lines)


_WILDS_SUMMARY_METRICS = (
    "overall_accuracy",
    "served_accuracy",
    "coverage",
    "mean_utility",
    "alerts",
    "risk_alerts",
    "adaptations",
    "resets",
    "abstains",
    "mean_shift_score",
    "mean_risk_capital",
    "mean_reliability",
    "mean_parameter_drift",
)


def _aggregate(values: list[float]) -> AggregateStat:
    array = np.asarray(values, dtype=np.float64)
    return AggregateStat(mean=float(array.mean()), std=float(array.std(ddof=0)))


def _default_suite_configs() -> tuple[WildsCivilCommentsConfig, ...]:
    return (
        WildsCivilCommentsConfig(
            suite_name="compact",
            train_limit=1200,
            validation_limit=400,
            test_limit=1600,
            batch_size=32,
            steps=18,
            max_tfidf_features=2500,
            svd_dim=48,
            min_group_count=48,
        ),
        WildsCivilCommentsConfig(
            suite_name="medium",
            train_limit=2400,
            validation_limit=800,
            test_limit=3000,
            batch_size=48,
            steps=24,
            max_tfidf_features=3500,
            svd_dim=64,
            min_group_count=56,
        ),
    )


def _aggregate_wilds_results(
    *,
    name: str,
    seeds: tuple[int, ...],
    configs: tuple[WildsCivilCommentsConfig, ...],
    results: list[WildsCivilCommentsBenchmarkResult],
) -> WildsBenchmarkAggregate:
    by_strategy: dict[str, list] = {}
    for result in results:
        for strategy in result.benchmark.strategies:
            by_strategy.setdefault(strategy.name, []).append(strategy)

    strategy_aggregates: list[WildsStrategyAggregate] = []
    for strategy_name, strategy_runs in sorted(by_strategy.items()):
        metrics = {
            metric_name: _aggregate([float(getattr(strategy, metric_name)) for strategy in strategy_runs])
            for metric_name in _WILDS_SUMMARY_METRICS
        }
        regime_names = sorted({regime for strategy in strategy_runs for regime in strategy.regime_accuracy})
        regime_accuracy = {
            regime_name: _aggregate(
                [float(strategy.regime_accuracy.get(regime_name, 0.0)) for strategy in strategy_runs]
            )
            for regime_name in regime_names
        }
        diagnostic_names = sorted({key for strategy in strategy_runs for key in strategy.diagnostics})
        diagnostics = {
            diagnostic_name: _aggregate(
                [float(strategy.diagnostics.get(diagnostic_name, 0.0)) for strategy in strategy_runs]
            )
            for diagnostic_name in diagnostic_names
        }
        strategy_aggregates.append(
            WildsStrategyAggregate(
                name=strategy_name,
                metrics=metrics,
                regime_accuracy=regime_accuracy,
                diagnostics=diagnostics,
            )
        )

    return WildsBenchmarkAggregate(
        name=name,
        seeds=seeds,
        configs=configs,
        strategies=tuple(strategy_aggregates),
    )


def _summarize_suite(benchmarks: list[WildsBenchmarkAggregate]) -> tuple[WildsStrategySummary, ...]:
    by_strategy: dict[str, dict[str, float | list[float]]] = {}
    for benchmark in benchmarks:
        utility_scores = {
            strategy.name: strategy.metrics["mean_utility"].mean
            for strategy in benchmark.strategies
        }
        accuracy_scores = {
            strategy.name: strategy.metrics["overall_accuracy"].mean
            for strategy in benchmark.strategies
        }
        best_utility = max(utility_scores.values())
        best_accuracy = max(accuracy_scores.values())
        frozen_utility = utility_scores.get("frozen", 0.0)
        frozen_accuracy = accuracy_scores.get("frozen", 0.0)
        for strategy in benchmark.strategies:
            slot = by_strategy.setdefault(
                strategy.name,
                {
                    "utility_wins": 0.0,
                    "accuracy_wins": 0.0,
                    "utility_margins": [],
                    "accuracy_margins": [],
                },
            )
            if utility_scores[strategy.name] >= best_utility - 1e-6:
                slot["utility_wins"] = float(slot["utility_wins"]) + 1.0
            if accuracy_scores[strategy.name] >= best_accuracy - 1e-6:
                slot["accuracy_wins"] = float(slot["accuracy_wins"]) + 1.0
            utility_margins = slot["utility_margins"]
            accuracy_margins = slot["accuracy_margins"]
            assert isinstance(utility_margins, list)
            assert isinstance(accuracy_margins, list)
            utility_margins.append(utility_scores[strategy.name] - frozen_utility)
            accuracy_margins.append(accuracy_scores[strategy.name] - frozen_accuracy)

    summaries: list[WildsStrategySummary] = []
    for strategy_name in sorted(by_strategy):
        slot = by_strategy[strategy_name]
        utility_margins = slot["utility_margins"]
        accuracy_margins = slot["accuracy_margins"]
        assert isinstance(utility_margins, list)
        assert isinstance(accuracy_margins, list)
        summaries.append(
            WildsStrategySummary(
                name=strategy_name,
                utility_wins=int(slot["utility_wins"]),
                accuracy_wins=int(slot["accuracy_wins"]),
                mean_utility_margin_vs_frozen=float(np.mean(utility_margins)) if utility_margins else 0.0,
                mean_accuracy_margin_vs_frozen=float(np.mean(accuracy_margins)) if accuracy_margins else 0.0,
            )
        )
    return tuple(summaries)


def run_wilds_civilcomments_suite(
    *,
    seeds: tuple[int, ...] = (7, 11),
    configs: tuple[WildsCivilCommentsConfig, ...] | None = None,
) -> WildsCivilCommentsSuiteResult:
    effective_configs = _default_suite_configs() if configs is None else configs
    benchmarks: list[WildsBenchmarkAggregate] = []
    for config in effective_configs:
        results = [
            run_wilds_civilcomments_benchmark(
                config=WildsCivilCommentsConfig(
                    **{
                        **asdict(config),
                        "seed": seed,
                    }
                )
            )
            for seed in seeds
        ]
        benchmarks.append(
            _aggregate_wilds_results(
                name=config.suite_name,
                seeds=seeds,
                configs=effective_configs,
                results=results,
            )
        )
    return WildsCivilCommentsSuiteResult(
        seeds=seeds,
        benchmarks=tuple(benchmarks),
        summary=_summarize_suite(benchmarks),
    )


def wilds_civilcomments_suite_to_dict(result: WildsCivilCommentsSuiteResult) -> dict:
    return asdict(result)


def render_wilds_civilcomments_suite_report(result: WildsCivilCommentsSuiteResult) -> str:
    lines = [
        "Adaptive Reliability Layer WILDS CivilComments Suite",
        f"seeds={','.join(str(seed) for seed in result.seeds)}",
        "",
    ]
    if result.summary:
        lines.append("suite_summary")
        lines.append(
            "strategy                      utility_wins   accuracy_wins   mean_utility_delta_vs_frozen   mean_accuracy_delta_vs_frozen"
        )
        for strategy in result.summary:
            lines.append(
                f"{strategy.name:<28}"
                f"{strategy.utility_wins:>6}"
                f"{strategy.accuracy_wins:>16}"
                f"{strategy.mean_utility_margin_vs_frozen:>30.3f}"
                f"{strategy.mean_accuracy_margin_vs_frozen:>31.3f}"
            )
        lines.append("")

    for benchmark in result.benchmarks:
        config = benchmark.configs[0] if benchmark.configs else None
        config_bits = ""
        if config is not None:
            matching = [item for item in benchmark.configs if item.suite_name == benchmark.name]
            if matching:
                item = matching[0]
                config_bits = (
                    f" steps={item.steps} train={item.train_limit} val={item.validation_limit} "
                    f"test={item.test_limit}"
                )
        lines.append(f"[{benchmark.name}{config_bits}]")
        lines.append(
            "strategy                      acc(mean±std)   utility(mean±std)   "
            "risk_capital(mean±std)   risk_alerts(mean±std)   drift(mean±std)"
        )
        for strategy in benchmark.strategies:
            accuracy = strategy.metrics["overall_accuracy"]
            utility = strategy.metrics["mean_utility"]
            risk_capital = strategy.metrics["mean_risk_capital"]
            risk_alerts = strategy.metrics["risk_alerts"]
            parameter_drift = strategy.metrics["mean_parameter_drift"]
            lines.append(
                f"{strategy.name:<28}"
                f"{accuracy.mean:>7.3f}±{accuracy.std:<6.3f}"
                f"{utility.mean:>18.3f}±{utility.std:<6.3f}"
                f"{risk_capital.mean:>23.3f}±{risk_capital.std:<6.3f}"
                f"{risk_alerts.mean:>22.3f}±{risk_alerts.std:<6.3f}"
                f"{parameter_drift.mean:>16.3f}±{parameter_drift.std:<6.3f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip()
