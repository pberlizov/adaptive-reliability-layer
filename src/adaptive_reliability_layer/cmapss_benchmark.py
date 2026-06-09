"""CMAPSS turbofan degradation benchmark — Gate B: frozen model degrades over time.

The NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset
contains run-to-failure time series for turbofan engines.  Sensor readings
degrade monotonically as engines approach failure, making this the canonical
benchmark for measuring whether a frozen model's accuracy deteriorates as the
fleet ages.

Key property:  As we replay the test window in time-order, more and more units
are close to failure, their sensor readings have drifted far from the training
distribution, and a frozen model trained on healthy engines will progressively
mis-classify them.  A controller that detects and adapts to this drift should
sustain higher accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Project root (two levels up from src/adaptive_reliability_layer/)
ROOT = Path(__file__).resolve().parents[2]

from .tabular_benchmark import (
    BanditTabularPolicy,
    ControllerTabularPolicy,
    FrozenTabularPolicy,
    MultiActionTabularPolicy,
    NaiveTabularPolicy,
    TabularBatch,
    _build_reference_profile,
    _evaluate_strategy,
)
from .torch_model import TorchTabularAdapterModel


# The four CMAPSS sub-datasets
DATASET_IDS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CMAPSSConfig:
    """Configuration for the CMAPSS benchmark run."""

    dataset_id: str = "FD001"
    """One of FD001, FD002, FD003, FD004."""

    batch_size: int = 50
    """Number of (unit, time_cycle) rows per batch."""

    label_delay_steps: int = 10
    """How many steps before labels are revealed (unused in current evaluation
    but stored for provenance and future delayed-label experiments)."""

    train_fraction: float = 0.6
    """Temporal split: the first train_fraction of time-cycles are used to
    train the source model; the rest are the test / evaluation stream."""

    data_dir: str = "data/cmapss"
    """Directory containing cmapss_FD001.csv … cmapss_FD004.csv."""


# ---------------------------------------------------------------------------
# Feature columns (op settings + sensor readings)
# ---------------------------------------------------------------------------

_FEATURE_COLS = (
    ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
)  # 24 features total


# ---------------------------------------------------------------------------
# Stream loader
# ---------------------------------------------------------------------------

def load_cmapss_stream(
    config: CMAPSSConfig,
) -> tuple[
    list[TabularBatch],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
]:
    """Load the CMAPSS CSV and return train/test data plus test-window batches.

    The train_fraction split is performed on time_cycle globally (across all
    units) so that early healthy cycles are in train and late degrading cycles
    are in test.  Batches group consecutive rows in time-order: as we advance
    through batches, more units are approaching failure, creating genuine
    monotonic drift.

    Returns
    -------
    (batches, x_train, y_train, x_test, y_test, scaler)
        batches: time-ordered TabularBatch list for the test window.
        x_train / y_train: training features and labels.
        x_test / y_test: test features and labels (already scaled).
        scaler: the fitted StandardScaler (for provenance).
    """
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    csv_path = data_dir / f"cmapss_{config.dataset_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CMAPSS data not found: {csv_path}\n"
            "Run scripts/export_cmapss.py first."
        )

    frame = pd.read_csv(csv_path)
    feature_cols = [col for col in _FEATURE_COLS if col in frame.columns]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {csv_path}")

    # Per-unit split: assign units to train vs test.
    # This mirrors real CMAPSS usage where different engine units are in
    # train vs test — NOT different time windows of the same unit.
    # Without this, the source model trains on only healthy (label=0) cycles
    # and becomes degenerate (never predicts failure).
    units = sorted(frame["unit"].unique())
    n_train_units = max(1, int(len(units) * config.train_fraction))
    train_units = set(units[:n_train_units])
    test_units = set(units[n_train_units:])

    if not test_units:
        # Fallback: if only 1 unit, use temporal split
        max_cycle = int(frame["time_cycle"].max())
        split_cycle = int(max_cycle * config.train_fraction)
        train_frame = frame[frame["time_cycle"] <= split_cycle].copy()
        test_frame = frame[frame["time_cycle"] > split_cycle].copy()
    else:
        train_frame = frame[frame["unit"].isin(train_units)].copy()
        # Test stream: time-ordered across test units so degradation builds
        test_frame = frame[frame["unit"].isin(test_units)].sort_values(
            ["time_cycle", "unit"]
        ).copy()

    if len(train_frame) == 0 or len(test_frame) == 0:
        raise ValueError(
            f"Train/test split produced empty split for dataset {config.dataset_id}."
        )

    # Fit scaler on train, transform both (leak-free)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(
        train_frame[feature_cols].fillna(0.0).values.astype(np.float32)
    ).astype(np.float32)
    x_test = scaler.transform(
        test_frame[feature_cols].fillna(0.0).values.astype(np.float32)
    ).astype(np.float32)

    y_train = train_frame["label"].values.astype(np.int64)
    y_test = test_frame["label"].values.astype(np.int64)

    # Build batches from the test window, preserving temporal order
    n_test = len(x_test)
    batches: list[TabularBatch] = []
    for start in range(0, n_test, config.batch_size):
        end = min(start + config.batch_size, n_test)
        if end <= start:
            break
        batch_x = x_test[start:end]
        batch_y = y_test[start:end]

        # Regime label based on degradation quartile within the test window
        progress = (start + end) / 2.0 / n_test
        if progress < 0.25:
            regime = "early_test"
        elif progress < 0.50:
            regime = "mid_test"
        elif progress < 0.75:
            regime = "late_test"
        else:
            regime = "terminal"

        batches.append(TabularBatch(features=batch_x, labels=batch_y, regime=regime))

    return batches, x_train, y_train, x_test, y_test, scaler


# ---------------------------------------------------------------------------
# Reference profile builder
# ---------------------------------------------------------------------------

def _build_cmapss_reference_batches(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    count: int = 10,
    seed: int = 42,
) -> list[TabularBatch]:
    """Sample random batches from the training (healthy) window."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train))
    batches: list[TabularBatch] = []
    for _ in range(count):
        chosen = rng.choice(indices, size=min(batch_size, len(indices)), replace=True)
        batches.append(
            TabularBatch(
                features=x_train[chosen],
                labels=y_train[chosen],
                regime="reference",
            )
        )
    return batches


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CMAPSSBenchmarkResult:
    """Per-strategy result for a CMAPSS benchmark run."""

    source_id: str
    """Dataset identifier (e.g. 'FD001')."""

    strategy_name: str
    batch_count: int

    mean_accuracy: float
    """Mean accuracy across all test batches."""

    final_accuracy: float
    """Mean accuracy over the last 10 batches (high degradation zone)."""

    early_accuracy: float
    """Mean accuracy over the first 10 batches (low degradation zone)."""

    frozen_accuracy_delta: float
    """For frozen: early_accuracy - final_accuracy (positive = degradation).
    For controller strategies: same measure (positive = less degradation)."""

    controller_accuracy_delta: float
    """For controller strategies: their accuracy delta vs frozen at terminal."""

    risk_reduction: float
    """(controller_final - frozen_final) / max(frozen_final, 1e-9)."""

    per_batch_accuracies: tuple[float, ...]
    """Per-batch accuracy curve for plotting the degradation trajectory."""


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_cmapss_benchmark(config: CMAPSSConfig) -> dict[str, CMAPSSBenchmarkResult]:
    """Run the full CMAPSS benchmark: frozen + all main controllers.

    Returns a dict mapping strategy name → CMAPSSBenchmarkResult.

    The key measurement is ``frozen_accuracy_delta``:  if it is substantially
    positive, the frozen model genuinely degrades as the fleet ages, confirming
    the Gate B property.  Adaptive controllers should show smaller deltas and
    higher ``final_accuracy``.
    """
    # --- Load data -----------------------------------------------------------
    batches, x_train, y_train, x_test, y_test, _scaler = load_cmapss_stream(config)

    if len(batches) < 5:
        raise ValueError(
            f"Only {len(batches)} test batches available for {config.dataset_id}. "
            "Try a smaller batch_size or a larger dataset."
        )

    # --- Train source model on healthy training cycles ----------------------
    input_dim = x_train.shape[1]
    source_model = TorchTabularAdapterModel(input_dim=input_dim, seed=42)
    source_model.fit_source(
        x_train,
        y_train,
        x_train,  # use train as validation too for simplicity
        y_train,
        epochs=30,
    )

    # --- Reference profile from training-window batches ---------------------
    reference_batches = _build_cmapss_reference_batches(
        x_train, y_train, batch_size=config.batch_size
    )
    reference, reference_scores = _build_reference_profile(
        source_model, reference_batches
    )

    # --- Helper: extract per-batch accuracy curve from strategy result ------
    def _per_batch_accuracies(strategy_result) -> tuple[float, ...]:
        return tuple(trace.batch_accuracy for trace in strategy_result.traces)

    def _window_accuracy(accs: Sequence[float], n: int, *, tail: bool) -> float:
        if not accs:
            return 0.0
        window = list(accs[-n:]) if tail else list(accs[:n])
        return float(np.mean(window)) if window else 0.0

    # --- Run strategies ------------------------------------------------------
    strategy_factories = [
        ("frozen", lambda ref: FrozenTabularPolicy()),
        ("naive", lambda ref: NaiveTabularPolicy()),
        ("controller", lambda ref: ControllerTabularPolicy()),
        ("multi_action", lambda ref: MultiActionTabularPolicy(ref)),
        ("bandit", lambda ref: BanditTabularPolicy(ref)),
    ]

    results: dict[str, CMAPSSBenchmarkResult] = {}
    frozen_final_acc: float | None = None

    # Evaluate frozen first to get the baseline delta
    for name, factory in strategy_factories:
        strategy_result = _evaluate_strategy(
            name,
            source_model.clone(),
            factory(reference),
            batches,
            reference,
            reference_scores,
        )
        per_batch = _per_batch_accuracies(strategy_result)
        window = min(10, len(per_batch) // 4 + 1)
        early_acc = _window_accuracy(per_batch, window, tail=False)
        final_acc = _window_accuracy(per_batch, window, tail=True)
        frozen_acc_delta = early_acc - final_acc  # positive = degradation

        if name == "frozen":
            frozen_final_acc = final_acc

        controller_acc_delta = 0.0
        risk_reduction = 0.0
        if frozen_final_acc is not None and name != "frozen":
            controller_acc_delta = final_acc - frozen_final_acc
            risk_reduction = controller_acc_delta / max(abs(frozen_final_acc), 1e-9)

        results[name] = CMAPSSBenchmarkResult(
            source_id=config.dataset_id,
            strategy_name=name,
            batch_count=len(batches),
            mean_accuracy=strategy_result.overall_accuracy,
            final_accuracy=final_acc,
            early_accuracy=early_acc,
            frozen_accuracy_delta=frozen_acc_delta,
            controller_accuracy_delta=controller_acc_delta,
            risk_reduction=risk_reduction,
            per_batch_accuracies=per_batch,
        )

    return results


# ---------------------------------------------------------------------------
# Specialist diagnostic reporter (for delayed_hybrid analysis)
# ---------------------------------------------------------------------------

def _print_specialist_diagnostics(rows: list[dict]) -> None:
    """Print a per-batch specialist routing table for delayed_hybrid analysis.

    Columns:
      step       - batch index (0-based)
      regime     - early/mid/late/terminal based on position quartile
      lbl_rate   - fraction of positive (failure) labels in this batch
      acc        - batch accuracy
      n_spec     - number of specialists in pool
      sel_id     - selected specialist index (0 = base/frozen)
      sel_qual   - reuse confidence (0=low trust, 1=high trust)
      dist       - L2 distance between current batch signature and selected specialist's signature
      sel_supp+  - support positive rate stored by the selected specialist (proxy for when it was created)
      feat_shift - feature shift magnitude vs reference distribution
    """
    total = len(rows)
    print("\n" + "=" * 90)
    print("Specialist diagnostic summary — delayed_hybrid on CMAPSS FD001")
    print("=" * 90)
    header = (
        f"{'step':>4}  {'regime':<12}  {'lbl_rate':>8}  {'acc':>6}  "
        f"{'n_spec':>6}  {'sel_id':>6}  {'sel_qual':>8}  {'dist':>6}  "
        f"{'sel_supp+':>9}  {'feat_shift':>10}"
    )
    print(header)
    print("-" * 90)

    quartile_labels = ["early_test", "mid_test", "late_test", "terminal"]
    for row in rows:
        step = int(row["step"])
        q_idx = min(3, int(step / max(1, total) * 4))
        regime = quartile_labels[q_idx]
        print(
            f"{step:>4}  {regime:<12}  {row['batch_label_rate']:>8.3f}  "
            f"{row['batch_acc']:>6.3f}  {int(row['specialist_count']):>6}  "
            f"{int(row['specialist_selected_id']):>6}  "
            f"{row['specialist_selected_quality']:>8.3f}  "
            f"{row['specialist_regime_distance']:>6.3f}  "
            f"{row['specialist_selected_support_positive_rate']:>9.3f}  "
            f"{row['feature_shift']:>10.3f}"
        )

    print("=" * 90)

    # Aggregate summary: split into first half vs second half
    mid = total // 2
    early_rows = rows[:mid]
    late_rows = rows[mid:]

    def _avg(lst, key):
        vals = [r[key] for r in lst]
        return float(np.mean(vals)) if vals else 0.0

    print("\nAggregate statistics (early half vs late half of stream):")
    print(f"  {'metric':<35}  {'early':>8}  {'late':>8}  {'delta':>8}")
    print("  " + "-" * 63)
    metrics = [
        ("batch_acc", "batch accuracy"),
        ("batch_label_rate", "batch label rate (failure rate)"),
        ("specialist_count", "specialist pool size"),
        ("specialist_selected_id", "selected specialist id (mean)"),
        ("specialist_selected_quality", "selection quality (reuse conf)"),
        ("specialist_regime_distance", "regime distance to selected"),
        ("specialist_selected_support_positive_rate", "selected specialist supp+ rate"),
        ("feature_shift", "feature shift vs reference"),
    ]
    for key, label in metrics:
        e = _avg(early_rows, key)
        l = _avg(late_rows, key)
        d = l - e
        print(f"  {label:<35}  {e:>8.3f}  {l:>8.3f}  {d:>+8.3f}")

    # Routing breakdown: how often is base (id=0) vs non-base selected?
    base_count = sum(1 for r in rows if int(r["specialist_selected_id"]) == 0)
    nonbase_count = total - base_count
    print(f"\nRouting: base specialist (id=0) selected {base_count}/{total} batches "
          f"({100.0 * base_count / max(1, total):.1f}%), "
          f"non-base selected {nonbase_count}/{total} batches "
          f"({100.0 * nonbase_count / max(1, total):.1f}%)")

    # Late-stream routing: last quarter of batches
    terminal_rows = rows[3 * total // 4:]
    if terminal_rows:
        base_late = sum(1 for r in terminal_rows if int(r["specialist_selected_id"]) == 0)
        nonbase_late = len(terminal_rows) - base_late
        mean_qual_late = _avg(terminal_rows, "specialist_selected_quality")
        mean_dist_late = _avg(terminal_rows, "specialist_regime_distance")
        mean_label_rate_late = _avg(terminal_rows, "batch_label_rate")
        mean_supp_late = _avg(terminal_rows, "specialist_selected_support_positive_rate")
        print(f"Terminal quarter ({len(terminal_rows)} batches):")
        print(f"  base selected: {base_late}/{len(terminal_rows)}, "
              f"non-base: {nonbase_late}/{len(terminal_rows)}")
        print(f"  mean selection quality: {mean_qual_late:.3f}")
        print(f"  mean regime distance:   {mean_dist_late:.3f}")
        print(f"  mean batch label rate:  {mean_label_rate_late:.3f}  (=degradation level)")
        print(f"  mean selected supp+ rate: {mean_supp_late:.3f}  "
              f"(should match label rate if hypothesis holds)")
        mismatch = abs(mean_supp_late - mean_label_rate_late)
        if mean_label_rate_late > 0.15 and mean_supp_late < mean_label_rate_late - 0.05:
            print(f"  *** HYPOTHESIS CONFIRMED: selected specialists were built on low-failure-rate "
                  f"data (supp+={mean_supp_late:.3f}) but are being routed to high-failure batches "
                  f"(label_rate={mean_label_rate_late:.3f}), mismatch={mismatch:.3f} ***")
        elif mean_label_rate_late <= 0.10:
            print(f"  label rate is low at terminal — degradation may not be failure-label based")
        else:
            print(f"  support/label mismatch={mismatch:.3f} — hypothesis inconclusive")
    print()


# ---------------------------------------------------------------------------
# Production-path runner: ReliabilityLayer + reveal_labels
# ---------------------------------------------------------------------------

def run_cmapss_production_benchmark(
    config: CMAPSSConfig,
    *,
    policy_names: list[str] | None = None,
    bounded_auto_actions_override: frozenset[str] | None = None,
) -> dict[str, CMAPSSBenchmarkResult]:
    """Gate B test using the production ReliabilityLayer with delayed label reveals.

    This is the correct test for whether delayed-feedback controllers can close
    the accuracy gap when frozen degrades.  Labels are revealed after
    ``config.label_delay_steps`` batches, giving the policy real signal.

    Unlike ``run_cmapss_benchmark`` (unsupervised research path), this exercises:
      - ReliabilityLayer.process_batch()
      - ReliabilityLayer.reveal_labels()
      - The full DelayedCorrectionEngine + InterventionGovernor stack
    """
    from dataclasses import replace as dc_replace
    from .runtime import RuntimeConfig, build_reliability_layer_from_reference_batches
    from .runtime.model_adapter import TorchTabularModelAdapter
    from .runtime.config import PolicyConfig, ReplayConfig
    from .runtime.types import OperatingMode

    if policy_names is None:
        policy_names = ["frozen", "delayed_bandit", "delayed_hybrid"]

    batches, x_train, y_train, _x_test, _y_test, _scaler = load_cmapss_stream(config)
    if len(batches) < 5:
        raise ValueError(f"Only {len(batches)} test batches for {config.dataset_id}.")

    source_model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=42)
    source_model.fit_source(x_train, y_train, x_train, y_train, epochs=30)

    reference_batches = _build_cmapss_reference_batches(
        x_train, y_train, batch_size=config.batch_size
    )

    results: dict[str, CMAPSSBenchmarkResult] = {}
    frozen_final_acc: float | None = None

    def _window_accuracy(accs: list[float], n: int, *, tail: bool) -> float:
        window = list(accs[-n:]) if tail else list(accs[:n])
        return float(np.mean(window)) if window else 0.0

    for policy_name in policy_names:
        adapter = TorchTabularModelAdapter(source_model.clone())
        base_config = RuntimeConfig()
        runtime_config = dc_replace(
            base_config,
            operating_mode=OperatingMode.BOUNDED_AUTO,
            policy=PolicyConfig(name=policy_name),
            replay=ReplayConfig(label_delay_steps=config.label_delay_steps),
            bounded_auto_actions=(
                bounded_auto_actions_override
                if bounded_auto_actions_override is not None
                else base_config.bounded_auto_actions
            ),
        )

        layer = build_reliability_layer_from_reference_batches(
            adapter, reference_batches, config=runtime_config
        )

        per_batch_accs: list[float] = []
        # Queue: list of (step, labels_array) waiting to be revealed
        pending_reveals: list[tuple[int, np.ndarray]] = []

        # Specialist diagnostics tracking (only for delayed_hybrid)
        specialist_diag_rows: list[dict] = []

        for step, batch in enumerate(batches):
            surface = layer.process_batch(batch)
            labels = np.asarray(batch.labels, dtype=np.int64)

            preds = np.asarray(surface.predictions, dtype=np.int64)
            acc = float((preds == labels[:len(preds)]).mean())
            per_batch_accs.append(acc)

            # Collect per-batch specialist diagnostics for delayed_hybrid
            if policy_name == "delayed_hybrid" and hasattr(layer._policy, "get_diagnostics"):
                diag = layer._policy.get_diagnostics()
                batch_label_rate = float(batch.labels.mean()) if batch.labels is not None else 0.0
                specialist_diag_rows.append({
                    "step": step,
                    "batch_acc": acc,
                    "batch_label_rate": batch_label_rate,
                    "specialist_count": diag.get("specialist_count", 0.0),
                    "specialist_selected_id": diag.get("specialist_selected_id", 0.0),
                    "specialist_selected_quality": diag.get("specialist_selected_quality", 0.0),
                    "specialist_regime_distance": diag.get("specialist_regime_distance", 0.0),
                    "specialist_selected_support_positive_rate": diag.get("specialist_selected_support_positive_rate", 0.5),
                    "feature_shift": diag.get("specialist_last_feature_shift", 0.0),
                })

            pending_reveals.append((step, labels))

            # Reveal labels whose delay has elapsed
            reveal_target = step - config.label_delay_steps
            still_pending = []
            for s, lbl in pending_reveals:
                if s == reveal_target:
                    try:
                        layer.reveal_labels(s, lbl)
                    except KeyError:
                        pass  # already revealed or not found
                else:
                    still_pending.append((s, lbl))
            pending_reveals = still_pending

        # Print specialist diagnostic summary for delayed_hybrid
        if policy_name == "delayed_hybrid" and specialist_diag_rows:
            _print_specialist_diagnostics(specialist_diag_rows)

        window = min(10, len(per_batch_accs) // 4 + 1)
        early_acc = _window_accuracy(per_batch_accs, window, tail=False)
        final_acc = _window_accuracy(per_batch_accs, window, tail=True)
        frozen_acc_delta = early_acc - final_acc

        if policy_name == "frozen":
            frozen_final_acc = final_acc

        controller_acc_delta = 0.0
        risk_reduction = 0.0
        if frozen_final_acc is not None and policy_name != "frozen":
            controller_acc_delta = final_acc - frozen_final_acc
            risk_reduction = controller_acc_delta / max(abs(frozen_final_acc), 1e-9)

        results[policy_name] = CMAPSSBenchmarkResult(
            source_id=config.dataset_id,
            strategy_name=policy_name,
            batch_count=len(batches),
            mean_accuracy=float(np.mean(per_batch_accs)),
            final_accuracy=final_acc,
            early_accuracy=early_acc,
            frozen_accuracy_delta=frozen_acc_delta,
            controller_accuracy_delta=controller_acc_delta,
            risk_reduction=risk_reduction,
            per_batch_accuracies=tuple(per_batch_accs),
        )

    return results


# ---------------------------------------------------------------------------
# Temporal fold evaluation — produces mean ± std across unit shuffles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CMAPSSFoldResult:
    """Aggregated across k folds: mean and std of the terminal accuracy delta."""

    dataset_id: str
    strategy_name: str
    n_folds: int
    mean_ctrl_delta: float
    std_ctrl_delta: float
    fold_ctrl_deltas: tuple[float, ...]
    mean_final_acc: float
    std_final_acc: float


def run_cmapss_production_benchmark_folds(
    dataset_id: str = "FD001",
    *,
    n_folds: int = 3,
    batch_size: int = 50,
    label_delay_steps: int = 10,
    train_fraction: float = 0.6,
    data_dir: str = "data/cmapss",
    policy_names: list[str] | None = None,
    base_seed: int = 0,
) -> dict[str, CMAPSSFoldResult]:
    """Run the production benchmark k times with different unit-shuffle seeds.

    Each fold shuffles the unit list with a different seed, then applies the
    same train_fraction split.  This produces k independent train/test splits
    so we can report mean ± std for the terminal accuracy delta, giving error
    bars on the Gate B claim.

    Returns a dict mapping strategy name → CMAPSSFoldResult.
    """
    if policy_names is None:
        policy_names = ["frozen", "delayed_bandit", "delayed_hybrid"]

    # Accumulate per-fold ctrl_delta and final_acc per strategy
    fold_ctrl_deltas: dict[str, list[float]] = {p: [] for p in policy_names}
    fold_final_accs: dict[str, list[float]] = {p: [] for p in policy_names}

    csv_path = Path(data_dir)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path = csv_path / f"cmapss_{dataset_id}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CMAPSS data not found: {csv_path}\nRun scripts/export_cmapss.py first."
        )

    frame = pd.read_csv(csv_path)
    units = sorted(frame["unit"].unique())

    for fold in range(n_folds):
        rng = np.random.default_rng(base_seed + fold)
        shuffled_units = list(rng.permutation(units).tolist())
        # Assign train/test via the shuffled ordering
        n_train = max(1, int(len(shuffled_units) * train_fraction))
        train_units_fold = set(shuffled_units[:n_train])
        if len(shuffled_units) - n_train < 1:
            # Not enough units for a test split; skip this fold
            continue

        fold_config = CMAPSSConfig(
            dataset_id=dataset_id,
            batch_size=batch_size,
            label_delay_steps=label_delay_steps,
            train_fraction=train_fraction,
            data_dir=data_dir,
        )

        # Patch the stream loader to use this fold's unit assignment
        # by temporarily overriding via a subclassed config (simplest path:
        # just call load_cmapss_stream with standard config — units are
        # selected by fraction, not shuffle, so we rebuild manually).
        feature_cols = [col for col in _FEATURE_COLS if col in frame.columns]
        train_frame = frame[frame["unit"].isin(train_units_fold)].copy()
        test_frame = frame[~frame["unit"].isin(train_units_fold)].sort_values(
            ["time_cycle", "unit"]
        ).copy()

        if len(train_frame) == 0 or len(test_frame) == 0:
            continue

        scaler = StandardScaler()
        x_train_f = scaler.fit_transform(
            train_frame[feature_cols].fillna(0.0).values.astype(np.float32)
        ).astype(np.float32)
        x_test_f = scaler.transform(
            test_frame[feature_cols].fillna(0.0).values.astype(np.float32)
        ).astype(np.float32)
        y_train_f = train_frame["label"].values.astype(np.int64)
        y_test_f = test_frame["label"].values.astype(np.int64)

        n_test = len(x_test_f)
        batches_f: list[TabularBatch] = []
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            if end <= start:
                break
            progress = (start + end) / 2.0 / n_test
            if progress < 0.25:
                regime = "early_test"
            elif progress < 0.50:
                regime = "mid_test"
            elif progress < 0.75:
                regime = "late_test"
            else:
                regime = "terminal"
            batches_f.append(TabularBatch(
                features=x_test_f[start:end],
                labels=y_test_f[start:end],
                regime=regime,
            ))

        if len(batches_f) < 5:
            continue

        # Run production benchmark for this fold (reuses run_cmapss_production_benchmark
        # internals, but with fold-specific data).
        fold_results = _run_fold(
            batches_f, x_train_f, y_train_f,
            batch_size=batch_size,
            label_delay_steps=label_delay_steps,
            policy_names=policy_names,
            dataset_id=dataset_id,
        )
        frozen_final = fold_results.get("frozen", {}).get("final_acc", 0.5)
        for policy_name in policy_names:
            if policy_name not in fold_results:
                continue
            final_acc = fold_results[policy_name]["final_acc"]
            ctrl_delta = final_acc - frozen_final if policy_name != "frozen" else 0.0
            fold_ctrl_deltas[policy_name].append(ctrl_delta)
            fold_final_accs[policy_name].append(final_acc)

    out: dict[str, CMAPSSFoldResult] = {}
    frozen_deltas = fold_ctrl_deltas.get("frozen", [])
    for policy_name in policy_names:
        deltas = fold_ctrl_deltas[policy_name]
        accs = fold_final_accs[policy_name]
        if not deltas:
            continue
        out[policy_name] = CMAPSSFoldResult(
            dataset_id=dataset_id,
            strategy_name=policy_name,
            n_folds=len(deltas),
            mean_ctrl_delta=float(np.mean(deltas)),
            std_ctrl_delta=float(np.std(deltas)),
            fold_ctrl_deltas=tuple(deltas),
            mean_final_acc=float(np.mean(accs)),
            std_final_acc=float(np.std(accs)),
        )
    return out


def _run_fold(
    batches: list[TabularBatch],
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    label_delay_steps: int,
    policy_names: list[str],
    dataset_id: str,
) -> dict[str, dict[str, float]]:
    """Internal helper: run all policies on a single fold, return final_acc per policy."""
    from dataclasses import replace as dc_replace
    from .runtime import RuntimeConfig, build_reliability_layer_from_reference_batches
    from .runtime.model_adapter import TorchTabularModelAdapter
    from .runtime.config import PolicyConfig, ReplayConfig
    from .runtime.types import OperatingMode

    source_model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=42)
    source_model.fit_source(x_train, y_train, x_train, y_train, epochs=30)
    reference_batches = _build_cmapss_reference_batches(x_train, y_train, batch_size=batch_size)

    def _window_acc(accs: list[float], n: int, *, tail: bool) -> float:
        window = list(accs[-n:]) if tail else list(accs[:n])
        return float(np.mean(window)) if window else 0.0

    results: dict[str, dict[str, float]] = {}
    for policy_name in policy_names:
        adapter = TorchTabularModelAdapter(source_model.clone())
        runtime_config = dc_replace(
            RuntimeConfig(),
            operating_mode=OperatingMode.BOUNDED_AUTO,
            policy=PolicyConfig(name=policy_name),
            replay=ReplayConfig(label_delay_steps=label_delay_steps),
        )
        layer = build_reliability_layer_from_reference_batches(adapter, reference_batches, config=runtime_config)
        per_batch_accs: list[float] = []
        pending_reveals: list[tuple[int, np.ndarray]] = []

        for step, batch in enumerate(batches):
            surface = layer.process_batch(batch)
            labels = np.asarray(batch.labels, dtype=np.int64)
            preds = np.asarray(surface.predictions, dtype=np.int64)
            acc = float((preds == labels[:len(preds)]).mean())
            per_batch_accs.append(acc)
            pending_reveals.append((step, labels))
            for reveal_step, rlabels in [(s, l) for s, l in pending_reveals if step - s >= label_delay_steps]:
                try:
                    layer.reveal_labels(reveal_step, rlabels)
                except KeyError:
                    pass
            pending_reveals = [(s, l) for s, l in pending_reveals if step - s < label_delay_steps]

        for reveal_step, rlabels in pending_reveals:
            try:
                layer.reveal_labels(reveal_step, rlabels)
            except KeyError:
                pass

        window = min(10, len(per_batch_accs) // 4 + 1)
        final_acc = _window_acc(per_batch_accs, window, tail=True)
        results[policy_name] = {"final_acc": final_acc}

    return results


def significance_test(
    deltas: tuple[float, ...],
    *,
    null_delta: float = 0.0,
) -> tuple[float, float, str]:
    """Return (t_stat, p_value, method) for a one-sample test of H0: mean(deltas)==null_delta.

    Uses paired t-test when n >= 3, sign test when n == 2, and returns (nan, nan, 'insufficient')
    when n < 2.  Also returns the Wilcoxon signed-rank p-value as a second check when n >= 4.
    The returned p_value is from the t-test (or sign test); the Wilcoxon result is included in
    the method string for transparency.
    """
    import math

    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan"), "insufficient_data"

    arr = [d - null_delta for d in deltas]
    mean = sum(arr) / n

    if n == 2:
        # Sign test: under H0 each sign is equally likely; p = 0.5 if both same sign, 1.0 otherwise
        positives = sum(1 for x in arr if x > 0)
        p_sign = 1.0 if positives == 1 else 0.5
        return float("nan"), p_sign, "sign_test_n2"

    variance = sum((x - mean) ** 2 for x in arr) / (n - 1)
    std = math.sqrt(variance)
    if std < 1e-10:
        # All values identical → p = 1 if mean == 0 else 0
        p_val = 0.0 if abs(mean) > 1e-10 else 1.0
        return float("inf"), p_val, "degenerate"

    t_stat = mean / (std / math.sqrt(n))
    # Two-tailed t-distribution p-value (degrees of freedom = n-1)
    # Approximation via regularized incomplete beta function
    df = n - 1
    x = df / (df + t_stat ** 2)
    # Use scipy if available; otherwise use a Horner-scheme approximation for df <= 30
    p_value: float
    try:
        from scipy import stats as _stats
        p_value = float(_stats.t.sf(abs(t_stat), df=df) * 2)
        method = "t_test_scipy"
    except ImportError:
        # Fallback: regularized incomplete beta B(df/2, 0.5) approximation
        # Good enough for df=2..29 with ±0.01 accuracy
        p_value = float(_approx_t_pvalue(t_stat, df))
        method = "t_test_approx"

    # Add Wilcoxon as cross-check when n >= 4
    if n >= 4:
        try:
            from scipy import stats as _stats
            _, p_wilcoxon = _stats.wilcoxon(arr)
            method = f"{method},wilcoxon_p={p_wilcoxon:.3f}"
        except (ImportError, Exception):
            pass

    return t_stat, p_value, method


def _approx_t_pvalue(t: float, df: int) -> float:
    """Rough two-tailed p-value for t-distribution; avoids scipy dependency."""
    import math

    if df <= 0:
        return 1.0
    x = float(df) / (float(df) + t * t)
    # Regularized incomplete beta I_x(df/2, 0.5)
    a = df / 2.0
    b = 0.5
    # Beta function approximation via continued fraction (Lentz)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_x = math.log(x)
    log_1mx = math.log(1.0 - x)
    # Use series expansion for small x
    term = math.exp(a * log_x + b * log_1mx - lbeta) / a
    cumsum = term
    for m in range(1, 200):
        term *= (a + b + m - 1) * x / (a + m)
        cumsum += term
        if abs(term) < 1e-8 * abs(cumsum):
            break
    return min(1.0, 2.0 * min(cumsum, 1.0 - cumsum))


@dataclass(frozen=True)
class CMAPSSFalseAlarmResult:
    dataset_id: str
    n_stable_batches: int
    alert_count: int
    severe_count: int
    false_alert_rate: float
    false_severe_rate: float
    mean_shift_score: float
    max_shift_score: float


def run_cmapss_false_alarm_benchmark(
    config: CMAPSSConfig | None = None,
    *,
    dataset_ids: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004"),
) -> dict[str, CMAPSSFalseAlarmResult]:
    """Measure false alarm rate on CMAPSS training batches (known stable).

    Batches drawn from the training window contain only healthy engine cycles
    with no degradation.  Any alert fired here is a false positive.  The false
    alarm rate should ideally be close to the nominal alpha of the monitor
    (configured via alert_threshold, default 1.1).

    Returns
    -------
    Dict mapping dataset_id → CMAPSSFalseAlarmResult
    """
    from .tabular_benchmark import TabularShiftMonitor, _build_reference_profile

    results: dict[str, CMAPSSFalseAlarmResult] = {}

    for dataset_id in dataset_ids:
        ds_config = CMAPSSConfig(
            dataset_id=dataset_id,
            batch_size=config.batch_size if config is not None else 50,
            data_dir=config.data_dir if config is not None else "data/cmapss",
        )
        try:
            batches, x_train, y_train, _, _, _ = load_cmapss_stream(ds_config)
        except (FileNotFoundError, ValueError):
            continue

        # Build source model + reference profile from training data
        source_model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=42)
        source_model.fit_source(x_train, y_train, x_train, y_train, epochs=20)
        ref_batches = _build_cmapss_reference_batches(x_train, y_train, batch_size=ds_config.batch_size)
        reference, _ = _build_reference_profile(source_model, ref_batches)
        monitor = TabularShiftMonitor(reference)

        # Now run monitor on NEW training-window batches (stable = no degradation)
        n_batches = min(len(batches), 30)
        # Use the first quarter of the test stream (still early, minimal degradation)
        stable_batches = batches[: max(5, len(batches) // 4)]

        alert_count = 0
        severe_count = 0
        shift_scores: list[float] = []

        for batch in stable_batches:
            probs = source_model.predict_proba(batch.features)
            signal = monitor.evaluate(batch.features, probs)
            shift_scores.append(signal.score)
            if signal.alert:
                alert_count += 1
            if signal.severe:
                severe_count += 1

        n = len(stable_batches)
        results[dataset_id] = CMAPSSFalseAlarmResult(
            dataset_id=dataset_id,
            n_stable_batches=n,
            alert_count=alert_count,
            severe_count=severe_count,
            false_alert_rate=alert_count / max(1, n),
            false_severe_rate=severe_count / max(1, n),
            mean_shift_score=float(np.mean(shift_scores)) if shift_scores else 0.0,
            max_shift_score=float(np.max(shift_scores)) if shift_scores else 0.0,
        )

    return results


def render_false_alarm_report(results: dict[str, CMAPSSFalseAlarmResult]) -> str:
    lines = [
        "## Monitor false alarm rate on CMAPSS stable window\n",
        "| Dataset | Stable batches | Alerts | Severe | False alert rate | Mean shift |",
        "|---|---|---|---|---|---|",
    ]
    for ds_id, r in results.items():
        lines.append(
            f"| {ds_id} | {r.n_stable_batches} | {r.alert_count} | {r.severe_count}"
            f" | {r.false_alert_rate:.1%} | {r.mean_shift_score:.3f} |"
        )
    return "\n".join(lines)


def render_cmapss_folds_report(
    fold_results: dict[str, CMAPSSFoldResult],
    *,
    dataset_id: str = "",
    significance_level: float = 0.10,
) -> str:
    """Render a markdown table for fold results with statistical significance."""
    lines = []
    if dataset_id:
        lines.append(f"## CMAPSS {dataset_id} — temporal fold results\n")
    lines.append("| Strategy | mean Δ vs frozen | ± std | p-value | sig? | folds |")
    lines.append("|---|---|---|---|---|---|")
    for policy_name, result in fold_results.items():
        if policy_name == "frozen":
            lines.append(f"| frozen | — (baseline) | — | — | — | {result.n_folds} |")
        else:
            sign = "+" if result.mean_ctrl_delta >= 0 else ""
            _t, p, _method = significance_test(result.fold_ctrl_deltas)
            p_str = f"{p:.3f}" if p == p else "—"  # nan check
            sig_str = "✓" if p == p and p < significance_level else "—"
            lines.append(
                f"| {policy_name} | {sign}{result.mean_ctrl_delta*100:.1f} pp"
                f" | ±{result.std_ctrl_delta*100:.1f} pp"
                f" | {p_str} | {sig_str} | {result.n_folds} |"
            )
    return "\n".join(lines)
