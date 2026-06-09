"""Monitor quality evaluation benchmark.

Tests the TabularShiftMonitor (and optionally the MartingaleRiskMonitor)
independently of any controller by running them against synthetic streams
with known ground-truth drift onset times.

Metrics reported:
- False alarm rate (FAR): fraction of stable batches that fire an alarm
- Detection latency: batches between onset and first alarm (None if missed)
- Missed detection rate (MDR): fraction of trials with no alarm in drift window
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# TabularShiftMonitor and TabularReferenceProfile live in tabular_benchmark, which
# has a hard torch import at module level.  We attempt the clean import first; if
# torch is not installed we fall back to a local, torch-free re-implementation
# that is behaviourally identical.
try:
    from .tabular_benchmark import (  # noqa: F401 (re-exported for callers)
        TabularBatch,
        TabularReferenceProfile,
        TabularShiftMonitor,
    )
    _USING_NATIVE = True
except (ImportError, ModuleNotFoundError):
    _USING_NATIVE = False

    # --- local fallback dataclasses / monitor ---------------------------------
    import math as _math

    @dataclass(frozen=True)  # type: ignore[no-redef]
    class TabularReferenceProfile:  # type: ignore[no-redef]
        feature_mean: "np.ndarray"
        feature_variance: "np.ndarray"
        mean_entropy: float
        mean_probability: float
        positive_rate: float
        mean_confidence: float

    @dataclass(frozen=True)  # type: ignore[no-redef]
    class TabularBatch:  # type: ignore[no-redef]
        features: "np.ndarray"
        labels: "np.ndarray"
        regime: str

    def _be(p: float) -> float:
        p = max(min(p, 1.0 - 1e-6), 1e-6)
        return -(p * _math.log(p) + (1.0 - p) * _math.log(1.0 - p))

    class TabularShiftMonitor:  # type: ignore[no-redef]
        def __init__(
            self,
            reference: "TabularReferenceProfile",
            *,
            alert_threshold: float = 1.1,
            severe_threshold: float = 1.75,
        ) -> None:
            self._reference = reference
            self._alert_threshold = alert_threshold
            self._severe_threshold = severe_threshold

        def evaluate(self, features: "np.ndarray", probabilities: "list[float]") -> object:
            batch_mean = features.mean(axis=0)
            batch_variance = features.var(axis=0)
            normalized_mean_gap = float(np.mean(
                np.abs(batch_mean - self._reference.feature_mean)
                / np.sqrt(self._reference.feature_variance + 1e-6)
            ))
            normalized_variance_gap = float(np.mean(
                np.abs(batch_variance - self._reference.feature_variance)
                / (self._reference.feature_variance + 1e-6)
            ))
            feature_score = normalized_mean_gap + 0.5 * normalized_variance_gap

            mean_entropy = float(np.mean([_be(p) for p in probabilities]))
            mean_probability = float(np.mean(probabilities))
            positive_rate = float(np.mean([1.0 if p >= 0.5 else 0.0 for p in probabilities]))
            mean_confidence = float(np.mean([max(p, 1.0 - p) for p in probabilities]))

            entropy_gap = abs(mean_entropy - self._reference.mean_entropy)
            probability_gap = abs(mean_probability - self._reference.mean_probability)
            rate_gap = abs(positive_rate - self._reference.positive_rate)
            confidence_gap = abs(mean_confidence - self._reference.mean_confidence)
            output_score = entropy_gap + 0.75 * probability_gap + rate_gap + 0.5 * confidence_gap

            collapse_risk = (
                max(0.0, self._reference.mean_entropy - mean_entropy)
                + max(0.0, abs(positive_rate - 0.5) - abs(self._reference.positive_rate - 0.5))
            )

            score = feature_score + 0.75 * output_score + 0.65 * collapse_risk

            class _Signal:
                pass

            sig = _Signal()
            sig.score = score  # type: ignore[attr-defined]
            sig.alert = score >= self._alert_threshold  # type: ignore[attr-defined]
            sig.severe = score >= self._severe_threshold or collapse_risk >= 0.30  # type: ignore[attr-defined]
            sig.feature_score = feature_score  # type: ignore[attr-defined]
            sig.output_score = output_score  # type: ignore[attr-defined]
            sig.collapse_risk = collapse_risk  # type: ignore[attr-defined]
            sig.mean_entropy = mean_entropy  # type: ignore[attr-defined]
            sig.mean_probability = mean_probability  # type: ignore[attr-defined]
            sig.positive_rate = positive_rate  # type: ignore[attr-defined]
            sig.mean_confidence = mean_confidence  # type: ignore[attr-defined]
            return sig

from .risk import MartingaleRiskMonitor


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MonitorEvalConfig:
    n_stable_batches: int = 50
    n_drift_batches: int = 50
    batch_size: int = 200
    n_features: int = 10
    drift_magnitudes: list[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    n_trials: int = 10
    shift_types: list[str] = field(default_factory=lambda: ["abrupt", "gradual", "recurring"])
    alert_threshold: float = 1.1
    # How many of the first stable batches to use for building the reference profile
    n_reference_batches: int = 20


@dataclass
class MonitorEvalResult:
    trial: int
    shift_type: str
    drift_magnitude: float
    detector_name: str
    false_alarm_rate: float
    # Number of batches after onset before first alarm fires (None = missed)
    detection_latency_batches: Optional[int]
    missed_detection: bool


# ---------------------------------------------------------------------------
# Synthetic stream builders
# ---------------------------------------------------------------------------

def build_stable_stream(config: MonitorEvalConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Return n_stable_batches batches drawn from N(0, 1) in R^n_features."""
    return [
        rng.standard_normal((config.batch_size, config.n_features))
        for _ in range(config.n_stable_batches)
    ]


def build_drift_stream(
    config: MonitorEvalConfig,
    shift_type: str,
    drift_magnitude: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int]:
    """Return (batches, true_onset_step).

    onset_step is the index of the first batch that is no longer pure N(0,1).

    shift_type:
        "abrupt"    – stable then suddenly N(drift_magnitude, 1)
        "gradual"   – mean shifts linearly from 0 to drift_magnitude over 20 batches
        "recurring" – stable → drift → stable → drift  (two drift episodes)
    """
    n_stable = config.n_stable_batches
    n_drift = config.n_drift_batches
    n_feat = config.n_features
    bs = config.batch_size

    if shift_type == "abrupt":
        stable_batches = [rng.standard_normal((bs, n_feat)) for _ in range(n_stable)]
        drift_batches = [
            rng.standard_normal((bs, n_feat)) + drift_magnitude
            for _ in range(n_drift)
        ]
        batches = stable_batches + drift_batches
        onset_step = n_stable

    elif shift_type == "gradual":
        ramp_len = min(20, n_drift)
        stable_batches = [rng.standard_normal((bs, n_feat)) for _ in range(n_stable)]
        drift_batches = []
        for i in range(n_drift):
            progress = min(1.0, i / max(1, ramp_len - 1))
            current_mean = progress * drift_magnitude
            drift_batches.append(rng.standard_normal((bs, n_feat)) + current_mean)
        batches = stable_batches + drift_batches
        onset_step = n_stable  # first departure from 0 mean

    elif shift_type == "recurring":
        # Pattern: stable(n_stable/2) → drift(n_drift/2) → stable(n_stable/2) → drift(n_drift/2)
        half_stable = n_stable // 2
        half_drift = n_drift // 2
        seg1_stable = [rng.standard_normal((bs, n_feat)) for _ in range(half_stable)]
        seg1_drift = [rng.standard_normal((bs, n_feat)) + drift_magnitude for _ in range(half_drift)]
        seg2_stable = [rng.standard_normal((bs, n_feat)) for _ in range(half_stable)]
        seg2_drift = [rng.standard_normal((bs, n_feat)) + drift_magnitude for _ in range(half_drift)]
        batches = seg1_stable + seg1_drift + seg2_stable + seg2_drift
        onset_step = half_stable  # first drift episode onset

    else:
        raise ValueError(f"Unknown shift_type: {shift_type!r}")

    return batches, onset_step


# ---------------------------------------------------------------------------
# Reference profile builder (numpy-only, no torch required)
# ---------------------------------------------------------------------------

def _build_tabular_reference(
    reference_batches: list[np.ndarray],
) -> TabularReferenceProfile:
    """Build a TabularReferenceProfile from a list of feature-only batches.

    Since we have no real model, we synthesise plausible probability values
    using a fixed logistic transform of a random linear projection of the
    features.  The key thing is that the reference profile captures the
    stable-data statistics so the monitor can compare against them.
    """
    rng_ref = np.random.default_rng(seed=0)
    all_features = np.concatenate(reference_batches, axis=0)
    n_feat = all_features.shape[1]
    w = rng_ref.standard_normal(n_feat) * 0.3  # small weights → probs near 0.5

    logits = all_features @ w
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    feature_mean = all_features.mean(axis=0)
    feature_variance = all_features.var(axis=0) + 1e-6  # guard against zeros

    def _binary_entropy(p: float) -> float:
        p = max(min(p, 1.0 - 1e-6), 1e-6)
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    mean_entropy = float(np.mean([_binary_entropy(p) for p in probs]))
    mean_probability = float(np.mean(probs))
    positive_rate = float(np.mean(probs >= 0.5))
    mean_confidence = float(np.mean(np.maximum(probs, 1.0 - probs)))

    return TabularReferenceProfile(
        feature_mean=feature_mean,
        feature_variance=feature_variance,
        mean_entropy=mean_entropy,
        mean_probability=mean_probability,
        positive_rate=positive_rate,
        mean_confidence=mean_confidence,
    )


def _pseudo_probabilities(
    features: np.ndarray,
    w: np.ndarray,
) -> list[float]:
    """Produce pseudo-probabilities using the same projection as the reference."""
    logits = features @ w
    probs = 1.0 / (1.0 + np.exp(-logits))
    return list(np.clip(probs, 1e-6, 1.0 - 1e-6).tolist())


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_arl_monitor(
    batches: list[np.ndarray],
    onset_step: int,
    config: MonitorEvalConfig,
    trial: int,
    shift_type: str,
    drift_magnitude: float,
) -> MonitorEvalResult:
    """Evaluate the TabularShiftMonitor on a single stream.

    The reference profile is built from the first n_reference_batches stable
    batches (indices 0 .. n_reference_batches-1).  Then:
      - Batches [n_reference_batches, onset_step) are the *remaining stable*
        window used to measure false alarm rate.
      - Batches [onset_step, ...) are the drift window.

    Detection fires when signal.alert is True OR signal.score > alert_threshold.
    """
    n_ref = config.n_reference_batches
    reference_batches = batches[:n_ref]
    reference = _build_tabular_reference(reference_batches)

    monitor = TabularShiftMonitor(
        reference,
        alert_threshold=config.alert_threshold,
        severe_threshold=config.alert_threshold * 1.6,
    )

    # Fixed projection weights for pseudo-probabilities (same seed as reference)
    rng_ref = np.random.default_rng(seed=0)
    n_feat = batches[0].shape[1]
    w = rng_ref.standard_normal(n_feat) * 0.3

    # --- Stable window: batches [n_ref, onset_step) ---
    false_alarms = 0
    stable_count = 0
    for step in range(n_ref, onset_step):
        if step >= len(batches):
            break
        probs = _pseudo_probabilities(batches[step], w)
        signal = monitor.evaluate(batches[step], probs)
        stable_count += 1
        if signal.alert or signal.score > config.alert_threshold:
            false_alarms += 1

    false_alarm_rate = false_alarms / stable_count if stable_count > 0 else 0.0

    # --- Drift window: batches [onset_step, end) ---
    first_alarm_step: Optional[int] = None
    for step in range(onset_step, len(batches)):
        probs = _pseudo_probabilities(batches[step], w)
        signal = monitor.evaluate(batches[step], probs)
        if signal.alert or signal.score > config.alert_threshold:
            first_alarm_step = step
            break

    if first_alarm_step is None:
        detection_latency = None
        missed = True
    else:
        detection_latency = max(0, first_alarm_step - onset_step)
        missed = False

    return MonitorEvalResult(
        trial=trial,
        shift_type=shift_type,
        drift_magnitude=drift_magnitude,
        detector_name="TabularShiftMonitor",
        false_alarm_rate=false_alarm_rate,
        detection_latency_batches=detection_latency,
        missed_detection=missed,
    )


def _simple_linear_error_rate(
    batch: np.ndarray,
    w: np.ndarray,
    true_positive_rate: float = 0.5,
) -> float:
    """Pseudo-error rate: fraction of samples where logit disagrees with 0.5-threshold."""
    probs = np.asarray(_pseudo_probabilities(batch, w))
    predictions = (probs >= 0.5).astype(int)
    # Stable reference: assume half the samples are positive (balanced)
    expected = np.zeros(len(predictions), dtype=int)
    expected[: len(expected) // 2] = 1
    return float(np.mean(predictions != expected))


def evaluate_adwin_monitor(
    batches: list[np.ndarray],
    onset_step: int,
    config: MonitorEvalConfig,
    trial: int,
    shift_type: str,
    drift_magnitude: float,
) -> MonitorEvalResult | None:
    """Evaluate ADWIN drift detector (river.drift.ADWIN) on the same stream.

    Uses the running pseudo-error rate as the monitored signal.
    Returns None if river is not installed.
    """
    try:
        from river.drift import ADWIN as _ADWIN
    except ImportError:
        return None

    n_ref = config.n_reference_batches
    rng_ref = np.random.default_rng(seed=0)
    w = rng_ref.standard_normal(batches[0].shape[1]) * 0.3
    detector = _ADWIN(delta=0.002)

    # Warm up on reference batches
    for step in range(n_ref):
        for sample in batches[step]:
            prob = float(_pseudo_probabilities(sample.reshape(1, -1), w)[0])
            detector.update(int(prob < 0.5))

    # Stable window
    false_alarms = 0
    stable_count = 0
    for step in range(n_ref, onset_step):
        if step >= len(batches):
            break
        for sample in batches[step]:
            prob = float(_pseudo_probabilities(sample.reshape(1, -1), w)[0])
            if detector.update(int(prob < 0.5)):
                false_alarms += 1
                break  # one alarm per batch
        stable_count += 1

    false_alarm_rate = false_alarms / stable_count if stable_count > 0 else 0.0

    # Drift window — reset detector at onset and run fresh
    detector2 = _ADWIN(delta=0.002)
    first_alarm_step: Optional[int] = None
    for step in range(onset_step, len(batches)):
        for sample in batches[step]:
            prob = float(_pseudo_probabilities(sample.reshape(1, -1), w)[0])
            if detector2.update(int(prob < 0.5)):
                first_alarm_step = step
                break
        if first_alarm_step is not None:
            break

    if first_alarm_step is None:
        detection_latency = None
        missed = True
    else:
        detection_latency = max(0, first_alarm_step - onset_step)
        missed = False

    return MonitorEvalResult(
        trial=trial,
        shift_type=shift_type,
        drift_magnitude=drift_magnitude,
        detector_name="ADWIN",
        false_alarm_rate=false_alarm_rate,
        detection_latency_batches=detection_latency,
        missed_detection=missed,
    )


def evaluate_page_hinkley_monitor(
    batches: list[np.ndarray],
    onset_step: int,
    config: MonitorEvalConfig,
    trial: int,
    shift_type: str,
    drift_magnitude: float,
) -> MonitorEvalResult | None:
    """Evaluate Page-Hinkley drift detector (river.drift.PageHinkley).

    Returns None if river is not installed.
    """
    try:
        from river.drift import PageHinkley as _PageHinkley
    except ImportError:
        return None

    n_ref = config.n_reference_batches
    rng_ref = np.random.default_rng(seed=0)
    w = rng_ref.standard_normal(batches[0].shape[1]) * 0.3

    def _run_pht(batches_window: list[np.ndarray], onset: int) -> tuple[float, Optional[int], bool]:
        detector = _PageHinkley(min_instances=30, delta=0.005, threshold=50.0)
        false_alarms = 0
        stable_count = 0
        for step in range(n_ref, onset):
            if step >= len(batches_window):
                break
            batch_errors = [
                float(_pseudo_probabilities(s.reshape(1, -1), w)[0]) < 0.5
                for s in batches_window[step]
            ]
            for err in batch_errors:
                if detector.update(float(err)):
                    false_alarms += 1
                    break
            stable_count += 1
        far = false_alarms / stable_count if stable_count > 0 else 0.0

        detector2 = _PageHinkley(min_instances=30, delta=0.005, threshold=50.0)
        first_alarm: Optional[int] = None
        for step in range(onset, len(batches_window)):
            for s in batches_window[step]:
                err = float(_pseudo_probabilities(s.reshape(1, -1), w)[0]) < 0.5
                if detector2.update(float(err)):
                    first_alarm = step
                    break
            if first_alarm is not None:
                break

        missed = first_alarm is None
        latency = max(0, first_alarm - onset) if first_alarm is not None else None
        return far, latency, missed

    false_alarm_rate, detection_latency, missed = _run_pht(batches, onset_step)
    return MonitorEvalResult(
        trial=trial,
        shift_type=shift_type,
        drift_magnitude=drift_magnitude,
        detector_name="PageHinkley",
        false_alarm_rate=false_alarm_rate,
        detection_latency_batches=detection_latency,
        missed_detection=missed,
    )


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_monitor_eval(
    config: MonitorEvalConfig,
    *,
    include_baselines: bool = True,
) -> list[MonitorEvalResult]:
    """Run all (shift_type × drift_magnitude × trial) combinations.

    When ``include_baselines=True``, also runs ADWIN and Page-Hinkley if
    ``river`` is installed — allowing head-to-head comparison.
    """
    results: list[MonitorEvalResult] = []

    for shift_type in config.shift_types:
        for magnitude in config.drift_magnitudes:
            for trial in range(config.n_trials):
                rng = np.random.default_rng(seed=trial * 1000 + int(magnitude * 100))
                batches, onset_step = build_drift_stream(
                    config, shift_type, magnitude, rng
                )
                results.append(evaluate_arl_monitor(
                    batches, onset_step, config, trial, shift_type, magnitude
                ))
                if include_baselines:
                    adwin_result = evaluate_adwin_monitor(
                        batches, onset_step, config, trial, shift_type, magnitude
                    )
                    if adwin_result is not None:
                        results.append(adwin_result)
                    pht_result = evaluate_page_hinkley_monitor(
                        batches, onset_step, config, trial, shift_type, magnitude
                    )
                    if pht_result is not None:
                        results.append(pht_result)

    return results


# ---------------------------------------------------------------------------
# Report renderer
# ---------------------------------------------------------------------------

def render_monitor_eval_report(results: list[MonitorEvalResult]) -> str:
    """Render a markdown report from a list of MonitorEvalResult."""
    from collections import defaultdict

    # Group by (detector_name, shift_type, magnitude)
    GroupKey = tuple  # (detector_name, shift_type, magnitude)
    far_groups: dict[GroupKey, list[float]] = defaultdict(list)
    latency_groups: dict[GroupKey, list[Optional[int]]] = defaultdict(list)
    missed_groups: dict[GroupKey, list[bool]] = defaultdict(list)

    for r in results:
        key = (r.detector_name, r.shift_type, r.drift_magnitude)
        far_groups[key].append(r.false_alarm_rate)
        latency_groups[key].append(r.detection_latency_batches)
        missed_groups[key].append(r.missed_detection)

    lines: list[str] = ["## Monitor Evaluation Results", ""]

    # --- False alarm rate table ---
    lines.append("### False alarm rate (stable window)")
    lines.append("")
    lines.append("| detector | shift_type | magnitude | false_alarm_rate |")
    lines.append("| --- | --- | --- | --- |")
    for key in sorted(far_groups.keys()):
        detector, shift_type, magnitude = key
        mean_far = float(np.mean(far_groups[key]))
        lines.append(f"| {detector} | {shift_type} | {magnitude:.1f} | {mean_far:.4f} |")

    lines.append("")

    # --- Detection latency table ---
    lines.append("### Detection latency (batches after onset)")
    lines.append("")
    lines.append(
        "| detector | shift_type | magnitude | mean_latency | missed_detection_rate |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for key in sorted(latency_groups.keys()):
        detector, shift_type, magnitude = key
        latencies = latency_groups[key]
        missed = missed_groups[key]
        mdr = float(np.mean(missed))
        detected_latencies = [lat for lat in latencies if lat is not None]
        if detected_latencies:
            mean_lat = float(np.mean(detected_latencies))
            lat_str = f"{mean_lat:.2f}"
        else:
            lat_str = "N/A"
        lines.append(
            f"| {detector} | {shift_type} | {magnitude:.1f} | {lat_str} | {mdr:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)
