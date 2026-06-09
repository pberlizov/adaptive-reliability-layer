from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .failure_analysis import (
    ProductionFailureAnalysisReport,
    SourceFailureAnalysis,
    VariantAnalysis,
    run_production_failure_analysis,
)


@dataclass(frozen=True)
class CorrectionWinCriteria:
    min_mean_utility_delta_vs_frozen: float = 0.03
    min_mean_utility_delta_vs_scheduled: float = 0.03
    max_mean_utility_gap_vs_full: float = 0.003
    max_worst_source_utility_gap_vs_full: float = 0.01
    min_sources_within_full_gap: int = 2
    max_mean_revealed_burden_regression_vs_full: float = 0.002
    max_mean_explicit_action_rate: float = 0.01


@dataclass(frozen=True)
class CorrectionPathSourceSummary:
    source_id: str
    description: str
    full_utility: float
    candidate_utility: float
    utility_delta_vs_frozen: float
    utility_delta_vs_scheduled: float
    utility_gap_vs_full: float
    revealed_burden_delta_vs_full: float | None
    explicit_action_rate: float


@dataclass(frozen=True)
class CorrectionPathCandidateEvaluation:
    key: str
    label: str
    source_summaries: tuple[CorrectionPathSourceSummary, ...]
    mean_utility_delta_vs_frozen: float
    mean_utility_delta_vs_scheduled: float
    mean_utility_gap_vs_full: float
    worst_source_utility_gap_vs_full: float
    sources_within_full_gap: int
    mean_revealed_burden_regression_vs_full: float | None
    mean_explicit_action_rate: float
    checks: tuple[tuple[str, bool, str], ...]
    passed: bool


@dataclass(frozen=True)
class CorrectionPathEvaluationReport:
    config_path: str
    controller_name: str
    criteria: CorrectionWinCriteria
    failure_analysis: ProductionFailureAnalysisReport
    candidates: tuple[CorrectionPathCandidateEvaluation, ...]


def _burden_delta(candidate: VariantAnalysis, full: VariantAnalysis) -> float | None:
    if candidate.revealed_burden is None or full.revealed_burden is None:
        return None
    return candidate.revealed_burden - full.revealed_burden


def _source_summary(source: SourceFailureAnalysis, *, candidate_key: str) -> CorrectionPathSourceSummary:
    frozen = source.variants["frozen"]
    scheduled = source.variants["scheduled_retrain"]
    full = source.variants["full"]
    candidate = source.variants[candidate_key]
    return CorrectionPathSourceSummary(
        source_id=source.source_id,
        description=source.description,
        full_utility=full.mean_utility,
        candidate_utility=candidate.mean_utility,
        utility_delta_vs_frozen=candidate.mean_utility - frozen.mean_utility,
        utility_delta_vs_scheduled=candidate.mean_utility - scheduled.mean_utility,
        utility_gap_vs_full=full.mean_utility - candidate.mean_utility,
        revealed_burden_delta_vs_full=_burden_delta(candidate, full),
        explicit_action_rate=candidate.explicit_action_rate,
    )


def _evaluate_candidate(
    *,
    report: ProductionFailureAnalysisReport,
    candidate_key: str,
    label: str,
    criteria: CorrectionWinCriteria,
) -> CorrectionPathCandidateEvaluation:
    source_summaries = tuple(_source_summary(source, candidate_key=candidate_key) for source in report.sources)
    mean_vs_frozen = mean(item.utility_delta_vs_frozen for item in source_summaries)
    mean_vs_scheduled = mean(item.utility_delta_vs_scheduled for item in source_summaries)
    mean_gap_vs_full = mean(item.utility_gap_vs_full for item in source_summaries)
    worst_gap_vs_full = max(item.utility_gap_vs_full for item in source_summaries)
    within_full_gap = sum(item.utility_gap_vs_full <= criteria.max_worst_source_utility_gap_vs_full for item in source_summaries)
    burden_deltas = [item.revealed_burden_delta_vs_full for item in source_summaries if item.revealed_burden_delta_vs_full is not None]
    mean_burden_regression = mean(burden_deltas) if burden_deltas else None
    mean_explicit_action_rate = mean(item.explicit_action_rate for item in source_summaries)

    checks = (
        (
            "mean_utility_vs_frozen",
            mean_vs_frozen >= criteria.min_mean_utility_delta_vs_frozen,
            f"{mean_vs_frozen:+.3f} (need >= {criteria.min_mean_utility_delta_vs_frozen:+.3f})",
        ),
        (
            "mean_utility_vs_scheduled",
            mean_vs_scheduled >= criteria.min_mean_utility_delta_vs_scheduled,
            f"{mean_vs_scheduled:+.3f} (need >= {criteria.min_mean_utility_delta_vs_scheduled:+.3f})",
        ),
        (
            "mean_gap_vs_full",
            mean_gap_vs_full <= criteria.max_mean_utility_gap_vs_full,
            f"{mean_gap_vs_full:+.3f} (need <= {criteria.max_mean_utility_gap_vs_full:+.3f})",
        ),
        (
            "worst_gap_vs_full",
            worst_gap_vs_full <= criteria.max_worst_source_utility_gap_vs_full,
            f"{worst_gap_vs_full:+.3f} (need <= {criteria.max_worst_source_utility_gap_vs_full:+.3f})",
        ),
        (
            "source_coverage_within_full_gap",
            within_full_gap >= criteria.min_sources_within_full_gap,
            f"{within_full_gap} (need >= {criteria.min_sources_within_full_gap})",
        ),
        (
            "mean_explicit_action_rate",
            mean_explicit_action_rate <= criteria.max_mean_explicit_action_rate,
            f"{mean_explicit_action_rate:.3f} (need <= {criteria.max_mean_explicit_action_rate:.3f})",
        ),
    )
    if mean_burden_regression is not None:
        checks += (
            (
                "mean_revealed_burden_regression_vs_full",
                mean_burden_regression <= criteria.max_mean_revealed_burden_regression_vs_full,
                f"{mean_burden_regression:+.3f} (need <= {criteria.max_mean_revealed_burden_regression_vs_full:+.3f})",
            ),
        )

    passed = all(item[1] for item in checks)
    return CorrectionPathCandidateEvaluation(
        key=candidate_key,
        label=label,
        source_summaries=source_summaries,
        mean_utility_delta_vs_frozen=mean_vs_frozen,
        mean_utility_delta_vs_scheduled=mean_vs_scheduled,
        mean_utility_gap_vs_full=mean_gap_vs_full,
        worst_source_utility_gap_vs_full=worst_gap_vs_full,
        sources_within_full_gap=within_full_gap,
        mean_revealed_burden_regression_vs_full=mean_burden_regression,
        mean_explicit_action_rate=mean_explicit_action_rate,
        checks=checks,
        passed=passed,
    )


def run_correction_path_evaluation(
    *,
    config_path: str | Path = "configs/production_benchmark_sota_suite.yaml",
    source_ids: tuple[str, ...] | None = None,
    criteria: CorrectionWinCriteria = CorrectionWinCriteria(),
) -> CorrectionPathEvaluationReport:
    failure_report = run_production_failure_analysis(config_path=config_path, source_ids=source_ids)
    candidates = (
        _evaluate_candidate(
            report=failure_report,
            candidate_key="correction_only",
            label="correction_only",
            criteria=criteria,
        ),
        _evaluate_candidate(
            report=failure_report,
            candidate_key="no_explicit_actions",
            label="correction_plus_governor",
            criteria=criteria,
        ),
    )
    return CorrectionPathEvaluationReport(
        config_path=str(config_path),
        controller_name=failure_report.controller_name,
        criteria=criteria,
        failure_analysis=failure_report,
        candidates=candidates,
    )


def correction_path_evaluation_to_dict(report: CorrectionPathEvaluationReport) -> dict[str, Any]:
    return {
        "config_path": report.config_path,
        "controller_name": report.controller_name,
        "criteria": asdict(report.criteria),
        "failure_analysis": {
            "config_path": report.failure_analysis.config_path,
            "controller_name": report.failure_analysis.controller_name,
            "baseline_strategies": list(report.failure_analysis.baseline_strategies),
            "sources": [asdict(source) for source in report.failure_analysis.sources],
        },
        "candidates": [asdict(candidate) for candidate in report.candidates],
    }


def render_correction_path_evaluation(report: CorrectionPathEvaluationReport) -> str:
    lines = [
        "# Correction-Centric Parallel Path Evaluation",
        "",
        f"Controller under test: `{report.controller_name}`",
        f"Config: `{report.config_path}`",
        "",
        "## Win Criteria",
        "",
        f"- mean utility delta vs frozen >= `{report.criteria.min_mean_utility_delta_vs_frozen:+.3f}`",
        f"- mean utility delta vs scheduled retrain >= `{report.criteria.min_mean_utility_delta_vs_scheduled:+.3f}`",
        f"- mean utility gap vs full hybrid <= `{report.criteria.max_mean_utility_gap_vs_full:+.3f}`",
        f"- worst-source utility gap vs full hybrid <= `{report.criteria.max_worst_source_utility_gap_vs_full:+.3f}`",
        f"- sources within full-gap threshold >= `{report.criteria.min_sources_within_full_gap}`",
        f"- mean revealed-burden regression vs full hybrid <= `{report.criteria.max_mean_revealed_burden_regression_vs_full:+.3f}`",
        f"- mean explicit action rate <= `{report.criteria.max_mean_explicit_action_rate:.3f}`",
        "",
    ]
    for candidate in report.candidates:
        lines.extend(
            [
                f"## {candidate.label}",
                f"parallel-path verdict: `{'PASS' if candidate.passed else 'FAIL'}`",
                "",
                f"- mean utility delta vs frozen: `{candidate.mean_utility_delta_vs_frozen:+.3f}`",
                f"- mean utility delta vs scheduled retrain: `{candidate.mean_utility_delta_vs_scheduled:+.3f}`",
                f"- mean utility gap vs full hybrid: `{candidate.mean_utility_gap_vs_full:+.3f}`",
                f"- worst-source utility gap vs full hybrid: `{candidate.worst_source_utility_gap_vs_full:+.3f}`",
                f"- sources within full-gap threshold: `{candidate.sources_within_full_gap}`",
                f"- mean revealed-burden regression vs full hybrid: "
                f"`{candidate.mean_revealed_burden_regression_vs_full:+.3f}`"
                if candidate.mean_revealed_burden_regression_vs_full is not None
                else "- mean revealed-burden regression vs full hybrid: `n/a`",
                f"- mean explicit action rate: `{candidate.mean_explicit_action_rate:.3f}`",
                "",
                "| check | passed | detail |",
                "| --- | --- | --- |",
            ]
        )
        for name, passed, detail in candidate.checks:
            lines.append(f"| {name} | {'yes' if passed else 'no'} | {detail} |")
        lines.extend(
            [
                "",
                "| source | cand util | full util | Δ vs frozen | Δ vs scheduled | gap vs full | burden Δ vs full | explicit |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source in candidate.source_summaries:
            burden = "n/a" if source.revealed_burden_delta_vs_full is None else f"{source.revealed_burden_delta_vs_full:+.3f}"
            lines.append(
                f"| {source.source_id} | {source.candidate_utility:.3f} | {source.full_utility:.3f} | "
                f"{source.utility_delta_vs_frozen:+.3f} | {source.utility_delta_vs_scheduled:+.3f} | "
                f"{source.utility_gap_vs_full:+.3f} | {burden} | {source.explicit_action_rate:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_correction_path_evaluation(
    report: CorrectionPathEvaluationReport,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "correction_path_evaluation.md").write_text(
        render_correction_path_evaluation(report),
        encoding="utf-8",
    )
    (output / "correction_path_evaluation.json").write_text(
        json.dumps(correction_path_evaluation_to_dict(report), indent=2),
        encoding="utf-8",
    )
    return output / "correction_path_evaluation.md"
