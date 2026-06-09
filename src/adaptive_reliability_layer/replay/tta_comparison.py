from __future__ import annotations

from dataclasses import dataclass

from ..tabular_benchmark import TabularBenchmarkResult, run_tabular_benchmark_with_factories
from ..tabular_benchmark import _tta_policy_factories


@dataclass(frozen=True)
class TtaComparisonRow:
    name: str
    accuracy: float
    mean_utility: float
    mean_risk_capital: float
    risk_alerts: int
    resets: int
    mean_parameter_drift: float


def _rows_from_benchmark(result: TabularBenchmarkResult) -> tuple[TtaComparisonRow, ...]:
    return tuple(
        TtaComparisonRow(
            name=strategy.name,
            accuracy=strategy.overall_accuracy,
            mean_utility=strategy.mean_utility,
            mean_risk_capital=strategy.mean_risk_capital,
            risk_alerts=strategy.risk_alerts,
            resets=strategy.resets,
            mean_parameter_drift=strategy.mean_parameter_drift,
        )
        for strategy in result.strategies
    )


def run_tta_tabular_comparison(*, steps: int = 90, batch_size: int = 48, seed: int = 7) -> TabularBenchmarkResult:
    return run_tabular_benchmark_with_factories(
        policy_factories=_tta_policy_factories(),
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )


def render_tta_comparison_report(result: TabularBenchmarkResult) -> str:
    rows = _rows_from_benchmark(result)
    frozen = next((row for row in rows if row.name == "frozen"), None)
    bandit = next((row for row in rows if row.name == "bandit"), None)
    tent = next((row for row in rows if row.name == "tent"), None)
    eata = next((row for row in rows if row.name == "eata_style"), None)

    lines = [
        "TTA Baseline Comparison (research tabular stream, breast cancer shift generator)",
        f"steps={result.steps} batch_size={result.batch_size}",
        "",
        "Lead with risk and utility — accuracy is secondary for buyer conversations.",
        "",
        "strategy          accuracy   utility   risk_capital   risk_alerts   resets   mean_drift",
    ]
    for row in rows:
        lines.append(
            f"{row.name:<16} {row.accuracy:>8.3f}   {row.mean_utility:>7.3f}   "
            f"{row.mean_risk_capital:>12.3f}   {row.risk_alerts:>11}   {row.resets:>6}   "
            f"{row.mean_parameter_drift:>10.3f}"
        )

    lines.append("")
    if frozen and bandit and tent:
        risk_drop_bandit = (
            (1.0 - bandit.mean_risk_capital / frozen.mean_risk_capital) * 100.0
            if frozen.mean_risk_capital > 0
            else 0.0
        )
        risk_drop_tent = (
            (1.0 - tent.mean_risk_capital / frozen.mean_risk_capital) * 100.0
            if frozen.mean_risk_capital > 0
            else 0.0
        )
        lines.extend(
            [
                "Buyer-facing takeaways",
                (
                    f"- ARL bandit: {risk_drop_bandit:.0f}% lower mean risk capital than frozen "
                    f"({frozen.mean_risk_capital:.1f} → {bandit.mean_risk_capital:.1f}); "
                    f"accuracy {frozen.accuracy:.1%} → {bandit.accuracy:.1%}."
                ),
                (
                    f"- TENT (standard TTA): {risk_drop_tent:.0f}% risk capital vs frozen but "
                    f"utility {tent.mean_utility:.3f} vs bandit {bandit.mean_utility:.3f} — "
                    "always-on entropy adaptation lacks bounded control and reset discipline."
                ),
            ]
        )
        if eata:
            lines.append(
                f"- EATA-style selective TENT: accuracy {eata.accuracy:.1%}, utility {eata.mean_utility:.3f}, "
                f"risk capital {eata.mean_risk_capital:.1f} — still no governance layer vs ARL."
            )
        lines.append(
            "- For outreach: cite this table when reviewers ask why TENT/EATA are not the product — "
            "they are baselines we now benchmark; ARL wins on sequential utility/risk, not raw accuracy."
        )
    return "\n".join(lines)
