from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.temporal_benchmark_suite import (
    render_temporal_benchmark_suite_report,
    run_temporal_benchmark_suite,
    temporal_benchmark_suite_to_dict,
)


def main() -> None:
    result = run_temporal_benchmark_suite()
    report = render_temporal_benchmark_suite_report(result)
    print(report)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "temporal_benchmark_suite.json"
    md_path = results_dir / "temporal_benchmark_suite.md"
    json_path.write_text(json.dumps(temporal_benchmark_suite_to_dict(result), indent=2), encoding="utf-8")
    md_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
