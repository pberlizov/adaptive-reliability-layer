from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.recurrence_temporal_benchmark import (
    recurrence_temporal_benchmark_to_dict,
    render_recurrence_temporal_benchmark_report,
    run_recurrence_temporal_benchmark,
)


def main() -> None:
    result = run_recurrence_temporal_benchmark()
    report = render_recurrence_temporal_benchmark_report(result)
    print(report)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "recurrence_temporal_benchmark.json"
    md_path = results_dir / "recurrence_temporal_benchmark.md"
    json_path.write_text(json.dumps(recurrence_temporal_benchmark_to_dict(result), indent=2), encoding="utf-8")
    md_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
