from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.wilds_civilcomments_benchmark import (
    render_wilds_civilcomments_report,
    run_wilds_civilcomments_benchmark,
    wilds_civilcomments_benchmark_to_dict,
)


def main() -> None:
    result = run_wilds_civilcomments_benchmark()
    report = render_wilds_civilcomments_report(result)
    print(report)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "wilds_civilcomments_benchmark.json"
    md_path = results_dir / "wilds_civilcomments_benchmark.md"
    json_path.write_text(json.dumps(wilds_civilcomments_benchmark_to_dict(result), indent=2), encoding="utf-8")
    md_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
