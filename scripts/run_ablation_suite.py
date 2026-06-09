from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.ablation_suite import (
    ablation_suite_to_dict,
    render_ablation_suite_report,
    run_ablation_suite,
)


def main() -> None:
    result = run_ablation_suite()
    report = render_ablation_suite_report(result)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "ablation_suite.json"
    md_path = results_dir / "ablation_suite.md"
    json_path.write_text(json.dumps(ablation_suite_to_dict(result), indent=2), encoding="utf-8")
    md_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print()
    print(f"saved_json={json_path}")
    print(f"saved_markdown={md_path}")


if __name__ == "__main__":
    main()
