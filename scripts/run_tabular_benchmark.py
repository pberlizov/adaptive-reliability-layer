from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.tabular_benchmark import render_tabular_benchmark_report, run_tabular_benchmark


def main() -> None:
    result = run_tabular_benchmark()
    print(render_tabular_benchmark_report(result))


if __name__ == "__main__":
    main()
