from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.graph_shift_benchmark import (
    render_graph_shift_benchmark_report,
    run_graph_shift_benchmark,
)


def main() -> None:
    result = run_graph_shift_benchmark()
    print(render_graph_shift_benchmark_report(result))


if __name__ == "__main__":
    main()
