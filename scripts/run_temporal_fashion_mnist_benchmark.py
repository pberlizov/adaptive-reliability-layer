from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.temporal_fashion_mnist_benchmark import (
    render_temporal_fashion_mnist_report,
    run_temporal_fashion_mnist_benchmark,
)


def main() -> None:
    result = run_temporal_fashion_mnist_benchmark()
    print(render_temporal_fashion_mnist_report(result))


if __name__ == "__main__":
    main()
