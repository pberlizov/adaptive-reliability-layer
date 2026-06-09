from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer import run_simulation


def main() -> None:
    result = run_simulation()
    print("Adaptive Reliability Layer Simulation")
    print(f"steps={result.steps}")
    print(f"accuracy={result.accuracy:.3f}")
    print(f"alerts={result.alerts}")
    print(f"adaptations={result.adaptations}")
    print(f"resets={result.resets}")


if __name__ == "__main__":
    main()
