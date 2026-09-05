"""Run formal OLS-versus-LAD evaluation from the repository root."""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from outlier_analysis.evaluation import main  # noqa: E402


if __name__ == "__main__":
    main()
