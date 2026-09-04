"""Source-checkout entry point for the multivariate outlier analysis.

Run from the project root:
    python scripts/analyze_outliers.py
"""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from outlier_analysis.pipeline import run


if __name__ == "__main__":
    run()
