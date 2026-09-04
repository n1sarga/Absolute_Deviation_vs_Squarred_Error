"""Run LP-based LAD solver validation from a source checkout."""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from outlier_analysis.lad_validation import main


if __name__ == "__main__":
    main()
