"""Orchestrate analysis, table creation, figures, and console reporting."""

import numpy as np
import pandas as pd
import seaborn as sns

from .config import DATASETS, FIGURE_DIR, RESULT_DIR
from .detection import OutlierAnalysis, calculate_outliers, load_dataset
from .plotting import save_individual_figure, save_merged_figure


def summary_row(analysis: OutlierAnalysis) -> dict[str, object]:
    """Build one dataset-level summary record."""

    row: dict[str, object] = {
        "Dataset": analysis.spec.name,
        "Rows": len(analysis.flag),
        "VariablesUsed": len(analysis.spec.variables),
        "DistanceCutoff": analysis.cutoff,
        "Outliers": int(analysis.flag.sum()),
        "OutlierPercent": 100 * float(analysis.flag.mean()),
    }
    if analysis.known is not None:
        row.update(
            {
                "KnownOutliers": int(analysis.known.sum()),
                "KnownDetected": int(
                    np.logical_and(analysis.flag, analysis.known).sum()
                ),
                "AdditionalFlags": int(
                    np.logical_and(analysis.flag, ~analysis.known).sum()
                ),
                "KnownMissed": int(
                    np.logical_and(~analysis.flag, analysis.known).sum()
                ),
            }
        )
    return row


def run() -> pd.DataFrame:
    """Run every configured dataset and return the summary table."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    analyses: list[OutlierAnalysis] = []
    summary_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        analysis = calculate_outliers(spec, load_dataset(spec))
        analyses.append(analysis)
        summary_rows.append(summary_row(analysis))
        analysis.result.to_csv(
            RESULT_DIR / f"{spec.slug}_outliers.csv",
            index=False,
            float_format="%.6f",
        )
        save_individual_figure(analysis)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        RESULT_DIR / "outlier_summary.csv",
        index=False,
        float_format="%.4f",
    )
    save_merged_figure(analyses)
    print_summary(summary)
    return summary


def main() -> None:
    """Console-script entry point."""

    run()


def print_summary(summary: pd.DataFrame) -> None:
    """Print compact results and output locations."""

    display = summary.copy()
    display["DistanceCutoff"] = display["DistanceCutoff"].round(3)
    display["OutlierPercent"] = display["OutlierPercent"].round(1)
    print(display.fillna("-").to_string(index=False))
    print(f"\nFigures: {FIGURE_DIR}")
    print(f"Outlier tables: {RESULT_DIR}")
