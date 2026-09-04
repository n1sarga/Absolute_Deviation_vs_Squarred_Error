"""Orchestrate analysis, table creation, figures, and console reporting."""

import numpy as np
import pandas as pd
import seaborn as sns

from .classification import (
    RegressionOutlierClassification,
    classification_summary_row,
    classify_regression_outliers,
)
from .classification_plotting import (
    save_classification_figure,
    save_merged_classification_figure,
)
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
    classifications: list[RegressionOutlierClassification] = []
    classification_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        data = load_dataset(spec)
        analysis = calculate_outliers(spec, data)
        classification = classify_regression_outliers(spec, data)
        analyses.append(analysis)
        summary_rows.append(summary_row(analysis))
        classifications.append(classification)
        classification_rows.append(classification_summary_row(classification))
        analysis.result.to_csv(
            RESULT_DIR / f"{spec.slug}_outliers.csv",
            index=False,
            float_format="%.6f",
        )
        classification.result.to_csv(
            RESULT_DIR / f"{spec.slug}_outlier_classification.csv",
            index=False,
            float_format="%.6f",
        )
        save_individual_figure(analysis)
        save_classification_figure(classification)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        RESULT_DIR / "outlier_summary.csv",
        index=False,
        float_format="%.4f",
    )
    classification_summary = pd.DataFrame(classification_rows)
    classification_summary.to_csv(
        RESULT_DIR / "outlier_classification_summary.csv",
        index=False,
        float_format="%.4f",
    )
    save_merged_figure(analyses)
    save_merged_classification_figure(classifications)
    print_summary(summary, classification_summary)
    return summary


def main() -> None:
    """Console-script entry point."""

    run()


def print_summary(
    summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
) -> None:
    """Print compact results and output locations."""

    display = summary.copy()
    display["DistanceCutoff"] = display["DistanceCutoff"].round(3)
    display["OutlierPercent"] = display["OutlierPercent"].round(1)
    print(display.fillna("-").to_string(index=False))
    print("\nRegression outlier classes:")
    class_display = classification_summary.copy()
    class_display["LeverageCutoff"] = class_display["LeverageCutoff"].round(3)
    print(class_display.fillna("-").to_string(index=False))
    print(f"\nFigures: {FIGURE_DIR}")
    print(f"Outlier tables: {RESULT_DIR}")
