"""Run final cross-validation, inference, and research summary."""

import pandas as pd

from .config import DATASETS, RESULT_DIR
from .cross_validation import (
    compare_cross_validation_repeats,
    run_repeated_cross_validation,
    summarize_cross_validation,
    validate_prediction_coverage,
)
from .detection import load_dataset
from .inference import (
    average_out_of_fold_predictions,
    build_paired_inference,
)


def _load_stage5_findings() -> pd.DataFrame:
    """Load robustness/runtime results included in final synthesis."""

    path = RESULT_DIR / "stage5_key_findings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Required result missing: {path}. Run Step 5 evaluation first."
        )
    return pd.read_csv(path)


def build_final_findings(
    inference: pd.DataFrame,
    stage5: pd.DataFrame,
) -> pd.DataFrame:
    """Combine out-of-fold inference with contamination and runtime effects."""

    stage5_columns = [
        "Dataset",
        "MAEReductionPercent",
        "PredictionDriftReductionPercent",
        "SlopeShiftReductionPercent",
        "LADRuntimeMultiple",
    ]
    stage5_subset = stage5[stage5_columns].rename(
        columns={
            "MAEReductionPercent": "ContaminationMAEReductionPercent",
        }
    )
    return inference.merge(
        stage5_subset,
        on="Dataset",
        validate="one_to_one",
    )


def run_final_analysis() -> pd.DataFrame:
    """Run repeated CV, paired inference, plots, and final result table."""

    from .final_plotting import (
        save_bootstrap_forest_figure,
        save_cross_validation_figure,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_tables: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []

    for spec in DATASETS:
        result = run_repeated_cross_validation(spec, load_dataset(spec))
        validate_prediction_coverage(result.predictions)
        prediction_tables.append(result.predictions)
        metric_tables.append(result.repeat_metrics)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    repeat_metrics = pd.concat(metric_tables, ignore_index=True)
    summary = summarize_cross_validation(repeat_metrics)
    repeat_comparison = compare_cross_validation_repeats(repeat_metrics)
    averaged_predictions = average_out_of_fold_predictions(predictions)
    inference = build_paired_inference(averaged_predictions)
    final_findings = build_final_findings(
        inference,
        _load_stage5_findings(),
    )

    tables = {
        "cross_validation_predictions.csv": predictions,
        "cross_validation_average_predictions.csv": averaged_predictions,
        "cross_validation_repeat_metrics.csv": repeat_metrics,
        "cross_validation_summary.csv": summary,
        "cross_validation_paired_comparison.csv": repeat_comparison,
        "bootstrap_inference.csv": inference,
        "final_research_findings.csv": final_findings,
    }
    for filename, table in tables.items():
        table.to_csv(
            RESULT_DIR / filename,
            index=False,
            float_format="%.10g",
        )

    save_cross_validation_figure(repeat_metrics)
    save_bootstrap_forest_figure(inference)

    print("Final out-of-fold inference:")
    print(
        inference[
            [
                "Dataset",
                "OLSMAE",
                "LADMAE",
                "LADMinusOLSMAE",
                "MAECILower",
                "MAECIUpper",
                "MAEHolmPValue",
                "PreferredByMAE",
            ]
        ].to_string(index=False)
    )
    return final_findings


def main() -> None:
    """Console-script entry point."""

    run_final_analysis()
