"""Orchestrate OLS/LAD sample-condition and contamination experiments."""

import pandas as pd

from .config import DATASETS, RESULT_DIR
from .contamination import contamination_experiment, summarize_contamination
from .detection import load_dataset
from .robustness_conditions import (
    ConditionFits,
    condition_coefficient_rows,
    condition_metric_rows,
    prepare_condition_fits,
)


def run_robustness_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all condition and contamination experiments and save outputs."""

    from .robustness_plotting import (
        save_condition_contamination_figure,
        save_merged_condition_contamination_figure,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    conditions: list[ConditionFits] = []
    condition_metrics: list[dict[str, object]] = []
    condition_coefficients: list[dict[str, object]] = []
    contamination_metrics: list[pd.DataFrame] = []
    contamination_coefficients: list[pd.DataFrame] = []
    contamination_designs: list[pd.DataFrame] = []

    for dataset_number, spec in enumerate(DATASETS, start=1):
        condition = prepare_condition_fits(spec, load_dataset(spec))
        conditions.append(condition)
        condition_metrics.extend(condition_metric_rows(condition))
        condition_coefficients.extend(condition_coefficient_rows(condition))
        metrics, coefficients, design = contamination_experiment(
            condition,
            dataset_number,
        )
        contamination_metrics.append(metrics)
        contamination_coefficients.append(coefficients)
        contamination_designs.append(design)

    condition_metric_table = pd.DataFrame(condition_metrics)
    condition_coefficient_table = pd.DataFrame(condition_coefficients)
    contamination_metric_table = pd.concat(
        contamination_metrics,
        ignore_index=True,
    )
    contamination_coefficient_table = pd.concat(
        contamination_coefficients,
        ignore_index=True,
    )
    contamination_design_table = pd.concat(
        contamination_designs,
        ignore_index=True,
    )
    contamination_summary = summarize_contamination(
        contamination_metric_table
    )

    tables = {
        "sample_condition_metrics.csv": condition_metric_table,
        "sample_condition_coefficients.csv": condition_coefficient_table,
        "contamination_experiment_metrics.csv": contamination_metric_table,
        "contamination_experiment_coefficients.csv": (
            contamination_coefficient_table
        ),
        "contamination_experiment_design.csv": contamination_design_table,
        "contamination_experiment_summary.csv": contamination_summary,
    }
    for filename, table in tables.items():
        table.to_csv(
            RESULT_DIR / filename,
            index=False,
            float_format="%.10g",
        )

    for condition in conditions:
        save_condition_contamination_figure(
            condition.spec,
            condition_metric_table,
            contamination_summary,
        )
    save_merged_condition_contamination_figure(
        conditions,
        condition_metric_table,
        contamination_summary,
    )

    reported_conditions = condition_metric_table[
        condition_metric_table["EvaluationSet"].eq("Inliers only")
    ][
        ["Dataset", "FitCondition", "Model", "TrainingRows", "MAE", "RMSE"]
    ]
    print("Observed-condition fits evaluated on regular rows:")
    print(reported_conditions.to_string(index=False))
    print("\nControlled-contamination clean-data MAE:")
    print(
        contamination_summary[
            [
                "Dataset",
                "Model",
                "ContaminationPercent",
                "CleanMAEMean",
                "CleanMAEStd",
            ]
        ].to_string(index=False)
    )
    return condition_metric_table, contamination_summary


def main() -> None:
    """Console-script entry point."""

    run_robustness_experiments()
