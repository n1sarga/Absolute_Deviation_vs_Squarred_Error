"""Leakage-safe repeated cross-validation for OLS and LAD."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from .baseline import regression_metrics
from .comparison import SAMPLE_METRICS, pair_model_results
from .config import CV_REPEATS, CV_SPLITS, RANDOM_STATE, DatasetSpec
from .robustness_conditions import fit_models, predict


@dataclass(frozen=True)
class CrossValidationResult:
    """Out-of-fold predictions and repeat-level metrics for one dataset."""

    spec: DatasetSpec
    predictions: pd.DataFrame
    repeat_metrics: pd.DataFrame


def run_repeated_cross_validation(
    spec: DatasetSpec,
    data: pd.DataFrame,
    n_splits: int = CV_SPLITS,
    n_repeats: int = CV_REPEATS,
    random_state: int = RANDOM_STATE,
) -> CrossValidationResult:
    """Fit on training folds and predict untouched test folds."""

    if n_splits < 2 or n_repeats < 1:
        raise ValueError("cross-validation needs at least 2 folds and 1 repeat")
    if len(data) < n_splits:
        raise ValueError("cross-validation folds cannot exceed row count")

    predictors = data[list(spec.predictors)].to_numpy(dtype=float)
    response = data[spec.response].to_numpy(dtype=float)
    record_ids = (
        data[spec.id_column].to_numpy()
        if spec.id_column
        else np.arange(1, len(data) + 1)
    )
    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    prediction_rows: list[dict[str, object]] = []

    for split_number, (train_indices, test_indices) in enumerate(
        splitter.split(predictors)
    ):
        repeat = split_number // n_splits + 1
        fold = split_number % n_splits + 1
        models = fit_models(
            predictors[train_indices],
            response[train_indices],
        )
        model_predictions = {
            model_name: predict(fit, predictors[test_indices])
            for model_name, fit in models.items()
        }
        for position, row_index in enumerate(test_indices):
            ols_prediction = model_predictions["OLS"][position]
            lad_prediction = model_predictions["LAD"][position]
            prediction_rows.append(
                {
                    "Dataset": spec.name,
                    "Repeat": repeat,
                    "Fold": fold,
                    "RecordID": record_ids[row_index],
                    "Actual": response[row_index],
                    "OLSPrediction": ols_prediction,
                    "LADPrediction": lad_prediction,
                    "OLSResidual": response[row_index] - ols_prediction,
                    "LADResidual": response[row_index] - lad_prediction,
                    "OLSAbsoluteError": abs(response[row_index] - ols_prediction),
                    "LADAbsoluteError": abs(response[row_index] - lad_prediction),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    repeat_rows: list[dict[str, object]] = []
    for repeat, repeat_data in predictions.groupby("Repeat", sort=True):
        actual = repeat_data["Actual"].to_numpy(dtype=float)
        for model_name, prediction_column in (
            ("OLS", "OLSPrediction"),
            ("LAD", "LADPrediction"),
        ):
            repeat_rows.append(
                {
                    "Dataset": spec.name,
                    "Repeat": repeat,
                    "Rows": len(actual),
                    "Predictors": len(spec.predictors),
                    "Model": model_name,
                    **regression_metrics(
                        actual,
                        repeat_data[prediction_column].to_numpy(dtype=float),
                    ),
                }
            )

    return CrossValidationResult(
        spec=spec,
        predictions=predictions,
        repeat_metrics=pd.DataFrame(repeat_rows),
    )


def validate_prediction_coverage(
    predictions: pd.DataFrame,
    expected_repeats: int = CV_REPEATS,
) -> None:
    """Verify each observation appears exactly once within every repeat."""

    counts = predictions.groupby(["Repeat", "RecordID"]).size()
    repeats = predictions["Repeat"].nunique()
    row_counts = predictions.groupby("Repeat")["RecordID"].nunique()
    id_sets = predictions.groupby("Repeat")["RecordID"].agg(
        lambda values: frozenset(values)
    )
    if not counts.eq(1).all():
        raise ValueError("test observations must appear once per CV repeat")
    if (
        repeats != expected_repeats
        or row_counts.nunique() != 1
        or id_sets.nunique() != 1
    ):
        raise ValueError("cross-validation prediction coverage is incomplete")


def summarize_cross_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated out-of-fold metrics for each model."""

    return (
        metrics.groupby(["Dataset", "Model"], sort=False)
        .agg(
            Repeats=("Repeat", "nunique"),
            Rows=("Rows", "first"),
            Predictors=("Predictors", "first"),
            MAEMean=("MAE", "mean"),
            MAESD=("MAE", "std"),
            RMSEMean=("RMSE", "mean"),
            RMSESD=("RMSE", "std"),
            MedianAbsoluteErrorMean=("MedianAbsoluteError", "mean"),
            MedianAbsoluteErrorSD=("MedianAbsoluteError", "std"),
            R2Mean=("R2", "mean"),
            R2SD=("R2", "std"),
        )
        .reset_index()
    )


def compare_cross_validation_repeats(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair OLS and LAD metrics within each repeated partition."""

    return pair_model_results(
        metrics,
        key_columns=["Dataset", "Repeat", "Rows", "Predictors"],
        metric_columns=SAMPLE_METRICS,
    )
