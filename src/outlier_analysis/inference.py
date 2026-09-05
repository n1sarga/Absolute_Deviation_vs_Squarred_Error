"""Paired bootstrap intervals and multiplicity-adjusted MAE tests."""

import numpy as np
import pandas as pd

from .config import (
    BOOTSTRAP_REPETITIONS,
    CONFIDENCE_LEVEL,
    RANDOM_STATE,
    RANDOMIZATION_REPETITIONS,
    SIGNIFICANCE_LEVEL,
)


def average_out_of_fold_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average each observation's out-of-fold predictions across repeats."""

    required = {
        "Dataset",
        "Repeat",
        "RecordID",
        "Actual",
        "OLSPrediction",
        "LADPrediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction data missing columns: {sorted(missing)}")
    actual_counts = predictions.groupby(["Dataset", "RecordID"])[
        "Actual"
    ].nunique()
    if not actual_counts.eq(1).all():
        raise ValueError("actual response changed across CV repeats")

    return (
        predictions.groupby(["Dataset", "RecordID"], sort=False)
        .agg(
            Actual=("Actual", "first"),
            CVRepeats=("Repeat", "nunique"),
            OLSPrediction=("OLSPrediction", "mean"),
            LADPrediction=("LADPrediction", "mean"),
        )
        .reset_index()
    )


def bootstrap_metric_differences(
    actual: np.ndarray,
    ols_predictions: np.ndarray,
    lad_predictions: np.ndarray,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    confidence_level: float = CONFIDENCE_LEVEL,
    random_state: int | np.random.SeedSequence = RANDOM_STATE,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap LAD-minus-OLS MAE, RMSE, and median-AE differences."""

    actual = np.asarray(actual, dtype=float)
    ols_predictions = np.asarray(ols_predictions, dtype=float)
    lad_predictions = np.asarray(lad_predictions, dtype=float)
    if not (
        actual.ndim == ols_predictions.ndim == lad_predictions.ndim == 1
        and len(actual) == len(ols_predictions) == len(lad_predictions)
        and len(actual) > 1
    ):
        raise ValueError("bootstrap arrays must be equal nontrivial vectors")
    if repetitions < 2 or not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap settings are invalid")

    ols_absolute = np.abs(actual - ols_predictions)
    lad_absolute = np.abs(actual - lad_predictions)
    ols_squared = np.square(actual - ols_predictions)
    lad_squared = np.square(actual - lad_predictions)
    observed = {
        "MAE": float(lad_absolute.mean() - ols_absolute.mean()),
        "RMSE": float(
            np.sqrt(lad_squared.mean()) - np.sqrt(ols_squared.mean())
        ),
        "MedianAbsoluteError": float(
            np.median(lad_absolute) - np.median(ols_absolute)
        ),
    }
    bootstrapped = {
        metric: np.empty(repetitions, dtype=float) for metric in observed
    }
    rng = np.random.default_rng(random_state)
    chunk_size = min(250, repetitions)

    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(
            0,
            len(actual),
            size=(stop - start, len(actual)),
        )
        bootstrapped["MAE"][start:stop] = (
            lad_absolute[indices].mean(axis=1)
            - ols_absolute[indices].mean(axis=1)
        )
        bootstrapped["RMSE"][start:stop] = (
            np.sqrt(lad_squared[indices].mean(axis=1))
            - np.sqrt(ols_squared[indices].mean(axis=1))
        )
        bootstrapped["MedianAbsoluteError"][start:stop] = (
            np.median(lad_absolute[indices], axis=1)
            - np.median(ols_absolute[indices], axis=1)
        )

    alpha = 1.0 - confidence_level
    return {
        metric: (
            estimate,
            float(np.quantile(bootstrapped[metric], alpha / 2.0)),
            float(np.quantile(bootstrapped[metric], 1.0 - alpha / 2.0)),
        )
        for metric, estimate in observed.items()
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm step-down family-wise adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be a finite vector")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must lie between zero and one")

    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def paired_randomization_p_value(
    paired_differences: np.ndarray,
    repetitions: int = RANDOMIZATION_REPETITIONS,
    random_state: int | np.random.SeedSequence = RANDOM_STATE,
) -> float:
    """Test zero paired mean by randomly swapping model labels within rows."""

    differences = np.asarray(paired_differences, dtype=float)
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError("paired differences must be a nontrivial vector")
    if not np.isfinite(differences).all() or repetitions < 1:
        raise ValueError("randomization settings are invalid")
    if np.allclose(differences, 0.0):
        return 1.0

    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(random_state)
    extreme = 0
    chunk_size = min(500, repetitions)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        signs = rng.choice(
            (-1.0, 1.0),
            size=(stop - start, len(differences)),
        )
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int((permuted >= observed - 1e-15).sum())
    return (extreme + 1.0) / (repetitions + 1.0)


def build_paired_inference(
    averaged_predictions: pd.DataFrame,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    confidence_level: float = CONFIDENCE_LEVEL,
    randomization_repetitions: int = RANDOMIZATION_REPETITIONS,
) -> pd.DataFrame:
    """Build dataset-level bootstrap intervals and paired MAE randomization tests."""

    rows: list[dict[str, object]] = []
    for dataset_number, (dataset, group) in enumerate(
        averaged_predictions.groupby("Dataset", sort=False),
        start=1,
    ):
        actual = group["Actual"].to_numpy(dtype=float)
        ols_predictions = group["OLSPrediction"].to_numpy(dtype=float)
        lad_predictions = group["LADPrediction"].to_numpy(dtype=float)
        ols_absolute = np.abs(actual - ols_predictions)
        lad_absolute = np.abs(actual - lad_predictions)
        paired_loss_difference = lad_absolute - ols_absolute
        intervals = bootstrap_metric_differences(
            actual,
            ols_predictions,
            lad_predictions,
            repetitions=repetitions,
            confidence_level=confidence_level,
            random_state=np.random.SeedSequence(
                [RANDOM_STATE, dataset_number, 6]
            ),
        )
        p_value = paired_randomization_p_value(
            paired_loss_difference,
            repetitions=randomization_repetitions,
            random_state=np.random.SeedSequence(
                [RANDOM_STATE, dataset_number, 7]
            ),
        )

        row: dict[str, object] = {
            "Dataset": dataset,
            "Rows": len(group),
            "CVRepeats": int(group["CVRepeats"].iloc[0]),
            "BootstrapRepetitions": repetitions,
            "RandomizationRepetitions": randomization_repetitions,
            "ConfidenceLevel": confidence_level,
            "OLSMAE": float(ols_absolute.mean()),
            "LADMAE": float(lad_absolute.mean()),
            "MAEReductionPercent": float(
                100.0 * (ols_absolute.mean() - lad_absolute.mean())
                / ols_absolute.mean()
            ),
            "PairedRandomizationPValue": p_value,
        }
        for metric, (estimate, lower, upper) in intervals.items():
            row[f"LADMinusOLS{metric}"] = estimate
            row[f"{metric}CILower"] = lower
            row[f"{metric}CIUpper"] = upper
        rows.append(row)

    inference = pd.DataFrame(rows)
    inference["MAEHolmPValue"] = holm_adjust(
        inference["PairedRandomizationPValue"].to_numpy(dtype=float)
    )
    inference["RejectEqualMAEHolm"] = (
        inference["MAEHolmPValue"] < SIGNIFICANCE_LEVEL
    )
    inference["PreferredByMAE"] = np.select(
        [
            inference["RejectEqualMAEHolm"]
            & inference["LADMinusOLSMAE"].lt(0.0),
            inference["RejectEqualMAEHolm"]
            & inference["LADMinusOLSMAE"].gt(0.0),
        ],
        ["LAD", "OLS"],
        default="No clear difference",
    )
    return inference
