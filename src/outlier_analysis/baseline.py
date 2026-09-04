"""Fit full-data OLS and LAD baselines for every configured dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATASETS, RESULT_DIR, DatasetSpec
from .detection import load_dataset
from .lad import LADFit, fit_lad_linear_program
from .ols import OLSFit, fit_ordinary_least_squares


@dataclass(frozen=True)
class BaselineComparison:
    """Full-data OLS and LAD fits for one dataset."""

    spec: DatasetSpec
    response: np.ndarray
    record_ids: np.ndarray
    ols: OLSFit
    lad: LADFit


def fit_baseline_models(
    spec: DatasetSpec,
    data: pd.DataFrame,
) -> BaselineComparison:
    """Fit both objectives to one complete dataset."""

    predictors = data[list(spec.predictors)].to_numpy(dtype=float)
    response = data[spec.response].to_numpy(dtype=float)
    record_ids = (
        data[spec.id_column].to_numpy()
        if spec.id_column
        else np.arange(1, len(data) + 1)
    )
    return BaselineComparison(
        spec=spec,
        response=response,
        record_ids=record_ids,
        ols=fit_ordinary_least_squares(predictors, response),
        lad=fit_lad_linear_program(predictors, response),
    )


def regression_metrics(
    response: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate in-sample residual metrics in response units."""

    residuals = response - predictions
    absolute = np.abs(residuals)
    squared_error_sum = float(np.square(residuals).sum())
    total_sum_squares = float(np.square(response - response.mean()).sum())
    r_squared = (
        1.0 - squared_error_sum / total_sum_squares
        if total_sum_squares > np.finfo(float).eps
        else float("nan")
    )
    return {
        "SAE": float(absolute.sum()),
        "SSE": squared_error_sum,
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.square(residuals).mean())),
        "MedianAbsoluteError": float(np.median(absolute)),
        "R2": r_squared,
    }


def metric_rows(comparison: BaselineComparison) -> list[dict[str, object]]:
    """Build tidy OLS and LAD metric rows."""

    rows: list[dict[str, object]] = []
    for model_name, predictions in (
        ("OLS", comparison.ols.predictions),
        ("LAD", comparison.lad.predictions),
    ):
        rows.append(
            {
                "Dataset": comparison.spec.name,
                "Model": model_name,
                "Rows": len(comparison.response),
                "Predictors": len(comparison.spec.predictors),
                **regression_metrics(comparison.response, predictions),
            }
        )
    return rows


def optimality_row(comparison: BaselineComparison) -> dict[str, object]:
    """Verify each model minimizes its defining in-sample loss."""

    ols_metrics = regression_metrics(
        comparison.response,
        comparison.ols.predictions,
    )
    lad_metrics = regression_metrics(
        comparison.response,
        comparison.lad.predictions,
    )
    tolerance = 1e-8
    return {
        "Dataset": comparison.spec.name,
        "OLSSSE": ols_metrics["SSE"],
        "LADSSE": lad_metrics["SSE"],
        "OLSSAE": ols_metrics["SAE"],
        "LADSAE": lad_metrics["SAE"],
        "OLSMinimizesSSE": ols_metrics["SSE"] <= lad_metrics["SSE"] + tolerance,
        "LADMinimizesSAE": lad_metrics["SAE"] <= ols_metrics["SAE"] + tolerance,
    }


def coefficient_rows(comparison: BaselineComparison) -> list[dict[str, object]]:
    """Build original-unit and robust-standardized coefficient rows."""

    rows: list[dict[str, object]] = []
    for model_name, fit in (("OLS", comparison.ols), ("LAD", comparison.lad)):
        scaled_coefficients = (
            fit.coefficients
            * fit.scaling.predictor_scale
            / fit.scaling.response_scale
        )
        rows.append(
            {
                "Dataset": comparison.spec.name,
                "Model": model_name,
                "Term": "Intercept",
                "Estimate": fit.intercept,
                "RobustStandardizedEstimate": float("nan"),
            }
        )
        rows.extend(
            {
                "Dataset": comparison.spec.name,
                "Model": model_name,
                "Term": predictor,
                "Estimate": estimate,
                "RobustStandardizedEstimate": scaled_estimate,
            }
            for predictor, estimate, scaled_estimate in zip(
                comparison.spec.predictors,
                fit.coefficients,
                scaled_coefficients,
            )
        )
    return rows


def prediction_table(comparison: BaselineComparison) -> pd.DataFrame:
    """Build row-level actual values, predictions, and residuals."""

    return pd.DataFrame(
        {
            "RecordID": comparison.record_ids,
            "Actual": comparison.response,
            "OLSPrediction": comparison.ols.predictions,
            "LADPrediction": comparison.lad.predictions,
            "OLSResidual": comparison.ols.residuals,
            "LADResidual": comparison.lad.residuals,
        }
    )


def run_baseline_models() -> pd.DataFrame:
    """Fit all baselines and save tables and figures."""

    from .baseline_plotting import (
        save_baseline_figure,
        save_merged_baseline_figure,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons: list[BaselineComparison] = []
    metrics: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    optimality: list[dict[str, object]] = []

    for spec in DATASETS:
        comparison = fit_baseline_models(spec, load_dataset(spec))
        comparisons.append(comparison)
        metrics.extend(metric_rows(comparison))
        coefficients.extend(coefficient_rows(comparison))
        optimality.append(optimality_row(comparison))
        prediction_table(comparison).to_csv(
            RESULT_DIR / f"{spec.slug}_baseline_predictions.csv",
            index=False,
            float_format="%.10f",
        )
        save_baseline_figure(comparison)

    metric_table = pd.DataFrame(metrics)
    coefficient_table = pd.DataFrame(coefficients)
    optimality_table = pd.DataFrame(optimality)
    metric_table.to_csv(
        RESULT_DIR / "baseline_model_metrics.csv",
        index=False,
        float_format="%.10g",
    )
    coefficient_table.to_csv(
        RESULT_DIR / "baseline_model_coefficients.csv",
        index=False,
        float_format="%.10g",
    )
    optimality_table.to_csv(
        RESULT_DIR / "baseline_optimality_checks.csv",
        index=False,
        float_format="%.10g",
    )
    save_merged_baseline_figure(comparisons)

    print(metric_table.to_string(index=False))
    print("\nObjective checks:")
    print(optimality_table.to_string(index=False))
    return metric_table


def main() -> None:
    """Console-script entry point."""

    run_baseline_models()
