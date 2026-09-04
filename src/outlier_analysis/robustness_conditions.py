"""Fit OLS and LAD under full-data and inlier-only conditions."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .baseline import regression_metrics
from .classification import classify_regression_outliers
from .config import DatasetSpec
from .lad import LADFit, fit_lad_linear_program
from .ols import OLSFit, fit_ordinary_least_squares


RegressionFit = OLSFit | LADFit
ModelFitter = Callable[[np.ndarray, np.ndarray], RegressionFit]
MODEL_FITTERS: dict[str, ModelFitter] = {
    "OLS": fit_ordinary_least_squares,
    "LAD": fit_lad_linear_program,
}


@dataclass(frozen=True)
class ConditionFits:
    """Model fits and fixed arrays for one dataset's observed conditions."""

    spec: DatasetSpec
    predictors: np.ndarray
    response: np.ndarray
    record_ids: np.ndarray
    regular_mask: np.ndarray
    full_models: dict[str, RegressionFit]
    inlier_models: dict[str, RegressionFit]

    @property
    def inlier_predictors(self) -> np.ndarray:
        """Predictor rows classified as regression-regular."""

        return self.predictors[self.regular_mask]

    @property
    def inlier_response(self) -> np.ndarray:
        """Response rows classified as regression-regular."""

        return self.response[self.regular_mask]

    @property
    def inlier_record_ids(self) -> np.ndarray:
        """Identifiers of rows classified as regression-regular."""

        return self.record_ids[self.regular_mask]


def fit_models(
    predictors: np.ndarray,
    response: np.ndarray,
) -> dict[str, RegressionFit]:
    """Fit OLS and LAD to the same arrays."""

    return {
        model_name: fitter(predictors, response)
        for model_name, fitter in MODEL_FITTERS.items()
    }


def predict(fit: RegressionFit, predictors: np.ndarray) -> np.ndarray:
    """Predict in original response units from either fitted model."""

    return fit.intercept + predictors @ fit.coefficients


def prepare_condition_fits(
    spec: DatasetSpec,
    data: pd.DataFrame,
) -> ConditionFits:
    """Fit both models to full data and regression-regular rows."""

    classification = classify_regression_outliers(spec, data)
    regular_mask = classification.category == "Regular"
    predictors = data[list(spec.predictors)].to_numpy(dtype=float)
    response = data[spec.response].to_numpy(dtype=float)
    record_ids = (
        data[spec.id_column].to_numpy()
        if spec.id_column
        else np.arange(1, len(data) + 1)
    )
    if regular_mask.sum() <= len(spec.predictors):
        raise ValueError(
            f"{spec.name} has too few regular rows for regression fitting"
        )

    return ConditionFits(
        spec=spec,
        predictors=predictors,
        response=response,
        record_ids=record_ids,
        regular_mask=regular_mask,
        full_models=fit_models(predictors, response),
        inlier_models=fit_models(
            predictors[regular_mask],
            response[regular_mask],
        ),
    )


def condition_metric_rows(condition: ConditionFits) -> list[dict[str, object]]:
    """Evaluate full-data and inlier-only fits on both observed samples."""

    rows: list[dict[str, object]] = []
    evaluation_sets = {
        "Full data": (condition.predictors, condition.response),
        "Inliers only": (
            condition.inlier_predictors,
            condition.inlier_response,
        ),
    }
    fit_sets = {
        "Full data": condition.full_models,
        "Inliers only": condition.inlier_models,
    }

    for fit_condition, models in fit_sets.items():
        train_rows = (
            len(condition.response)
            if fit_condition == "Full data"
            else len(condition.inlier_response)
        )
        for evaluation_set, (predictors, response) in evaluation_sets.items():
            for model_name, fit in models.items():
                rows.append(
                    {
                        "Dataset": condition.spec.name,
                        "FitCondition": fit_condition,
                        "EvaluationSet": evaluation_set,
                        "Model": model_name,
                        "TrainingRows": train_rows,
                        "EvaluationRows": len(response),
                        **regression_metrics(response, predict(fit, predictors)),
                    }
                )
    return rows


def condition_coefficient_rows(
    condition: ConditionFits,
) -> list[dict[str, object]]:
    """Return coefficients for full-data and inlier-only fits."""

    rows: list[dict[str, object]] = []
    for fit_condition, models in (
        ("Full data", condition.full_models),
        ("Inliers only", condition.inlier_models),
    ):
        for model_name, fit in models.items():
            rows.append(
                {
                    "Dataset": condition.spec.name,
                    "FitCondition": fit_condition,
                    "Model": model_name,
                    "Term": "Intercept",
                    "Estimate": fit.intercept,
                }
            )
            rows.extend(
                {
                    "Dataset": condition.spec.name,
                    "FitCondition": fit_condition,
                    "Model": model_name,
                    "Term": term,
                    "Estimate": estimate,
                }
                for term, estimate in zip(
                    condition.spec.predictors,
                    fit.coefficients,
                )
            )
    return rows
