"""Ordinary least-squares regression solved with a stable least-squares system."""

from dataclasses import dataclass

import numpy as np

from .lad import (
    RegressionScaling,
    robust_regression_scaling,
    validate_regression_inputs,
)


@dataclass(frozen=True)
class OLSFit:
    """Fitted OLS parameters, residuals, and matrix diagnostics."""

    intercept: float
    coefficients: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    squared_error_sum: float
    rank: int
    singular_values: np.ndarray
    scaling: RegressionScaling


def fit_ordinary_least_squares(
    predictors: np.ndarray,
    response: np.ndarray,
) -> OLSFit:
    """Fit OLS after robust scaling, then restore original data units."""

    x, y = validate_regression_inputs(predictors, response)
    scaling = robust_regression_scaling(x, y)
    scaled_x = scaling.transform_predictors(x)
    scaled_y = scaling.transform_response(y)
    design = np.column_stack((np.ones(len(scaled_x)), scaled_x))

    parameters, _, rank, singular_values = np.linalg.lstsq(
        design,
        scaled_y,
        rcond=None,
    )
    intercept, coefficients = scaling.restore_parameters(
        float(parameters[0]),
        parameters[1:],
    )
    predictions = intercept + x @ coefficients
    residuals = y - predictions

    return OLSFit(
        intercept=intercept,
        coefficients=coefficients,
        predictions=predictions,
        residuals=residuals,
        squared_error_sum=float(np.square(residuals).sum()),
        rank=int(rank),
        singular_values=singular_values,
        scaling=scaling,
    )
