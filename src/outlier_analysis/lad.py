"""Least-absolute-deviation regression solved as a linear program."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack


@dataclass(frozen=True)
class RegressionScaling:
    """Robust location and scale values used to condition the linear program."""

    predictor_center: np.ndarray
    predictor_scale: np.ndarray
    response_center: float
    response_scale: float

    def transform_predictors(self, predictors: np.ndarray) -> np.ndarray:
        """Transform predictors to robust-scaled coordinates."""

        return (predictors - self.predictor_center) / self.predictor_scale

    def transform_response(self, response: np.ndarray) -> np.ndarray:
        """Transform response to robust-scaled coordinates."""

        return (response - self.response_center) / self.response_scale

    def restore_parameters(
        self,
        scaled_intercept: float,
        scaled_coefficients: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Convert scaled regression parameters to original data units."""

        coefficients = (
            self.response_scale * scaled_coefficients / self.predictor_scale
        )
        intercept = (
            self.response_center
            + self.response_scale * scaled_intercept
            - float(coefficients @ self.predictor_center)
        )
        return intercept, coefficients


@dataclass(frozen=True)
class LADFit:
    """Fitted LAD parameters and linear-program diagnostics."""

    intercept: float
    coefficients: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    absolute_error_sum: float
    iterations: int
    solver_status: int
    solver_message: str
    scaling: RegressionScaling


def validate_regression_inputs(
    predictors: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite two-dimensional X and one-dimensional y arrays."""

    x = np.asarray(predictors, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.ndim != 2:
        raise ValueError("predictors must be a two-dimensional array")
    if y.ndim != 1:
        raise ValueError("response must be a one-dimensional array")
    if len(x) != len(y):
        raise ValueError("predictors and response must have equal row counts")
    if len(x) == 0 or x.shape[1] == 0:
        raise ValueError("regression data must contain rows and predictors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("regression data must contain only finite values")
    return x, y


def robust_regression_scaling(
    predictors: np.ndarray,
    response: np.ndarray,
) -> RegressionScaling:
    """Estimate median/IQR scaling, using one for any constant column."""

    predictor_center = np.median(predictors, axis=0)
    predictor_scale = np.subtract(
        *np.percentile(predictors, [75, 25], axis=0)
    )
    predictor_scale = np.where(
        predictor_scale > np.finfo(float).eps,
        predictor_scale,
        1.0,
    )

    response_center = float(np.median(response))
    response_q1, response_q3 = np.percentile(response, [25, 75])
    response_scale = float(response_q3 - response_q1)
    if response_scale <= np.finfo(float).eps:
        response_scale = 1.0

    return RegressionScaling(
        predictor_center=predictor_center,
        predictor_scale=predictor_scale,
        response_center=response_center,
        response_scale=response_scale,
    )


def fit_lad_linear_program(
    predictors: np.ndarray,
    response: np.ndarray,
) -> LADFit:
    """Fit LAD by minimizing residual magnitudes with SciPy HiGHS.

    Variables are the unrestricted intercept/coefficients followed by one
    nonnegative absolute-residual bound per observation. For design matrix A,
    the constraints are A beta - u <= y and -A beta - u <= -y.
    """

    x, y = validate_regression_inputs(predictors, response)
    scaling = robust_regression_scaling(x, y)
    scaled_x = scaling.transform_predictors(x)
    scaled_y = scaling.transform_response(y)

    design = np.column_stack((np.ones(len(scaled_x)), scaled_x))
    parameter_count = design.shape[1]
    observation_count = len(design)

    sparse_design = csr_matrix(design)
    negative_identity = -eye(observation_count, format="csr")
    upper_constraints = hstack((sparse_design, negative_identity), format="csr")
    lower_constraints = hstack((-sparse_design, negative_identity), format="csr")
    constraint_matrix = vstack(
        (upper_constraints, lower_constraints),
        format="csr",
    )
    constraint_bounds = np.concatenate((scaled_y, -scaled_y))

    objective = np.concatenate(
        (np.zeros(parameter_count), np.ones(observation_count))
    )
    variable_bounds = (
        [(None, None)] * parameter_count
        + [(0.0, None)] * observation_count
    )

    solution = linprog(
        objective,
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=variable_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not solution.success:
        raise RuntimeError(
            f"LAD linear program failed ({solution.status}): {solution.message}"
        )

    scaled_parameters = solution.x[:parameter_count]
    intercept, coefficients = scaling.restore_parameters(
        float(scaled_parameters[0]),
        scaled_parameters[1:],
    )
    predictions = intercept + x @ coefficients
    residuals = y - predictions

    return LADFit(
        intercept=intercept,
        coefficients=coefficients,
        predictions=predictions,
        residuals=residuals,
        absolute_error_sum=float(np.abs(residuals).sum()),
        iterations=int(solution.nit),
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        scaling=scaling,
    )
