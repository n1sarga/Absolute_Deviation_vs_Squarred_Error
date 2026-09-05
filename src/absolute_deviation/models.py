from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog


@dataclass
class RegressionResult:
    model: str
    intercept: float
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    objective: float
    runtime_seconds: float
    success: bool = True
    status: str = "ok"


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be one- or two-dimensional")
    return X


def _as_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    return y


def _design(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    X = _as_2d(X)
    if fit_intercept:
        return np.column_stack([np.ones(X.shape[0]), X])
    return X


def fit_ols(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> RegressionResult:
    """Ordinary least squares: minimize the sum of squared residuals."""
    X = _as_2d(X)
    y = _as_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of observations")
    A = _design(X, fit_intercept)
    start = perf_counter()
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    runtime = perf_counter() - start
    fitted = A @ beta
    residuals = y - fitted
    intercept = float(beta[0]) if fit_intercept else 0.0
    coefficients = beta[1:].copy() if fit_intercept else beta.copy()
    return RegressionResult(
        model="OLS",
        intercept=intercept,
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        objective=float(np.dot(residuals, residuals)),
        runtime_seconds=runtime,
    )


def fit_lad(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> RegressionResult:
    """Least absolute deviations via the Wagner-style linear-programming formulation.

    Residuals are decomposed as y - X beta = e_plus - e_minus with
    e_plus, e_minus >= 0, and the LP minimizes sum(e_plus + e_minus).
    """
    X = _as_2d(X)
    y = _as_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of observations")
    A = _design(X, fit_intercept)
    n, p = A.shape
    c = np.concatenate([np.zeros(p), np.ones(n), np.ones(n)])
    A_eq = np.hstack([A, np.eye(n), -np.eye(n)])
    b_eq = y
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
    start = perf_counter()
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not result.success:
        return RegressionResult(
            model="LAD",
            intercept=np.nan,
            coefficients=np.full(X.shape[1], np.nan),
            fitted=np.full_like(y, np.nan),
            residuals=np.full_like(y, np.nan),
            objective=np.nan,
            runtime_seconds=runtime,
            success=False,
            status=result.message,
        )
    beta = result.x[:p]
    fitted = A @ beta
    residuals = y - fitted
    intercept = float(beta[0]) if fit_intercept else 0.0
    coefficients = beta[1:].copy() if fit_intercept else beta.copy()
    return RegressionResult(
        model="LAD",
        intercept=intercept,
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        objective=float(np.abs(residuals).sum()),
        runtime_seconds=runtime,
        success=True,
        status=result.message,
    )


def regression_metrics(y: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    y = _as_1d(y)
    fitted = _as_1d(fitted)
    residuals = y - fitted
    return {
        "SSE": float(np.dot(residuals, residuals)),
        "SAE": float(np.abs(residuals).sum()),
        "MAE": float(np.abs(residuals).mean()),
        "RMSE": float(np.sqrt(np.mean(residuals**2))),
    }
