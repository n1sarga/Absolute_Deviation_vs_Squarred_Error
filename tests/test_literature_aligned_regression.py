import numpy as np

from absolute_deviation.data import DATASETS
from absolute_deviation.models import fit_lad, fit_ols, regression_metrics


def test_only_three_empirical_datasets_are_active():
    assert set(DATASETS) == {"boston_housing", "concrete_strength", "hbk"}


def test_ols_minimizes_sse_against_lad():
    X = np.arange(8.0).reshape(-1, 1)
    y = np.array([1, 3, 5, 7, 9, 11, 13, 35], dtype=float)
    ols = fit_ols(X, y)
    lad = fit_lad(X, y)
    assert regression_metrics(y, ols.fitted)["SSE"] <= regression_metrics(y, lad.fitted)["SSE"] + 1e-8


def test_lad_minimizes_sae_against_ols():
    X = np.arange(8.0).reshape(-1, 1)
    y = np.array([1, 3, 5, 7, 9, 11, 13, 35], dtype=float)
    ols = fit_ols(X, y)
    lad = fit_lad(X, y)
    assert regression_metrics(y, lad.fitted)["SAE"] <= regression_metrics(y, ols.fitted)["SAE"] + 1e-8


def test_lad_exact_line_with_intercept():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 + 4.0 * X[:, 0]
    lad = fit_lad(X, y)
    assert lad.success
    assert np.isclose(lad.intercept, 2.0, atol=1e-8)
    assert np.allclose(lad.coefficients, [4.0], atol=1e-8)
    assert np.isclose(np.abs(lad.residuals).sum(), 0.0, atol=1e-8)


def test_lad_multiple_predictors():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2]], dtype=float)
    y = 1 + 2 * X[:, 0] - 3 * X[:, 1]
    lad = fit_lad(X, y)
    assert lad.success
    assert np.allclose(lad.fitted, y, atol=1e-8)
