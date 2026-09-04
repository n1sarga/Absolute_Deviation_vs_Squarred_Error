"""Tests for LP-based least-absolute-deviation regression."""

import numpy as np
import pytest

from outlier_analysis.config import DATASETS
from outlier_analysis.detection import load_dataset
from outlier_analysis.lad import fit_lad_linear_program
from outlier_analysis.lad_validation import validate_lad_for_dataset


def test_lad_recovers_exact_line():
    predictors = np.arange(-5.0, 6.0).reshape(-1, 1)
    response = 2.0 + 3.0 * predictors[:, 0]

    fit = fit_lad_linear_program(predictors, response)

    assert fit.intercept == pytest.approx(2.0, abs=1e-9)
    assert fit.coefficients == pytest.approx([3.0], abs=1e-9)
    assert fit.absolute_error_sum == pytest.approx(0.0, abs=1e-8)


def test_lad_resists_single_vertical_outlier():
    predictors = np.arange(-5.0, 6.0).reshape(-1, 1)
    response = 2.0 + 3.0 * predictors[:, 0]
    response[5] += 100.0

    fit = fit_lad_linear_program(predictors, response)

    assert fit.intercept == pytest.approx(2.0, abs=1e-9)
    assert fit.coefficients == pytest.approx([3.0], abs=1e-9)
    assert fit.absolute_error_sum == pytest.approx(100.0, abs=1e-8)


@pytest.mark.parametrize("spec", DATASETS, ids=lambda spec: spec.slug)
def test_custom_lad_matches_reference_objective(spec):
    validation = validate_lad_for_dataset(spec, load_dataset(spec))

    assert validation.objective_match
    assert validation.objective_relative_difference < 1e-8


@pytest.mark.parametrize(
    ("predictors", "response", "message"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "two-dimensional"),
        (np.ones((2, 1)), np.ones((2, 1)), "one-dimensional"),
        (np.ones((2, 1)), np.ones(3), "equal row counts"),
        (np.array([[1.0], [np.nan]]), np.ones(2), "finite"),
    ],
)
def test_lad_rejects_invalid_inputs(predictors, response, message):
    with pytest.raises(ValueError, match=message):
        fit_lad_linear_program(predictors, response)
