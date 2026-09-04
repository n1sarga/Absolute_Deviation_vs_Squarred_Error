"""Tests for full-data OLS and LAD baseline comparisons."""

import numpy as np
import pytest

from outlier_analysis.baseline import (
    fit_baseline_models,
    metric_rows,
    optimality_row,
    prediction_table,
)
from outlier_analysis.config import DATASETS
from outlier_analysis.detection import load_dataset
from outlier_analysis.ols import fit_ordinary_least_squares


def test_ols_recovers_exact_line():
    predictors = np.arange(-5.0, 6.0).reshape(-1, 1)
    response = 2.0 + 3.0 * predictors[:, 0]

    fit = fit_ordinary_least_squares(predictors, response)

    assert fit.intercept == pytest.approx(2.0, abs=1e-10)
    assert fit.coefficients == pytest.approx([3.0], abs=1e-10)
    assert fit.squared_error_sum == pytest.approx(0.0, abs=1e-20)
    assert fit.rank == 2


@pytest.mark.parametrize("spec", DATASETS, ids=lambda spec: spec.slug)
def test_full_data_objectives_have_expected_order(spec):
    comparison = fit_baseline_models(spec, load_dataset(spec))
    checks = optimality_row(comparison)

    assert checks["OLSMinimizesSSE"]
    assert checks["LADMinimizesSAE"]
    assert comparison.ols.predictions.shape == comparison.response.shape
    assert comparison.lad.predictions.shape == comparison.response.shape
    assert np.isfinite(comparison.ols.predictions).all()
    assert np.isfinite(comparison.lad.predictions).all()


@pytest.mark.parametrize("spec", DATASETS, ids=lambda spec: spec.slug)
def test_baseline_output_tables_are_complete(spec):
    comparison = fit_baseline_models(spec, load_dataset(spec))
    metrics = metric_rows(comparison)
    predictions = prediction_table(comparison)

    assert [row["Model"] for row in metrics] == ["OLS", "LAD"]
    assert len(predictions) == len(comparison.response)
    assert predictions.columns.tolist() == [
        "RecordID",
        "Actual",
        "OLSPrediction",
        "LADPrediction",
        "OLSResidual",
        "LADResidual",
    ]
