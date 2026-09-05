"""Tests for repeated cross-validation and paired inference."""

import numpy as np
import pandas as pd
import pytest

from outlier_analysis.config import DatasetSpec
from outlier_analysis.cross_validation import (
    run_repeated_cross_validation,
    validate_prediction_coverage,
)
from outlier_analysis.inference import (
    average_out_of_fold_predictions,
    bootstrap_metric_differences,
    holm_adjust,
    paired_randomization_p_value,
)


def test_repeated_cross_validation_has_complete_test_coverage(tmp_path):
    predictors = np.arange(30.0)
    data = pd.DataFrame(
        {
            "ID": np.arange(1, 31),
            "X": predictors,
            "y": 2.0 + 3.0 * predictors,
        }
    )
    spec = DatasetSpec(
        name="Exact line",
        slug="exact_line",
        path=tmp_path / "unused.csv",
        predictors=("X",),
        response="y",
        id_column="ID",
    )

    result = run_repeated_cross_validation(
        spec,
        data,
        n_splits=5,
        n_repeats=2,
        random_state=10,
    )
    validate_prediction_coverage(result.predictions, expected_repeats=2)

    assert len(result.predictions) == 60
    assert len(result.repeat_metrics) == 4
    assert result.predictions["OLSAbsoluteError"].max() < 1e-10
    assert result.predictions["LADAbsoluteError"].max() < 1e-8


def test_average_predictions_preserves_pairing():
    predictions = pd.DataFrame(
        {
            "Dataset": ["D"] * 4,
            "Repeat": [1, 2, 1, 2],
            "RecordID": [1, 1, 2, 2],
            "Actual": [5.0, 5.0, 8.0, 8.0],
            "OLSPrediction": [4.0, 6.0, 7.0, 9.0],
            "LADPrediction": [5.0, 5.0, 8.0, 8.0],
        }
    )

    averaged = average_out_of_fold_predictions(predictions)

    assert len(averaged) == 2
    assert (averaged["CVRepeats"] == 2).all()
    assert averaged["OLSPrediction"].to_numpy() == pytest.approx([5.0, 8.0])
    assert averaged["LADPrediction"].to_numpy() == pytest.approx([5.0, 8.0])


def test_average_predictions_rejects_changed_response():
    predictions = pd.DataFrame(
        {
            "Dataset": ["D", "D"],
            "Repeat": [1, 2],
            "RecordID": [1, 1],
            "Actual": [5.0, 6.0],
            "OLSPrediction": [4.0, 4.0],
            "LADPrediction": [5.0, 5.0],
        }
    )

    with pytest.raises(ValueError, match="response changed"):
        average_out_of_fold_predictions(predictions)


def test_paired_bootstrap_recovers_constant_mae_advantage():
    actual = np.arange(20.0)
    ols_predictions = actual + 2.0
    lad_predictions = actual + 1.0

    intervals = bootstrap_metric_differences(
        actual,
        ols_predictions,
        lad_predictions,
        repetitions=200,
        random_state=42,
    )

    estimate, lower, upper = intervals["MAE"]
    assert estimate == pytest.approx(-1.0)
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(-1.0)


def test_holm_adjustment_matches_known_example():
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_paired_randomization_detects_large_consistent_difference():
    differences = -np.ones(20)

    p_value = paired_randomization_p_value(
        differences,
        repetitions=2000,
        random_state=42,
    )

    assert p_value < 0.01


@pytest.mark.parametrize(
    "p_values",
    [np.array([-0.1, 0.2]), np.array([0.2, np.nan])],
)
def test_holm_rejects_invalid_p_values(p_values):
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust(p_values)
