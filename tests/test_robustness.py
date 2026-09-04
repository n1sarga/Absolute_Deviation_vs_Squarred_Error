"""Tests for observed-condition and contamination experiments."""

import numpy as np
import pytest

from outlier_analysis.contamination import (
    contaminate_response,
    contaminated_row_count,
)
from outlier_analysis.robustness_conditions import (
    fit_models,
    predict,
)


@pytest.mark.parametrize(
    ("rows", "rate", "expected"),
    [(100, 0.0, 0), (100, 0.05, 5), (11, 0.10, 1), (3, 0.20, 1)],
)
def test_contaminated_row_count(rows, rate, expected):
    assert contaminated_row_count(rows, rate) == expected


def test_contamination_is_reproducible_and_changes_only_selected_rows():
    response = np.arange(20.0)
    first = contaminate_response(
        response,
        rate=0.20,
        noise_sd=50.0,
        rng=np.random.default_rng(42),
    )
    second = contaminate_response(
        response,
        rate=0.20,
        noise_sd=50.0,
        rng=np.random.default_rng(42),
    )
    contaminated, indices, noise = first
    unchanged = np.ones(len(response), dtype=bool)
    unchanged[indices] = False

    assert len(indices) == 4
    assert len(np.unique(indices)) == 4
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert np.array_equal(contaminated[unchanged], response[unchanged])
    assert contaminated[indices] == pytest.approx(response[indices] + noise)


def test_lad_moves_less_than_ols_after_one_vertical_gross_error():
    predictors = np.arange(-10.0, 11.0).reshape(-1, 1)
    response = 2.0 + 3.0 * predictors[:, 0]
    clean_models = fit_models(predictors, response)
    contaminated = response.copy()
    contaminated[10] += 1000.0
    contaminated_models = fit_models(predictors, contaminated)

    drift = {
        model: np.sqrt(
            np.mean(
                np.square(
                    predict(contaminated_models[model], predictors)
                    - predict(clean_models[model], predictors)
                )
            )
        )
        for model in clean_models
    }

    assert drift["LAD"] < 1e-7
    assert drift["OLS"] > 1.0


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_invalid_contamination_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="between zero and one"):
        contaminated_row_count(10, rate)
