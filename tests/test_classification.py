"""Tests for predictor-leverage and response-outlier classification."""

import numpy as np
import pytest

from outlier_analysis.classification import classify_regression_outliers
from outlier_analysis.config import CLASS_COLORS, DATASETS
from outlier_analysis.detection import load_dataset


SPECS_BY_SLUG = {spec.slug: spec for spec in DATASETS}


@pytest.mark.parametrize("spec", DATASETS, ids=lambda spec: spec.slug)
def test_every_row_receives_one_consistent_class(spec):
    classification = classify_regression_outliers(spec, load_dataset(spec))
    flagged = classification.leverage_flag | classification.response_flag

    assert len(classification.category) == len(classification.result)
    assert set(classification.category).issubset(CLASS_COLORS)
    assert np.array_equal(
        classification.category != "Regular",
        flagged,
    )
    assert np.array_equal(
        classification.result["LeverageFlag"].to_numpy(),
        classification.leverage_flag,
    )
    assert np.array_equal(
        classification.result["ResponseOutlierFlag"].to_numpy(),
        classification.response_flag,
    )


def test_hbk_classification_recovers_documented_observations():
    spec = SPECS_BY_SLUG["hbk"]
    classification = classify_regression_outliers(spec, load_dataset(spec))
    flagged_ids = set(
        classification.result.loc[
            classification.result["OutlierClass"] != "Regular",
            "Observation",
        ]
    )

    assert flagged_ids == set(range(1, 15))


def test_updated_synthetic_classification_is_reproducible():
    spec = SPECS_BY_SLUG["synthetic"]
    classification = classify_regression_outliers(spec, load_dataset(spec))
    flagged_ids = set(
        classification.result.loc[
            classification.result["OutlierClass"] != "Regular",
            "ID",
        ]
    )

    assert flagged_ids == {56, 76, *range(81, 91)}
