"""Regression tests for dataset loading and robust outlier flags."""

import pytest

from outlier_analysis.config import DATASETS
from outlier_analysis.detection import calculate_outliers, load_dataset


SPECS_BY_SLUG = {spec.slug: spec for spec in DATASETS}


@pytest.mark.parametrize("spec", DATASETS, ids=lambda spec: spec.slug)
def test_configured_dataset_loads(spec):
    data = load_dataset(spec)

    assert not data.empty
    assert set(spec.variables).issubset(data.columns)


@pytest.mark.parametrize(
    ("slug", "expected_rows", "expected_outliers"),
    [
        ("hbk", 75, 14),
        ("synthetic", 90, 12),
    ],
)
def test_reference_outlier_counts(slug, expected_rows, expected_outliers):
    spec = SPECS_BY_SLUG[slug]
    analysis = calculate_outliers(spec, load_dataset(spec))

    assert len(analysis.result) == expected_rows
    assert int(analysis.flag.sum()) == expected_outliers


def test_hbk_detector_finds_all_documented_outliers():
    spec = SPECS_BY_SLUG["hbk"]
    analysis = calculate_outliers(spec, load_dataset(spec))

    assert analysis.known is not None
    assert (analysis.flag[analysis.known]).all()
