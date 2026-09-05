"""Tests for paired model comparisons and runtime summaries."""

import pandas as pd
import pytest

from outlier_analysis.benchmark import summarize_runtime
from outlier_analysis.comparison import (
    pair_model_results,
    summarize_contamination_comparison,
)


def test_pair_model_results_calculates_direction_and_reduction():
    data = pd.DataFrame(
        {
            "Case": [1, 1, 2, 2],
            "Model": ["OLS", "LAD", "OLS", "LAD"],
            "Error": [10.0, 6.0, 5.0, 7.0],
        }
    )

    paired = pair_model_results(data, ["Case"], ("Error",))

    first = paired.loc[paired["Case"].eq(1)].iloc[0]
    second = paired.loc[paired["Case"].eq(2)].iloc[0]
    assert first["LADMinusOLS_Error"] == pytest.approx(-4.0)
    assert first["LADReductionPercent_Error"] == pytest.approx(40.0)
    assert first["LADWins_Error"]
    assert second["LADReductionPercent_Error"] == pytest.approx(-40.0)
    assert not second["LADWins_Error"]


def test_pair_model_results_rejects_unmatched_pairs():
    data = pd.DataFrame(
        {
            "Case": [1, 1, 2],
            "Model": ["OLS", "LAD", "OLS"],
            "Error": [10.0, 6.0, 5.0],
        }
    )

    with pytest.raises(ValueError, match="keys do not match"):
        pair_model_results(data, ["Case"], ("Error",))


def test_contamination_summary_uses_paired_repetitions():
    rows = []
    for repetition, (ols_mae, lad_mae) in enumerate(
        ((10.0, 5.0), (8.0, 6.0)),
        start=1,
    ):
        rows.append(
            {
                "Dataset": "Example",
                "ContaminationRate": 0.2,
                "ContaminationPercent": 20.0,
                "Repetition": repetition,
                "Rows": 20,
                "ContaminatedRows": 4,
                "NoiseSD": 100.0,
                "OLS_CleanMAE": ols_mae,
                "LAD_CleanMAE": lad_mae,
                "OLS_CleanRMSE": ols_mae + 1.0,
                "LAD_CleanRMSE": lad_mae + 1.0,
                "OLS_PredictionRMSEFromCleanFit": 4.0,
                "LAD_PredictionRMSEFromCleanFit": 1.0,
                "OLS_StandardizedSlopeL2Shift": 2.0,
                "LAD_StandardizedSlopeL2Shift": 0.5,
            }
        )
    paired = pd.DataFrame(rows)

    summary = summarize_contamination_comparison(paired).iloc[0]

    assert summary["Repetitions"] == 2
    assert summary["OLS_CleanMAE_Mean"] == pytest.approx(9.0)
    assert summary["LAD_CleanMAE_Mean"] == pytest.approx(5.5)
    assert summary["LADWinRate_CleanMAE"] == pytest.approx(1.0)
    assert summary["LADReductionPercent_CleanMAE"] == pytest.approx(
        100.0 * 3.5 / 9.0
    )


def test_runtime_summary_uses_ols_as_ratio_reference():
    benchmark = pd.DataFrame(
        {
            "Dataset": ["Example"] * 6,
            "FitCondition": ["Full data"] * 6,
            "Model": ["OLS"] * 3 + ["LAD"] * 3,
            "Rows": [100] * 6,
            "Predictors": [3] * 6,
            "Repetition": [1, 2, 3, 1, 2, 3],
            "Milliseconds": [1.0, 2.0, 3.0, 8.0, 10.0, 12.0],
        }
    )

    summary = summarize_runtime(benchmark)
    ols = summary[summary["Model"].eq("OLS")].iloc[0]
    lad = summary[summary["Model"].eq("LAD")].iloc[0]

    assert ols["MedianMilliseconds"] == pytest.approx(2.0)
    assert ols["MedianRelativeToOLS"] == pytest.approx(1.0)
    assert lad["MedianMilliseconds"] == pytest.approx(10.0)
    assert lad["MedianRelativeToOLS"] == pytest.approx(5.0)
