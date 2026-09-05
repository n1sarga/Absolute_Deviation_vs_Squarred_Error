"""Benchmark fit-only execution time for OLS and LAD."""

import platform
import sys
from importlib.metadata import version
from time import perf_counter_ns

import pandas as pd

from .config import RUNTIME_REPETITIONS, RUNTIME_WARMUPS
from .robustness_conditions import ConditionFits, MODEL_FITTERS


def benchmark_condition_models(
    conditions: list[ConditionFits],
    repetitions: int = RUNTIME_REPETITIONS,
    warmups: int = RUNTIME_WARMUPS,
) -> pd.DataFrame:
    """Time model fitting after untimed warm-up runs."""

    if repetitions < 1 or warmups < 0:
        raise ValueError("runtime repetitions must be positive and warmups nonnegative")

    rows: list[dict[str, object]] = []
    for condition in conditions:
        fit_conditions = {
            "Full data": (condition.predictors, condition.response),
            "Inliers only": (
                condition.inlier_predictors,
                condition.inlier_response,
            ),
        }
        for fit_condition, (predictors, response) in fit_conditions.items():
            for model_name, fitter in MODEL_FITTERS.items():
                for _ in range(warmups):
                    fitter(predictors, response)
                for repetition in range(1, repetitions + 1):
                    started = perf_counter_ns()
                    fitter(predictors, response)
                    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
                    rows.append(
                        {
                            "Dataset": condition.spec.name,
                            "FitCondition": fit_condition,
                            "Model": model_name,
                            "Rows": len(response),
                            "Predictors": predictors.shape[1],
                            "Repetition": repetition,
                            "Milliseconds": elapsed_ms,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_runtime(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Summarize timing distribution and ratio to OLS median."""

    summary = (
        benchmark.groupby(
            ["Dataset", "FitCondition", "Model", "Rows", "Predictors"],
            sort=False,
        )["Milliseconds"]
        .agg(
            Repetitions="count",
            MeanMilliseconds="mean",
            SDMilliseconds="std",
            MinMilliseconds="min",
            Q1Milliseconds=lambda values: values.quantile(0.25),
            MedianMilliseconds="median",
            Q3Milliseconds=lambda values: values.quantile(0.75),
            MaxMilliseconds="max",
        )
        .reset_index()
    )
    ols_medians = summary[summary["Model"].eq("OLS")][
        ["Dataset", "FitCondition", "MedianMilliseconds"]
    ].rename(columns={"MedianMilliseconds": "OLSMedianMilliseconds"})
    summary = summary.merge(
        ols_medians,
        on=["Dataset", "FitCondition"],
        validate="many_to_one",
    )
    summary["MedianRelativeToOLS"] = (
        summary["MedianMilliseconds"] / summary["OLSMedianMilliseconds"]
    )
    return summary


def runtime_environment() -> pd.DataFrame:
    """Record software and host context needed to interpret timings."""

    return pd.DataFrame(
        [
            {
                "Python": sys.version.split()[0],
                "Platform": platform.platform(),
                "Processor": platform.processor() or "Unavailable",
                "NumPy": version("numpy"),
                "SciPy": version("scipy"),
            }
        ]
    )
