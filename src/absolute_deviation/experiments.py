from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, load_dataset
from .models import fit_lad, fit_ols, regression_metrics

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "outputs" / "results"
FIGURE_DIR = ROOT / "outputs" / "figures"


def _ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def run_original_data() -> pd.DataFrame:
    _ensure_dirs()
    metric_rows: list[dict] = []
    for name in DATASETS:
        X, y, _ = load_dataset(name)
        for fitter in (fit_ols, fit_lad):
            result = fitter(X, y)
            if not result.success:
                raise RuntimeError(f"{name} LAD failed: {result.status}")
            metrics = regression_metrics(y, result.fitted)
            metric_rows.append(
                {
                    "dataset": name,
                    "model": result.model,
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULT_DIR / "original_data_metrics.csv", index=False)
    return metrics_df


def run_error_distribution_experiment(
    repetitions: int = 100,
    n: int = 150,
    seed: int = 1978,
) -> pd.DataFrame:
    _ensure_dirs()
    rng = np.random.default_rng(seed)
    beta_true = np.array([5.0, 1.0, 2.0, 3.0])
    rows: list[dict] = []
    distributions = ("normal", "laplace", "cauchy")
    for rep in range(repetitions):
        X = rng.normal(size=(n, 3))
        A = np.column_stack([np.ones(n), X])
        for distribution in distributions:
            if distribution == "normal":
                errors = rng.normal(size=n)
            elif distribution == "laplace":
                errors = rng.laplace(scale=1 / np.sqrt(2), size=n)
            else:
                errors = rng.standard_cauchy(size=n)
            y = A @ beta_true + errors
            for fitter in (fit_ols, fit_lad):
                result = fitter(X, y)
                metrics = regression_metrics(y, result.fitted)
                rows.append(
                    {
                        "repetition": rep,
                        "distribution": distribution,
                        "model": result.model,
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "distribution_experiment.csv", index=False)
    return df


def run_runtime_benchmark(
    sample_sizes: tuple[int, ...] = (50, 100, 250, 500),
    predictor_counts: tuple[int, ...] = (1, 3, 5),
    repetitions: int = 5,
    seed: int = 1973,
) -> pd.DataFrame:
    _ensure_dirs()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for n in sample_sizes:
        for p in predictor_counts:
            beta = np.arange(1, p + 1, dtype=float)
            for rep in range(repetitions):
                X = rng.normal(size=(n, p))
                y = 2.0 + X @ beta + rng.normal(size=n)
                for fitter in (fit_ols, fit_lad):
                    result = fitter(X, y)
                    rows.append(
                        {
                            "n": n,
                            "p": p,
                            "repetition": rep,
                            "model": result.model,
                            "runtime_seconds": result.runtime_seconds,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "runtime_results.csv", index=False)
    return df


def validate_lad_solver() -> pd.DataFrame:
    _ensure_dirs()
    examples = [
        (np.arange(1.0, 8.0).reshape(-1, 1), np.array([2, 4, 6, 8, 10, 12, 40], dtype=float)),
        (np.array([[0, 1], [1, 0], [1, 1], [2, 1], [1, 2]], dtype=float), np.array([1, 1, 2, 3, 3], dtype=float)),
    ]
    rows = []
    for i, (X, y) in enumerate(examples, start=1):
        ols = fit_ols(X, y)
        lad = fit_lad(X, y)
        ols_m = regression_metrics(y, ols.fitted)
        lad_m = regression_metrics(y, lad.fitted)
        rows.append(
            {
                "example": i,
                "ols_sse": ols_m["SSE"],
                "lad_sse": lad_m["SSE"],
                "ols_sae": ols_m["SAE"],
                "lad_sae": lad_m["SAE"],
                "ols_sse_is_minimum_against_lad": ols_m["SSE"] <= lad_m["SSE"] + 1e-8,
                "lad_sae_is_minimum_against_ols": lad_m["SAE"] <= ols_m["SAE"] + 1e-8,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "lad_solver_validation.csv", index=False)
    return df


def generate_all_results() -> None:
    run_original_data()
    run_error_distribution_experiment()
    run_runtime_benchmark()
    validate_lad_solver()


if __name__ == "__main__":
    generate_all_results()
