from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

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


def _coefficient_rows(dataset: str, result, names: list[str]) -> list[dict]:
    rows = [{"dataset": dataset, "model": result.model, "term": "Intercept", "estimate": result.intercept}]
    rows.extend(
        {"dataset": dataset, "model": result.model, "term": name, "estimate": float(value)}
        for name, value in zip(names, result.coefficients)
    )
    return rows


def run_original_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit OLS and LAD to each full dataset and report both loss criteria."""
    _ensure_dirs()
    metric_rows: list[dict] = []
    coef_rows: list[dict] = []
    for name in DATASETS:
        X, y, columns = load_dataset(name)
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
                    "runtime_seconds": result.runtime_seconds,
                }
            )
            coef_rows.extend(_coefficient_rows(name, result, columns))
    metrics_df = pd.DataFrame(metric_rows)
    coefs_df = pd.DataFrame(coef_rows)
    metrics_df.to_csv(RESULT_DIR / "original_data_metrics.csv", index=False)
    coefs_df.to_csv(RESULT_DIR / "original_data_coefficients.csv", index=False)
    return metrics_df, coefs_df


def _synthetic_design(seed: int = 1959, n: int = 200, p: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.arange(1, p + 1, dtype=float)
    y = 5.0 + X @ beta + rng.normal(scale=1.0, size=n)
    return X, y, beta


def run_contamination_experiment(
    contamination_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    repetitions: int = 30,
    seed: int = 1982,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Illustrate sensitivity to deliberately introduced large vertical errors.

    The exact contamination percentages and repetition count are project-selected
    experimental settings; they are not claimed to come from the papers.
    """
    _ensure_dirs()
    X, y_clean, _ = _synthetic_design(seed=seed)
    base_ols = fit_ols(X, y_clean)
    base_lad = fit_lad(X, y_clean)
    base = {"OLS": base_ols, "LAD": base_lad}
    rng = np.random.default_rng(seed + 1)
    metric_rows: list[dict] = []
    shift_rows: list[dict] = []
    n = len(y_clean)
    for level in contamination_levels:
        for rep in range(repetitions):
            y = y_clean.copy()
            m = int(round(level * n))
            if m:
                idx = rng.choice(n, size=m, replace=False)
                scale = max(float(np.std(y_clean)), 1.0)
                signs = rng.choice([-1.0, 1.0], size=m)
                y[idx] += signs * 15.0 * scale
            for fitter in (fit_ols, fit_lad):
                result = fitter(X, y)
                metrics = regression_metrics(y_clean, result.fitted)
                beta_result = np.concatenate([[result.intercept], result.coefficients])
                beta_base = np.concatenate([[base[result.model].intercept], base[result.model].coefficients])
                coefficient_shift = float(np.linalg.norm(beta_result - beta_base))
                prediction_shift = float(np.mean(np.abs(result.fitted - base[result.model].fitted)))
                metric_rows.append(
                    {
                        "contamination_fraction": level,
                        "repetition": rep,
                        "model": result.model,
                        **metrics,
                        "runtime_seconds": result.runtime_seconds,
                    }
                )
                shift_rows.append(
                    {
                        "contamination_fraction": level,
                        "repetition": rep,
                        "model": result.model,
                        "coefficient_shift_l2": coefficient_shift,
                        "prediction_shift_mae": prediction_shift,
                    }
                )
    metrics_df = pd.DataFrame(metric_rows)
    shifts_df = pd.DataFrame(shift_rows)
    metrics_df.to_csv(RESULT_DIR / "contamination_metrics.csv", index=False)
    shifts_df.to_csv(RESULT_DIR / "contamination_coefficient_changes.csv", index=False)
    return metrics_df, shifts_df


def run_error_distribution_experiment(
    repetitions: int = 100,
    n: int = 150,
    seed: int = 1978,
) -> pd.DataFrame:
    """Compare OLS and LAD under distributions discussed in the supplied literature."""
    _ensure_dirs()
    rng = np.random.default_rng(seed)
    beta_true = np.array([5.0, 1.0, 2.0, 3.0])
    rows: list[dict] = []
    distributions = ("normal", "laplace", "cauchy", "contaminated_normal")
    for rep in range(repetitions):
        X = rng.normal(size=(n, 3))
        A = np.column_stack([np.ones(n), X])
        for distribution in distributions:
            if distribution == "normal":
                errors = rng.normal(size=n)
            elif distribution == "laplace":
                errors = rng.laplace(scale=1 / np.sqrt(2), size=n)
            elif distribution == "cauchy":
                errors = rng.standard_cauchy(size=n)
            else:
                errors = rng.normal(size=n)
                mask = rng.random(n) < 0.10
                errors[mask] += rng.normal(scale=10.0, size=mask.sum())
            y = A @ beta_true + errors
            for fitter in (fit_ols, fit_lad):
                result = fitter(X, y)
                beta_hat = np.concatenate([[result.intercept], result.coefficients])
                metrics = regression_metrics(y, result.fitted)
                rows.append(
                    {
                        "repetition": rep,
                        "distribution": distribution,
                        "model": result.model,
                        "coefficient_error_l2": float(np.linalg.norm(beta_hat - beta_true)),
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
    """Check the defining inequalities of OLS and LAD on deterministic examples."""
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
    run_contamination_experiment()
    run_error_distribution_experiment()
    run_runtime_benchmark()
    validate_lad_solver()


if __name__ == "__main__":
    generate_all_results()
