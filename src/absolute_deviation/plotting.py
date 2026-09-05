from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_dataset
from .experiments import FIGURE_DIR, RESULT_DIR
from .models import fit_lad, fit_ols


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_clean_and_contaminated_lines(seed: int = 1959) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, 80)
    y_clean = 3.0 + 2.0 * x + rng.normal(scale=0.7, size=x.size)
    y_bad = y_clean.copy()
    y_bad[-4:] += np.array([15.0, 18.0, 21.0, 24.0])
    X = x.reshape(-1, 1)
    for y, suffix, title in (
        (y_clean, "clean", "Clean synthetic data"),
        (y_bad, "contaminated", "Synthetic data with large response errors"),
    ):
        ols, lad = fit_ols(X, y), fit_lad(X, y)
        order = np.argsort(x)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, s=24, alpha=0.8, label="Observations")
        ax.plot(x[order], ols.fitted[order], label="OLS")
        ax.plot(x[order], lad.fitted[order], label="LAD")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()
        _save(fig, f"ols_lad_{suffix}_fit.png")


def plot_contamination_results() -> None:
    metrics = pd.read_csv(RESULT_DIR / "contamination_metrics.csv")
    shifts = pd.read_csv(RESULT_DIR / "contamination_coefficient_changes.csv")
    for frame, value, ylabel, filename in (
        (metrics, "MAE", "MAE evaluated against clean responses", "mae_vs_contamination.png"),
        (metrics, "RMSE", "RMSE evaluated against clean responses", "rmse_vs_contamination.png"),
        (shifts, "coefficient_shift_l2", "Coefficient change from clean fit", "coefficient_shift_vs_contamination.png"),
    ):
        summary = frame.groupby(["contamination_fraction", "model"], as_index=False)[value].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, part in summary.groupby("model"):
            ax.plot(part["contamination_fraction"] * 100, part[value], marker="o", label=model)
        ax.set_xlabel("Contaminated responses (%)")
        ax.set_ylabel(ylabel)
        ax.legend()
        _save(fig, filename)


def plot_distribution_results() -> None:
    df = pd.read_csv(RESULT_DIR / "distribution_experiment.csv")
    summary = df.groupby(["distribution", "model"], as_index=False)["coefficient_error_l2"].median()
    order = ["normal", "laplace", "cauchy", "contaminated_normal"]
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, model in ((-width / 2, "OLS"), (width / 2, "LAD")):
        values = [float(summary[(summary.distribution == d) & (summary.model == model)]["coefficient_error_l2"].iloc[0]) for d in order]
        ax.bar(x + offset, values, width, label=model)
    ax.set_xticks(x, ["Normal", "Laplace", "Cauchy", "Contaminated normal"])
    ax.set_ylabel("Median coefficient estimation error")
    ax.legend()
    _save(fig, "error_distribution_comparison.png")


def plot_runtime_results() -> None:
    df = pd.read_csv(RESULT_DIR / "runtime_results.csv")
    summary = df.groupby(["n", "model"], as_index=False)["runtime_seconds"].median()
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, part in summary.groupby("model"):
        ax.plot(part["n"], part["runtime_seconds"], marker="o", label=model)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Median fit time (seconds)")
    ax.legend()
    _save(fig, "runtime_vs_sample_size.png")


def plot_real_dataset_predictions() -> None:
    for name in ("boston_housing", "concrete_strength", "hbk", "synthetic"):
        X, y, _ = load_dataset(name)
        ols, lad = fit_ols(X, y), fit_lad(X, y)
        lo, hi = float(np.min(y)), float(np.max(y))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y, ols.fitted, s=20, alpha=0.7, label="OLS")
        ax.scatter(y, lad.fitted, s=20, alpha=0.7, label="LAD")
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="Perfect fit")
        ax.set_xlabel("Observed response")
        ax.set_ylabel("Fitted response")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        _save(fig, f"{name}_actual_vs_fitted.png")


def generate_all_figures() -> None:
    plot_clean_and_contaminated_lines()
    plot_contamination_results()
    plot_distribution_results()
    plot_runtime_results()
    plot_real_dataset_predictions()


if __name__ == "__main__":
    generate_all_figures()
