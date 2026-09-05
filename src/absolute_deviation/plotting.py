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
        (y_clean, "clean", "Controlled clean regression example"),
        (y_bad, "contaminated", "Controlled example with large response errors"),
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
        (metrics, "SSE", "SSE evaluated against clean responses", "sse_vs_contamination.png"),
        (metrics, "SAE", "SAE evaluated against clean responses", "sae_vs_contamination.png"),
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
        values = [
            float(summary[(summary.distribution == d) & (summary.model == model)]["coefficient_error_l2"].iloc[0])
            for d in order
        ]
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
    for name in ("boston_housing", "concrete_strength", "hbk"):
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


def plot_hbk_multivariate_inlier_outlier() -> None:
    data_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "hbk.csv"
    df = pd.read_csv(data_path)
    X = df[["X1", "X2", "X3"]].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    ols = fit_ols(X, y)
    lad = fit_lad(X, y)
    abs_ols = np.abs(ols.residuals)
    abs_lad = np.abs(lad.residuals)
    groups = df["CaseGroup"].astype(str).str.lower()

    group_specs = (
        ("Regular cases (inliers)", groups.str.contains("regular"), "o"),
        ("Bad leverage cases (outliers)", groups.str.contains("bad leverage"), "X"),
        ("Good leverage cases", groups.str.contains("good leverage"), "^"),
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, mask, marker in group_specs:
        ax.scatter(abs_ols[mask], abs_lad[mask], s=55, marker=marker, alpha=0.85, label=label)

    upper = float(max(abs_ols.max(), abs_lad.max())) * 1.08
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1, label="Equal absolute residual")

    non_regular = ~groups.str.contains("regular")
    for observation, x_value, y_value in zip(
        df.loc[non_regular, "Observation"],
        abs_ols[non_regular],
        abs_lad[non_regular],
    ):
        ax.annotate(str(int(observation)), (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Absolute OLS residual (X1, X2, X3 used together)")
    ax.set_ylabel("Absolute LAD residual (X1, X2, X3 used together)")
    ax.set_title("HBK multivariate inlier/outlier visualization")
    ax.legend()
    _save(fig, "hbk_multivariate_inlier_outlier.png")


def plot_unlabeled_multivariate_residual_space(name: str, display_name: str) -> None:
    X, y, _ = load_dataset(name)
    ols = fit_ols(X, y)
    lad = fit_lad(X, y)
    abs_ols = np.abs(ols.residuals)
    abs_lad = np.abs(lad.residuals)

    residual_extent = np.maximum(abs_ols, abs_lad)
    n_highlight = min(5, len(y))
    highlight_idx = np.argsort(residual_extent)[-n_highlight:]
    highlight_mask = np.zeros(len(y), dtype=bool)
    highlight_mask[highlight_idx] = True

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        abs_ols[~highlight_mask],
        abs_lad[~highlight_mask],
        s=34,
        alpha=0.65,
        label="Other observations",
    )
    ax.scatter(
        abs_ols[highlight_mask],
        abs_lad[highlight_mask],
        s=70,
        marker="X",
        label="Largest residual observations (descriptive)",
    )

    upper = float(max(abs_ols.max(), abs_lad.max())) * 1.08
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1, label="Equal absolute residual")

    for idx in highlight_idx:
        ax.annotate(
            str(int(idx + 1)),
            (abs_ols[idx], abs_lad[idx]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Absolute OLS residual (all predictors used together)")
    ax.set_ylabel("Absolute LAD residual (all predictors used together)")
    ax.set_title(f"{display_name} multivariate residual visualization")
    ax.legend()
    _save(fig, f"{name}_multivariate_residuals.png")


def generate_all_figures() -> None:
    plot_clean_and_contaminated_lines()
    plot_contamination_results()
    plot_distribution_results()
    plot_runtime_results()
    plot_real_dataset_predictions()
    plot_hbk_multivariate_inlier_outlier()
    plot_unlabeled_multivariate_residual_space("boston_housing", "Boston Housing")
    plot_unlabeled_multivariate_residual_space("concrete_strength", "Concrete Strength")


if __name__ == "__main__":
    generate_all_figures()
