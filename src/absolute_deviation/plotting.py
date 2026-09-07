from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import FIGURE_DIR, RESULT_DIR
from .models import fit_lad, fit_ols


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _clear_old_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("*.png", "*.svg"):
        for path in FIGURE_DIR.glob(suffix):
            path.unlink()


def plot_empirical_sae_comparison() -> None:
    df = pd.read_csv(RESULT_DIR / "original_data_metrics.csv")
    labels = {
        "boston_housing": "Boston Housing",
        "concrete_strength": "Concrete Strength",
        "hbk": "HBK",
    }
    order = ["boston_housing", "concrete_strength", "hbk"]
    x = np.arange(len(order))
    width = 0.34

    ols = [
        float(df.loc[(df["dataset"] == name) & (df["model"] == "OLS"), "SAE"].iloc[0])
        for name in order
    ]
    lad = [
        float(df.loc[(df["dataset"] == name) & (df["model"] == "LAD"), "SAE"].iloc[0])
        for name in order
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, ols, width=width, label="OLS")
    ax.bar(x + width / 2, lad, width=width, label="LAD")
    ax.set_yscale("log")
    ax.set_xticks(x, [labels[name] for name in order])
    ax.set_ylabel("Sum of Absolute Errors (SAE)")
    ax.set_title("OLS vs LAD SAE on Benchmark Datasets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "empirical_sae_comparison")


def plot_hbk_multivariate_inlier_outlier() -> None:
    data_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "hbk.csv"
    )
    df = pd.read_csv(data_path)
    X = df[["X1", "X2", "X3"]].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)

    ols = fit_ols(X, y)
    lad = fit_lad(X, y)
    abs_ols = np.abs(ols.residuals)
    abs_lad = np.abs(lad.residuals)
    groups = df["CaseGroup"].astype(str).str.lower()

    group_specs = (
        ("Regular cases", groups.str.contains("regular"), "o"),
        ("Bad leverage cases", groups.str.contains("bad leverage"), "X"),
        ("Good leverage cases", groups.str.contains("good leverage"), "^"),
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, mask, marker in group_specs:
        ax.scatter(
            abs_ols[mask],
            abs_lad[mask],
            s=55,
            marker=marker,
            alpha=0.82,
            label=label,
        )

    upper = float(max(abs_ols.max(), abs_lad.max())) * 1.08
    ax.plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        linewidth=1,
        label="Equal absolute residual",
    )

    non_regular = ~groups.str.contains("regular")
    for observation, x_value, y_value in zip(
        df.loc[non_regular, "Observation"],
        abs_ols[non_regular],
        abs_lad[non_regular],
    ):
        ax.annotate(
            str(int(observation)),
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Absolute OLS residual")
    ax.set_ylabel("Absolute LAD residual")
    ax.set_title("HBK Multivariate Residual Comparison")
    ax.legend(frameon=False)
    _save(fig, "hbk_multivariate_inlier_outlier")


def plot_error_distribution_sae() -> None:
    df = pd.read_csv(RESULT_DIR / "distribution_experiment.csv")
    summary = (
        df.groupby(["distribution", "model"], as_index=False)["SAE"]
        .median()
    )
    order = ["normal", "laplace", "cauchy"]
    labels = ["Normal", "Laplace", "Cauchy"]
    x = np.arange(len(order))
    width = 0.34

    ols = [
        float(summary.loc[(summary["distribution"] == name) & (summary["model"] == "OLS"), "SAE"].iloc[0])
        for name in order
    ]
    lad = [
        float(summary.loc[(summary["distribution"] == name) & (summary["model"] == "LAD"), "SAE"].iloc[0])
        for name in order
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, ols, width=width, label="OLS")
    ax.bar(x + width / 2, lad, width=width, label="LAD")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median SAE")
    ax.set_title("Median SAE under Different Error Distributions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "error_distribution_sae_median")


def plot_runtime_comparison() -> None:
    df = pd.read_csv(RESULT_DIR / "runtime_results.csv")
    summary = (
        df.groupby(["n", "p", "model"], as_index=False)["runtime_seconds"]
        .median()
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharey=True)
    for ax, p in zip(axes, (1, 3, 5)):
        subset = summary[summary["p"] == p]
        for model in ("OLS", "LAD"):
            model_rows = subset[subset["model"] == model].sort_values("n")
            ax.plot(
                model_rows["n"],
                model_rows["runtime_seconds"],
                marker="o",
                label=model,
            )
        ax.set_title(f"p = {p}")
        ax.set_xlabel("Observations (n)")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Median runtime (seconds, log scale)")
    axes[-1].legend(frameon=False)
    fig.suptitle("OLS vs LAD Runtime as Problem Size Increases", y=1.02)
    _save(fig, "runtime_comparison")


def generate_all_figures() -> None:
    _clear_old_figures()
    plot_empirical_sae_comparison()
    plot_hbk_multivariate_inlier_outlier()
    plot_error_distribution_sae()
    plot_runtime_comparison()


if __name__ == "__main__":
    generate_all_figures()
