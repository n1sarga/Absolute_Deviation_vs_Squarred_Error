from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import FIGURE_DIR, RESULT_DIR
from .models import fit_lad, fit_ols


KEY_FIGURES = (
    "ols_lad_contaminated_fit.png",
    "sse_vs_contamination.png",
    "sae_vs_contamination.png",
    "hbk_multivariate_inlier_outlier.png",
)


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _clear_old_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def plot_contaminated_fit(seed: int = 1959) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, 80)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.7, size=x.size)
    y[-4:] += np.array([15.0, 18.0, 21.0, 24.0])
    X = x.reshape(-1, 1)

    ols = fit_ols(X, y)
    lad = fit_lad(X, y)
    order = np.argsort(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=24, alpha=0.75, label="Observations")
    ax.plot(x[order], ols.fitted[order], linewidth=2, label="OLS")
    ax.plot(x[order], lad.fitted[order], linewidth=2, label="LAD")
    ax.set_xlabel("Predictor x")
    ax.set_ylabel("Response y")
    ax.set_title("OLS and LAD with large response errors")
    ax.legend()
    _save(fig, "ols_lad_contaminated_fit.png")


def plot_contamination_objectives() -> None:
    metrics = pd.read_csv(RESULT_DIR / "contamination_metrics.csv")

    for value, ylabel, filename, title in (
        (
            "SSE",
            "Mean SSE against clean responses",
            "sse_vs_contamination.png",
            "Squared error as contamination increases",
        ),
        (
            "SAE",
            "Mean SAE against clean responses",
            "sae_vs_contamination.png",
            "Absolute error as contamination increases",
        ),
    ):
        summary = metrics.groupby(
            ["contamination_fraction", "model"], as_index=False
        )[value].mean()

        fig, ax = plt.subplots(figsize=(7, 5))
        for model, part in summary.groupby("model"):
            ax.plot(
                part["contamination_fraction"] * 100,
                part[value],
                marker="o",
                linewidth=2,
                label=model,
            )
        ax.set_xlabel("Contaminated responses (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _save(fig, filename)


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
    ax.set_title("HBK multivariate residual comparison")
    ax.legend()
    _save(fig, "hbk_multivariate_inlier_outlier.png")


def generate_all_figures() -> None:
    _clear_old_figures()
    plot_contaminated_fit()
    plot_contamination_objectives()
    plot_hbk_multivariate_inlier_outlier()


if __name__ == "__main__":
    generate_all_figures()
