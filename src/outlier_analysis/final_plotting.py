"""Plot repeated cross-validation and paired bootstrap inference."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR


SHORT_NAMES = {
    "Boston Housing": "Boston",
    "Concrete Strength": "Concrete",
    "Hawkins-Bradu-Kass (HBK)": "HBK",
    "Synthetic OLS-LAD": "Synthetic",
}


def save_cross_validation_figure(repeat_metrics: pd.DataFrame) -> None:
    """Save paired repeat-level MAE comparisons for every dataset."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, (dataset, subset) in zip(
        axes.flat,
        repeat_metrics.groupby("Dataset", sort=False),
    ):
        paired = subset.pivot(index="Repeat", columns="Model", values="MAE")
        for _, row in paired.iterrows():
            ax.plot(
                [0, 1],
                [row["OLS"], row["LAD"]],
                color="#A0A0A0",
                alpha=0.55,
                linewidth=1.0,
                zorder=1,
            )
        ax.scatter(
            np.zeros(len(paired)),
            paired["OLS"],
            color="#4E79A7",
            s=38,
            alpha=0.75,
            label="OLS repeats",
            zorder=2,
        )
        ax.scatter(
            np.ones(len(paired)),
            paired["LAD"],
            color="#F28E2B",
            marker="^",
            s=44,
            alpha=0.75,
            label="LAD repeats",
            zorder=2,
        )
        ax.set(
            title=dataset,
            ylabel="Out-of-fold MAE",
            xlim=(-0.35, 1.35),
            xticks=[0, 1],
            xticklabels=["OLS", "LAD"],
        )
        ax.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Repeated 5-fold cross-validation: paired OLS and LAD MAE",
        fontsize=17,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        FIGURE_DIR / "cross_validation_comparison.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_bootstrap_forest_figure(inference: pd.DataFrame) -> None:
    """Save LAD-minus-OLS MAE differences and bootstrap intervals."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, ax = plt.subplots(figsize=(11, 6.5))
    y_values = np.arange(len(inference))
    estimates = inference["LADMinusOLSMAE"].to_numpy(dtype=float)
    lower = inference["MAECILower"].to_numpy(dtype=float)
    upper = inference["MAECIUpper"].to_numpy(dtype=float)
    colors = [
        "#59A14F" if high < 0.0 else "#9C9C9C"
        for high in inference["MAECIUpper"]
    ]

    for position, estimate, low, high, color in zip(
        y_values,
        estimates,
        lower,
        upper,
        colors,
    ):
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=5,
            markersize=8,
        )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.4)
    ax.set(
        xlabel="LAD MAE − OLS MAE (95% paired-bootstrap CI)",
        yticks=y_values,
        yticklabels=[
            SHORT_NAMES.get(dataset, dataset) for dataset in inference["Dataset"]
        ],
        title="Out-of-fold absolute-error difference",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    figure.suptitle(
        "Paired bootstrap inference: negative values favor LAD",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(
        FIGURE_DIR / "bootstrap_mae_difference.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)
