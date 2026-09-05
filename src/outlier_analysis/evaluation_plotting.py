"""Plot formal stability and runtime comparisons."""

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


def _annotate_bars(ax: plt.Axes, decimals: int = 1, suffix: str = "%") -> None:
    """Label positive bar heights."""

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_model_comparison_dashboard(findings: pd.DataFrame) -> None:
    """Save 20%-contamination effect sizes and runtime cost."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    names = [SHORT_NAMES.get(name, name) for name in findings["Dataset"]]
    x_values = np.arange(len(findings))
    panels = (
        (
            "MAEReductionPercent",
            "Clean-reference MAE reduction",
            "LAD reduction versus OLS (%)",
            "#59A14F",
        ),
        (
            "PredictionDriftReductionPercent",
            "Prediction-drift reduction",
            "LAD reduction versus OLS (%)",
            "#4E79A7",
        ),
        (
            "SlopeShiftReductionPercent",
            "Standardized slope-shift reduction",
            "LAD reduction versus OLS (%)",
            "#B279A2",
        ),
    )

    for ax, (column, title, ylabel, color) in zip(axes.flat[:3], panels):
        ax.bar(x_values, findings[column], color=color, width=0.68)
        ax.set(
            title=title,
            ylabel=ylabel,
            xticks=x_values,
            xticklabels=names,
            ylim=(0, 105),
        )
        _annotate_bars(ax)
        ax.grid(axis="y", alpha=0.25)

    runtime_ax = axes.flat[3]
    runtime_ax.bar(
        x_values,
        findings["LADRuntimeMultiple"],
        color="#F28E2B",
        width=0.68,
    )
    runtime_ax.set(
        title="LAD median fit-time cost (inlier sample)",
        ylabel="LAD runtime / OLS runtime (log scale)",
        xticks=x_values,
        xticklabels=names,
        yscale="log",
    )
    _annotate_bars(runtime_ax, decimals=0, suffix="×")
    runtime_ax.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "OLS versus LAD formal comparison at 20% response contamination",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Positive reductions favor LAD. Runtime values are machine-specific.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(
        FIGURE_DIR / "model_comparison_dashboard.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_runtime_figure(runtime_summary: pd.DataFrame) -> None:
    """Save median fit time for both models and sample conditions."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, (dataset, subset) in zip(
        axes.flat,
        runtime_summary.groupby("Dataset", sort=False),
    ):
        conditions = ["Full data", "Inliers only"]
        x_values = np.arange(len(conditions))
        width = 0.34
        for offset, (model, color) in zip(
            (-width / 2, width / 2),
            (("OLS", "#4E79A7"), ("LAD", "#F28E2B")),
        ):
            values = [
                float(
                    subset.loc[
                        subset["FitCondition"].eq(condition)
                        & subset["Model"].eq(model),
                        "MedianMilliseconds",
                    ].iloc[0]
                )
                for condition in conditions
            ]
            ax.bar(
                x_values + offset,
                values,
                width=width,
                label=model,
                color=color,
            )
        ax.set(
            title=dataset,
            ylabel="Median fit time (ms, log scale)",
            xticks=x_values,
            xticklabels=conditions,
            yscale="log",
        )
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Fit-only runtime benchmark: OLS versus LP-based LAD",
        fontsize=17,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        FIGURE_DIR / "runtime_benchmark.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)
