"""Plot OLS and LAD sensitivity to sample condition and contamination."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR, DatasetSpec
from .robustness_conditions import ConditionFits


MODEL_STYLES = {
    "OLS": {"color": "#4C78A8", "marker": "o"},
    "LAD": {"color": "#F28E2B", "marker": "^"},
}


def _draw_condition_contamination_panel(
    ax: plt.Axes,
    spec: DatasetSpec,
    condition_metrics: pd.DataFrame,
    contamination_summary: pd.DataFrame,
    show_title: bool = True,
) -> None:
    """Draw clean-inlier error curves and observed-full-fit references."""

    dataset_summary = contamination_summary[
        contamination_summary["Dataset"].eq(spec.name)
    ]
    full_references = condition_metrics[
        condition_metrics["Dataset"].eq(spec.name)
        & condition_metrics["FitCondition"].eq("Full data")
        & condition_metrics["EvaluationSet"].eq("Inliers only")
    ]

    for model_name, style in MODEL_STYLES.items():
        model_summary = dataset_summary[
            dataset_summary["Model"].eq(model_name)
        ].sort_values("ContaminationPercent")
        x_values = model_summary["ContaminationPercent"].to_numpy(dtype=float)
        means = model_summary["CleanMAEMean"].to_numpy(dtype=float)
        deviations = model_summary["CleanMAEStd"].to_numpy(dtype=float)
        ax.errorbar(
            x_values,
            means,
            yerr=deviations,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            capsize=3,
            label=f"{model_name}: contaminated inlier fit",
        )
        reference_mae = float(
            full_references.loc[
                full_references["Model"].eq(model_name),
                "MAE",
            ].iloc[0]
        )
        ax.axhline(
            reference_mae,
            color=style["color"],
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label=f"{model_name}: observed full-data fit",
        )

    ax.set(
        xlabel="Artificial response contamination (%)",
        ylabel="MAE on unchanged regular rows",
        title=spec.name if show_title else None,
        xticks=np.sort(dataset_summary["ContaminationPercent"].unique()),
    )
    ax.grid(alpha=0.25)


def save_condition_contamination_figure(
    spec: DatasetSpec,
    condition_metrics: pd.DataFrame,
    contamination_summary: pd.DataFrame,
) -> None:
    """Save one full/inlier/contamination comparison for a dataset."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_condition_contamination_panel(
        ax,
        spec,
        condition_metrics,
        contamination_summary,
        show_title=False,
    )
    ax.legend(loc="best")
    fig.suptitle(
        f"{spec.name}: OLS and LAD contamination sensitivity",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        FIGURE_DIR / f"{spec.slug}_condition_contamination_comparison.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_merged_condition_contamination_figure(
    conditions: list[ConditionFits],
    condition_metrics: pd.DataFrame,
    contamination_summary: pd.DataFrame,
) -> None:
    """Save a two-by-two robustness comparison for all datasets."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for ax, condition in zip(axes.flat, conditions):
        _draw_condition_contamination_panel(
            ax,
            condition.spec,
            condition_metrics,
            contamination_summary,
        )
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "OLS versus LAD under observed and artificial response outliers",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Points show mean across 30 repetitions; error bars show ±1 standard deviation.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(
        FIGURE_DIR / "merged_condition_contamination_comparison.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
