"""Create individual and merged outlier-distribution figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .config import COLORS, FIGURE_DIR
from .detection import OutlierAnalysis


def plot_projection(
    ax: plt.Axes,
    analysis: OutlierAnalysis,
    annotate: bool,
) -> None:
    """Draw one robust-scaled PCA distribution on an existing axis."""

    for is_outlier, status in ((False, "Inlier"), (True, "Outlier")):
        mask = analysis.flag == is_outlier
        ax.scatter(
            analysis.projection[mask, 0],
            analysis.projection[mask, 1],
            s=45 if is_outlier else 30,
            alpha=0.82 if is_outlier else 0.58,
            color=COLORS[status],
            edgecolor="white",
            linewidth=0.35,
            label=status,
            zorder=3 if is_outlier else 2,
        )

    if annotate and np.any(analysis.flag):
        flagged_positions = np.flatnonzero(analysis.flag)
        top_positions = flagged_positions[
            np.argsort(analysis.distance[flagged_positions])[
                -min(10, len(flagged_positions)) :
            ]
        ]
        for position in top_positions:
            ax.annotate(
                str(analysis.record_ids[position]),
                (analysis.projection[position, 0], analysis.projection[position, 1]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )

    variance = analysis.explained_variance
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.grid(alpha=0.25)


def save_individual_figure(analysis: OutlierAnalysis) -> None:
    """Save PCA and distance-distribution panels for one dataset."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    plot_projection(axes[0], analysis, annotate=True)
    axes[0].set_title("Robust-scaled PCA distribution")
    axes[0].legend()

    for is_outlier, status in ((False, "Inlier"), (True, "Outlier")):
        mask = analysis.flag == is_outlier
        axes[1].hist(
            analysis.distance[mask],
            bins=24,
            alpha=0.68,
            color=COLORS[status],
            label=status,
        )
    axes[1].axvline(
        analysis.cutoff,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"Cutoff = {analysis.cutoff:.2f}",
    )
    axes[1].set(
        xlabel="Robust Mahalanobis distance",
        ylabel="Count",
        title="Distance distribution and cutoff",
    )
    axes[1].legend()

    count = int(analysis.flag.sum())
    fig.suptitle(
        f"{analysis.spec.name}: {count}/{len(analysis.flag)} "
        f"robust-distance outliers ({100 * count / len(analysis.flag):.1f}%)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{analysis.spec.slug}_outlier_distribution.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_merged_figure(analyses: list[OutlierAnalysis]) -> None:
    """Save a two-by-two comparison of all dataset projections."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, analysis in zip(axes.flat, analyses):
        plot_projection(ax, analysis, annotate=False)
        ax.set_title(
            f"{analysis.spec.name}\n{int(analysis.flag.sum())}/"
            f"{len(analysis.flag)} flagged ({100 * analysis.flag.mean():.1f}%)",
            fontsize=12,
            weight="bold",
        )
        legend = ax.get_legend()
        if legend:
            legend.remove()

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=9,
            label=status,
        )
        for status, color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True)
    fig.suptitle(
        "Robust multivariate outlier distributions across datasets",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(
        FIGURE_DIR / "merged_outlier_distributions.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
