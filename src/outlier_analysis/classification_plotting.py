"""Visualize regression-outlier classes and their diagnostic boundaries."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .classification import RegressionOutlierClassification
from .config import CLASS_COLORS, FIGURE_DIR


def category_handles() -> list[Line2D]:
    """Return consistent legend handles for all four classes."""

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=category,
        )
        for category, color in CLASS_COLORS.items()
    ]


def scatter_categories(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
) -> None:
    """Scatter values with a stable color for each diagnostic class."""

    for category, color in CLASS_COLORS.items():
        mask = categories == category
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=43 if category != "Regular" else 28,
            alpha=0.86 if category != "Regular" else 0.55,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3 if category != "Regular" else 2,
        )


def annotate_extremes(
    ax: plt.Axes,
    classification: RegressionOutlierClassification,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Label the most diagnostically extreme flagged observations."""

    flagged = classification.leverage_flag | classification.response_flag
    positions = np.flatnonzero(flagged)
    if not len(positions):
        return

    severity = np.maximum(
        classification.leverage_distance / classification.leverage_cutoff,
        classification.residual_score / classification.residual_cutoff,
    )
    top = positions[np.argsort(severity[positions])[-min(10, len(positions)) :]]
    for position in top:
        ax.annotate(
            str(classification.record_ids[position]),
            (x[position], y[position]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )


def plot_predictor_projection(
    ax: plt.Axes,
    classification: RegressionOutlierClassification,
) -> None:
    """Plot outlier classes in robust-scaled predictor PCA space."""

    projection = classification.projection
    scatter_categories(
        ax,
        projection[:, 0],
        projection[:, 1],
        classification.category,
    )
    variance = classification.explained_variance
    ax.set(
        xlabel=f"Predictor PC1 ({variance[0] * 100:.1f}% variance)",
        ylabel=f"Predictor PC2 ({variance[1] * 100:.1f}% variance)",
        title="Predictor-space distribution",
    )
    ax.grid(alpha=0.25)


def plot_classification_plane(
    ax: plt.Axes,
    classification: RegressionOutlierClassification,
    annotate: bool,
) -> None:
    """Plot leverage and response scores against their classification cutoffs."""

    scatter_categories(
        ax,
        classification.leverage_distance,
        classification.residual_score,
        classification.category,
    )
    ax.axvline(
        classification.leverage_cutoff,
        color="black",
        linestyle="--",
        linewidth=1.3,
    )
    ax.axhline(
        classification.residual_cutoff,
        color="black",
        linestyle=":",
        linewidth=1.5,
    )
    if annotate:
        annotate_extremes(
            ax,
            classification,
            classification.leverage_distance,
            classification.residual_score,
        )
    ax.set(
        xlabel="Predictor robust Mahalanobis distance",
        ylabel="Absolute robust residual z-score",
        title="Classification plane",
    )
    ax.grid(alpha=0.25)


def save_classification_figure(
    classification: RegressionOutlierClassification,
) -> None:
    """Save predictor PCA and diagnostic-classification panels for one dataset."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    plot_predictor_projection(axes[0], classification)
    plot_classification_plane(axes[1], classification, annotate=True)
    axes[0].legend(handles=category_handles(), loc="best")

    counts = {
        category: int(np.sum(classification.category == category))
        for category in CLASS_COLORS
    }
    fig.suptitle(
        f"{classification.spec.name}: leverage only={counts['Leverage only']}, "
        f"response only={counts['Response only']}, "
        f"both={counts['Leverage + response']}",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{classification.spec.slug}_outlier_classification.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_merged_classification_figure(
    classifications: list[RegressionOutlierClassification],
) -> None:
    """Save all four classification planes in one comparison figure."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, classification in zip(axes.flat, classifications):
        plot_classification_plane(ax, classification, annotate=False)
        flagged = classification.leverage_flag | classification.response_flag
        ax.set_title(
            f"{classification.spec.name}\n{int(flagged.sum())}/"
            f"{len(flagged)} classified as unusual",
            fontsize=12,
            weight="bold",
        )

    fig.legend(
        handles=category_handles(),
        loc="lower center",
        ncol=4,
        frameon=True,
    )
    fig.suptitle(
        "Regression outlier classification: leverage versus response",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(
        FIGURE_DIR / "merged_outlier_classifications.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
