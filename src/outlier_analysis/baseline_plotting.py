"""Plot full-data OLS and LAD baseline fits."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .baseline import BaselineComparison, regression_metrics
from .config import FIGURE_DIR


MODEL_STYLES = {
    "OLS": {"color": "#4C78A8", "marker": "o"},
    "LAD": {"color": "#F28E2B", "marker": "^"},
}


def _axis_limits(comparison: BaselineComparison) -> tuple[float, float]:
    """Return shared padded limits for actual and predicted values."""

    values = np.concatenate(
        (
            comparison.response,
            comparison.ols.predictions,
            comparison.lad.predictions,
        )
    )
    lower = float(values.min())
    upper = float(values.max())
    padding = max((upper - lower) * 0.05, 1e-9)
    return lower - padding, upper + padding


def _draw_identity(ax: plt.Axes, limits: tuple[float, float]) -> None:
    """Draw the perfect-prediction line and apply common axes."""

    ax.plot(
        limits,
        limits,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label="Perfect prediction",
        zorder=1,
    )
    ax.set(xlim=limits, ylim=limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)


def save_baseline_figure(comparison: BaselineComparison) -> None:
    """Save side-by-side OLS and LAD actual-versus-predicted plots."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    limits = _axis_limits(comparison)

    for ax, (model_name, predictions) in zip(
        axes,
        (
            ("OLS", comparison.ols.predictions),
            ("LAD", comparison.lad.predictions),
        ),
    ):
        style = MODEL_STYLES[model_name]
        metrics = regression_metrics(comparison.response, predictions)
        ax.scatter(
            comparison.response,
            predictions,
            s=31,
            alpha=0.58,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.25,
            label=model_name,
            zorder=2,
        )
        _draw_identity(ax, limits)
        ax.set(
            xlabel=f"Actual {comparison.spec.response}",
            ylabel=f"Predicted {comparison.spec.response}",
            title=(
                f"{model_name}\n"
                f"MAE = {metrics['MAE']:.3g}; RMSE = {metrics['RMSE']:.3g}"
            ),
        )
        ax.legend(loc="best")

    fig.suptitle(
        f"{comparison.spec.name}: full-data OLS and LAD fits",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        FIGURE_DIR / f"{comparison.spec.slug}_baseline_ols_lad_fit.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_merged_baseline_figure(
    comparisons: list[BaselineComparison],
) -> None:
    """Save one comparison panel per dataset with both fitted models."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, comparison in zip(axes.flat, comparisons):
        limits = _axis_limits(comparison)
        ols_metrics = regression_metrics(
            comparison.response,
            comparison.ols.predictions,
        )
        lad_metrics = regression_metrics(
            comparison.response,
            comparison.lad.predictions,
        )

        for model_name, predictions in (
            ("OLS", comparison.ols.predictions),
            ("LAD", comparison.lad.predictions),
        ):
            style = MODEL_STYLES[model_name]
            ax.scatter(
                comparison.response,
                predictions,
                s=29,
                alpha=0.48,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.2,
                label=model_name,
                zorder=2,
            )

        _draw_identity(ax, limits)
        ax.set(
            xlabel=f"Actual {comparison.spec.response}",
            ylabel=f"Predicted {comparison.spec.response}",
            title=(
                f"{comparison.spec.name}\n"
                f"MAE: OLS {ols_metrics['MAE']:.3g}, "
                f"LAD {lad_metrics['MAE']:.3g} | "
                f"RMSE: OLS {ols_metrics['RMSE']:.3g}, "
                f"LAD {lad_metrics['RMSE']:.3g}"
            ),
        )
        ax.legend(loc="best")

    fig.suptitle(
        "Full-data OLS versus least-absolute-deviation regression",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        FIGURE_DIR / "merged_baseline_ols_lad_fits.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
