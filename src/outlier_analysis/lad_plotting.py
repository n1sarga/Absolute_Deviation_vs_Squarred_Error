"""Plot parity between custom LP and reference LAD predictions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import FIGURE_DIR
from .lad_validation import LADValidation


def save_lad_validation_figure(validations: list[LADValidation]) -> None:
    """Save a two-by-two prediction parity figure for all datasets."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, validation in zip(axes.flat, validations):
        reference = validation.reference_predictions
        custom = validation.lp_fit.predictions
        lower = float(min(reference.min(), custom.min()))
        upper = float(max(reference.max(), custom.max()))
        padding = max((upper - lower) * 0.04, 1e-9)

        ax.scatter(
            reference,
            custom,
            s=32,
            alpha=0.62,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            label="Exact agreement",
        )
        ax.set(
            xlim=(lower - padding, upper + padding),
            ylim=(lower - padding, upper + padding),
            xlabel="Reference median-regression prediction",
            ylabel="Custom LP LAD prediction",
            title=(
                f"{validation.spec.name}\n"
                f"relative objective difference = "
                f"{validation.objective_relative_difference:.2e}"
            ),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(
        "LP-based LAD validation against median regression",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        FIGURE_DIR / "lad_solver_validation.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)
