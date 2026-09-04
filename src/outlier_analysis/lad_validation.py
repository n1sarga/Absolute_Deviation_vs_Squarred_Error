"""Validate the custom LP-based LAD estimator against median regression."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from .config import DATASETS, RESULT_DIR, DatasetSpec
from .detection import load_dataset
from .lad import LADFit, fit_lad_linear_program


@dataclass(frozen=True)
class LADValidation:
    """Custom and reference LAD fits for one configured dataset."""

    spec: DatasetSpec
    lp_fit: LADFit
    reference_intercept: float
    reference_coefficients: np.ndarray
    reference_predictions: np.ndarray
    reference_residuals: np.ndarray
    reference_objective: float
    record_ids: np.ndarray

    @property
    def objective_difference(self) -> float:
        return abs(self.lp_fit.absolute_error_sum - self.reference_objective)

    @property
    def objective_relative_difference(self) -> float:
        denominator = max(1.0, abs(self.reference_objective))
        return self.objective_difference / denominator

    @property
    def objective_match(self) -> bool:
        return bool(
            np.isclose(
                self.lp_fit.absolute_error_sum,
                self.reference_objective,
                rtol=1e-8,
                atol=1e-7,
            )
        )


def validate_lad_for_dataset(
    spec: DatasetSpec,
    data: pd.DataFrame,
) -> LADValidation:
    """Fit custom and reference LAD models under identical robust scaling."""

    predictors = data[list(spec.predictors)].to_numpy(dtype=float)
    response = data[spec.response].to_numpy(dtype=float)
    lp_fit = fit_lad_linear_program(predictors, response)

    scaled_x = lp_fit.scaling.transform_predictors(predictors)
    scaled_y = lp_fit.scaling.transform_response(response)
    reference = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        fit_intercept=True,
        solver="highs",
    ).fit(scaled_x, scaled_y)
    reference_intercept, reference_coefficients = (
        lp_fit.scaling.restore_parameters(
            float(reference.intercept_),
            np.asarray(reference.coef_, dtype=float),
        )
    )
    reference_predictions = (
        reference_intercept + predictors @ reference_coefficients
    )
    reference_residuals = response - reference_predictions
    reference_objective = float(np.abs(reference_residuals).sum())
    record_ids = (
        data[spec.id_column].to_numpy()
        if spec.id_column
        else np.arange(1, len(data) + 1)
    )

    return LADValidation(
        spec=spec,
        lp_fit=lp_fit,
        reference_intercept=reference_intercept,
        reference_coefficients=reference_coefficients,
        reference_predictions=reference_predictions,
        reference_residuals=reference_residuals,
        reference_objective=reference_objective,
        record_ids=record_ids,
    )


def validation_summary_row(validation: LADValidation) -> dict[str, object]:
    """Build dataset-level solver validation metrics."""

    lp_parameters = np.concatenate(
        ([validation.lp_fit.intercept], validation.lp_fit.coefficients)
    )
    reference_parameters = np.concatenate(
        ([validation.reference_intercept], validation.reference_coefficients)
    )
    return {
        "Dataset": validation.spec.name,
        "Rows": len(validation.record_ids),
        "Predictors": len(validation.spec.predictors),
        "LPObjectiveSAE": validation.lp_fit.absolute_error_sum,
        "ReferenceObjectiveSAE": validation.reference_objective,
        "ObjectiveAbsoluteDifference": validation.objective_difference,
        "ObjectiveRelativeDifference": validation.objective_relative_difference,
        "MaxParameterDifference": float(
            np.max(np.abs(lp_parameters - reference_parameters))
        ),
        "MaxPredictionDifference": float(
            np.max(
                np.abs(
                    validation.lp_fit.predictions
                    - validation.reference_predictions
                )
            )
        ),
        "LPIterations": validation.lp_fit.iterations,
        "ObjectiveMatch": validation.objective_match,
    }


def prediction_table(validation: LADValidation) -> pd.DataFrame:
    """Build row-level predictions from custom and reference LAD fits."""

    return pd.DataFrame(
        {
            "RecordID": validation.record_ids,
            "Actual": (
                validation.lp_fit.predictions + validation.lp_fit.residuals
            ),
            "LADPrediction": validation.lp_fit.predictions,
            "ReferencePrediction": validation.reference_predictions,
            "LADResidual": validation.lp_fit.residuals,
            "ReferenceResidual": validation.reference_residuals,
            "PredictionDifference": (
                validation.lp_fit.predictions
                - validation.reference_predictions
            ),
        }
    )


def run_validation() -> pd.DataFrame:
    """Validate all datasets and save summary, predictions, and figure."""

    from .lad_plotting import save_lad_validation_figure

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    validations: list[LADValidation] = []
    rows: list[dict[str, object]] = []

    for spec in DATASETS:
        validation = validate_lad_for_dataset(spec, load_dataset(spec))
        if not validation.objective_match:
            raise RuntimeError(f"LAD objective mismatch for {spec.name}")
        validations.append(validation)
        rows.append(validation_summary_row(validation))
        prediction_table(validation).to_csv(
            RESULT_DIR / f"{spec.slug}_lad_validation_predictions.csv",
            index=False,
            float_format="%.10f",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULT_DIR / "lad_solver_validation.csv",
        index=False,
        float_format="%.10g",
    )
    save_lad_validation_figure(validations)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    """Console-script entry point."""

    run_validation()
