"""Controlled response-contamination mechanics and summaries."""

import numpy as np
import pandas as pd

from .baseline import regression_metrics
from .config import (
    CONTAMINATION_RATES,
    CONTAMINATION_REPETITIONS,
    CONTAMINATION_SCALE_MULTIPLIER,
    RANDOM_STATE,
)
from .robustness_conditions import (
    ConditionFits,
    RegressionFit,
    fit_models,
    predict,
)


def contaminated_row_count(row_count: int, rate: float) -> int:
    """Convert a positive rate to a nonzero nearest-integer row count."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("contamination rate must be between zero and one")
    if rate == 0.0:
        return 0
    return min(row_count, max(1, int(np.floor(row_count * rate + 0.5))))


def contaminate_response(
    response: np.ndarray,
    rate: float,
    noise_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add zero-mean Gaussian gross errors to a fixed fraction of responses."""

    y = np.asarray(response, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("response must be a nonempty finite vector")
    if noise_sd <= 0.0 or not np.isfinite(noise_sd):
        raise ValueError("noise standard deviation must be positive and finite")

    count = contaminated_row_count(len(y), rate)
    contaminated = y.copy()
    if count == 0:
        return contaminated, np.array([], dtype=int), np.array([], dtype=float)

    indices = np.sort(rng.choice(len(y), size=count, replace=False))
    noise = rng.normal(loc=0.0, scale=noise_sd, size=count)
    contaminated[indices] += noise
    return contaminated, indices, noise


def _standardized_slopes(
    fit: RegressionFit,
    predictor_scale: np.ndarray,
    response_scale: float,
) -> np.ndarray:
    """Scale slopes against the fixed clean-reference IQRs."""

    return fit.coefficients * predictor_scale / response_scale


def contamination_experiment(
    condition: ConditionFits,
    dataset_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated response-contamination fits on regular observations."""

    predictors = condition.inlier_predictors
    response = condition.inlier_response
    record_ids = condition.inlier_record_ids
    predictor_q1, predictor_q3 = np.percentile(
        predictors,
        [25, 75],
        axis=0,
    )
    predictor_scale = predictor_q3 - predictor_q1
    predictor_scale = np.where(
        predictor_scale > np.finfo(float).eps,
        predictor_scale,
        1.0,
    )
    response_q1, response_q3 = np.percentile(response, [25, 75])
    response_scale = float(response_q3 - response_q1)
    if response_scale <= np.finfo(float).eps:
        response_scale = 1.0
    noise_sd = CONTAMINATION_SCALE_MULTIPLIER * response_scale

    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    clean_predictions = {
        model_name: predict(fit, predictors)
        for model_name, fit in condition.inlier_models.items()
    }
    clean_standardized_slopes = {
        model_name: _standardized_slopes(
            fit,
            predictor_scale,
            response_scale,
        )
        for model_name, fit in condition.inlier_models.items()
    }

    for rate in CONTAMINATION_RATES:
        for repetition in range(1, CONTAMINATION_REPETITIONS + 1):
            seed = np.random.SeedSequence(
                [RANDOM_STATE, dataset_number, int(rate * 10_000), repetition]
            )
            contaminated_response, indices, added_noise = contaminate_response(
                response,
                rate,
                noise_sd,
                np.random.default_rng(seed),
            )
            models = (
                condition.inlier_models
                if rate == 0.0
                else fit_models(predictors, contaminated_response)
            )

            design_rows.extend(
                {
                    "Dataset": condition.spec.name,
                    "ContaminationRate": rate,
                    "Repetition": repetition,
                    "RecordID": record_ids[index],
                    "OriginalResponse": response[index],
                    "AddedNoise": noise,
                    "ContaminatedResponse": contaminated_response[index],
                }
                for index, noise in zip(indices, added_noise)
            )

            for model_name, fit in models.items():
                fitted_predictions = predict(fit, predictors)
                clean_metrics = regression_metrics(response, fitted_predictions)
                contaminated_metrics = regression_metrics(
                    contaminated_response,
                    fitted_predictions,
                )
                standardized_slopes = _standardized_slopes(
                    fit,
                    predictor_scale,
                    response_scale,
                )
                slope_shift = (
                    standardized_slopes
                    - clean_standardized_slopes[model_name]
                )
                metric_rows.append(
                    {
                        "Dataset": condition.spec.name,
                        "Model": model_name,
                        "ContaminationRate": rate,
                        "ContaminationPercent": 100.0 * rate,
                        "Repetition": repetition,
                        "Rows": len(response),
                        "ContaminatedRows": len(indices),
                        "NoiseSD": noise_sd,
                        **{
                            f"Clean{metric}": value
                            for metric, value in clean_metrics.items()
                        },
                        "ContaminatedTrainingMAE": contaminated_metrics["MAE"],
                        "ContaminatedTrainingRMSE": contaminated_metrics["RMSE"],
                        "PredictionRMSEFromCleanFit": float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        fitted_predictions
                                        - clean_predictions[model_name]
                                    )
                                )
                            )
                        ),
                        "StandardizedSlopeL2Shift": float(
                            np.linalg.norm(slope_shift)
                        ),
                    }
                )

                coefficient_rows.append(
                    {
                        "Dataset": condition.spec.name,
                        "Model": model_name,
                        "ContaminationRate": rate,
                        "Repetition": repetition,
                        "Term": "Intercept",
                        "Estimate": fit.intercept,
                        "CleanReferenceEstimate": condition.inlier_models[
                            model_name
                        ].intercept,
                        "EstimateShift": (
                            fit.intercept
                            - condition.inlier_models[model_name].intercept
                        ),
                        "RobustStandardizedEstimate": float("nan"),
                        "CleanRobustStandardizedEstimate": float("nan"),
                    }
                )
                coefficient_rows.extend(
                    {
                        "Dataset": condition.spec.name,
                        "Model": model_name,
                        "ContaminationRate": rate,
                        "Repetition": repetition,
                        "Term": term,
                        "Estimate": estimate,
                        "CleanReferenceEstimate": clean_estimate,
                        "EstimateShift": estimate - clean_estimate,
                        "RobustStandardizedEstimate": scaled_estimate,
                        "CleanRobustStandardizedEstimate": clean_scaled,
                    }
                    for (
                        term,
                        estimate,
                        clean_estimate,
                        scaled_estimate,
                        clean_scaled,
                    ) in zip(
                        condition.spec.predictors,
                        fit.coefficients,
                        condition.inlier_models[model_name].coefficients,
                        standardized_slopes,
                        clean_standardized_slopes[model_name],
                    )
                )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(design_rows),
    )


def summarize_contamination(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated clean-data sensitivity measures."""

    summary = (
        metrics.groupby(
            ["Dataset", "Model", "ContaminationRate", "ContaminationPercent"],
            sort=False,
        )
        .agg(
            Repetitions=("Repetition", "nunique"),
            ContaminatedRows=("ContaminatedRows", "first"),
            CleanMAEMean=("CleanMAE", "mean"),
            CleanMAEStd=("CleanMAE", "std"),
            CleanRMSEMean=("CleanRMSE", "mean"),
            CleanRMSEStd=("CleanRMSE", "std"),
            PredictionDriftMean=("PredictionRMSEFromCleanFit", "mean"),
            PredictionDriftStd=("PredictionRMSEFromCleanFit", "std"),
            SlopeShiftMean=("StandardizedSlopeL2Shift", "mean"),
            SlopeShiftStd=("StandardizedSlopeL2Shift", "std"),
        )
        .reset_index()
    )
    standard_deviation_columns = [
        column for column in summary.columns if column.endswith("Std")
    ]
    summary[standard_deviation_columns] = summary[
        standard_deviation_columns
    ].fillna(0.0)
    return summary
