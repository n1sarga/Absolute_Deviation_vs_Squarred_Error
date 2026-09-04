"""Classify leverage points and conditional response outliers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import RobustScaler

from .config import (
    CUTOFF_PROBABILITY,
    RANDOM_STATE,
    RESIDUAL_Z_CUTOFF,
    SUPPORT_FRACTION,
    DatasetSpec,
)
from .detection import known_outlier_mask


@dataclass
class RegressionOutlierClassification:
    """Predictor-leverage and response-residual diagnostics for one dataset."""

    spec: DatasetSpec
    result: pd.DataFrame
    record_ids: np.ndarray
    projection: np.ndarray
    explained_variance: np.ndarray
    leverage_distance: np.ndarray
    leverage_cutoff: float
    residual_score: np.ndarray
    residual_cutoff: float
    leverage_flag: np.ndarray
    response_flag: np.ndarray
    category: np.ndarray
    known: np.ndarray | None


def robust_residual_score(residuals: np.ndarray) -> np.ndarray:
    """Return absolute residual deviations measured in robust scale units."""

    center = float(np.median(residuals))
    deviations = np.abs(residuals - center)
    mad = float(np.median(deviations))
    scale = 1.4826 * mad

    if scale <= np.finfo(float).eps:
        q1, q3 = np.percentile(residuals, [25, 75])
        scale = float((q3 - q1) / 1.349)
    if scale <= np.finfo(float).eps:
        scale = float(np.std(residuals, ddof=1))
    if scale <= np.finfo(float).eps:
        return np.zeros_like(residuals, dtype=float)

    return deviations / scale


def assign_categories(
    leverage_flag: np.ndarray,
    response_flag: np.ndarray,
) -> np.ndarray:
    """Assign one mutually exclusive diagnostic class to every row."""

    return np.select(
        [
            leverage_flag & response_flag,
            leverage_flag,
            response_flag,
        ],
        [
            "Leverage + response",
            "Leverage only",
            "Response only",
        ],
        default="Regular",
    )


def classify_regression_outliers(
    spec: DatasetSpec,
    data: pd.DataFrame,
) -> RegressionOutlierClassification:
    """Separate predictor-space leverage from conditional response outliers.

    Predictor leverage uses robust Mahalanobis distance in X-space. Conditional
    response outliers use absolute robust z-scores of residuals from an
    unpenalized median regression. The median regression is diagnostic only;
    the dedicated OLS-versus-LAD experiment is a later research stage.
    """

    predictors = data[list(spec.predictors)].astype(float)
    response = data[spec.response].to_numpy(dtype=float)
    scaled_predictors = RobustScaler().fit_transform(predictors)

    leverage_model = MinCovDet(
        support_fraction=SUPPORT_FRACTION,
        random_state=RANDOM_STATE,
    ).fit(scaled_predictors)
    leverage_distance = np.sqrt(leverage_model.mahalanobis(scaled_predictors))
    leverage_cutoff = float(
        np.sqrt(chi2.ppf(CUTOFF_PROBABILITY, df=len(spec.predictors)))
    )
    leverage_flag = leverage_distance > leverage_cutoff

    median_model = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        fit_intercept=True,
        solver="highs",
    ).fit(scaled_predictors, response)
    prediction = median_model.predict(scaled_predictors)
    residual = response - prediction
    residual_score = robust_residual_score(residual)
    response_flag = residual_score > RESIDUAL_Z_CUTOFF
    category = assign_categories(leverage_flag, response_flag)
    known_mask = known_outlier_mask(spec, data)

    pca = PCA(n_components=2).fit(scaled_predictors)
    projection = pca.transform(scaled_predictors)
    record_ids = (
        data[spec.id_column].to_numpy()
        if spec.id_column
        else np.arange(1, len(data) + 1)
    )

    result = data.copy()
    result.insert(0, "RecordID", record_ids)
    result["LeverageDistance"] = leverage_distance
    result["LeverageCutoff"] = leverage_cutoff
    result["LeverageFlag"] = leverage_flag
    result["MedianRegressionPrediction"] = prediction
    result["MedianRegressionResidual"] = residual
    result["ResidualRobustZ"] = residual_score
    result["ResidualZCutoff"] = RESIDUAL_Z_CUTOFF
    result["ResponseOutlierFlag"] = response_flag
    result["OutlierClass"] = category
    if known_mask is not None:
        result["KnownOutlier"] = known_mask

    return RegressionOutlierClassification(
        spec=spec,
        result=result,
        record_ids=record_ids,
        projection=projection,
        explained_variance=pca.explained_variance_ratio_,
        leverage_distance=leverage_distance,
        leverage_cutoff=leverage_cutoff,
        residual_score=residual_score,
        residual_cutoff=RESIDUAL_Z_CUTOFF,
        leverage_flag=leverage_flag,
        response_flag=response_flag,
        category=category,
        known=known_mask,
    )


def classification_summary_row(
    classification: RegressionOutlierClassification,
) -> dict[str, object]:
    """Build one summary row from mutually exclusive class counts."""

    counts = pd.Series(classification.category).value_counts()
    row = {
        "Dataset": classification.spec.name,
        "Rows": len(classification.category),
        "Predictors": len(classification.spec.predictors),
        "LeverageCutoff": classification.leverage_cutoff,
        "ResidualZCutoff": classification.residual_cutoff,
        "Regular": int(counts.get("Regular", 0)),
        "LeverageOnly": int(counts.get("Leverage only", 0)),
        "ResponseOnly": int(counts.get("Response only", 0)),
        "LeverageAndResponse": int(counts.get("Leverage + response", 0)),
        "AnyFlagged": int(np.logical_or(
            classification.leverage_flag,
            classification.response_flag,
        ).sum()),
    }
    if classification.known is not None:
        flagged = classification.leverage_flag | classification.response_flag
        row.update(
            {
                "KnownOutliers": int(classification.known.sum()),
                "KnownClassified": int(
                    np.logical_and(flagged, classification.known).sum()
                ),
                "AdditionalClassified": int(
                    np.logical_and(flagged, ~classification.known).sum()
                ),
                "KnownMissed": int(
                    np.logical_and(~flagged, classification.known).sum()
                ),
            }
        )
    return row
