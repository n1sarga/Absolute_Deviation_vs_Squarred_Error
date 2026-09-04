"""Load, validate, and analyze datasets for multivariate outliers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .config import (
    CUTOFF_PROBABILITY,
    DISTANCE_COLUMN,
    FLAG_COLUMN,
    RANDOM_STATE,
    STATUS_COLUMN,
    SUPPORT_FRACTION,
    DatasetSpec,
)


@dataclass
class OutlierAnalysis:
    """Calculated values needed for tables, summaries, and plots."""

    spec: DatasetSpec
    data: pd.DataFrame
    result: pd.DataFrame
    record_ids: np.ndarray
    projection: np.ndarray
    explained_variance: np.ndarray
    distance: np.ndarray
    cutoff: float
    flag: np.ndarray
    known: np.ndarray | None


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    """Read one CSV and reject missing or invalid measurement values."""

    if not spec.path.exists():
        raise FileNotFoundError(f"Dataset missing: {spec.path}")

    data = pd.read_csv(spec.path, **(spec.read_kwargs or {}))
    missing_columns = set(spec.variables) - set(data.columns)
    if missing_columns:
        raise ValueError(f"{spec.name} missing columns: {sorted(missing_columns)}")
    if data[list(spec.variables)].isna().any().any():
        raise ValueError(f"{spec.name} contains missing measurement values.")
    if not np.isfinite(data[list(spec.variables)].to_numpy(dtype=float)).all():
        raise ValueError(f"{spec.name} contains non-finite measurement values.")
    return data


def known_outlier_mask(spec: DatasetSpec, data: pd.DataFrame) -> np.ndarray | None:
    """Return published ground-truth flags when a dataset provides them."""

    if spec.known_label == "hbk":
        return data["Observation"].le(14).to_numpy()
    return None


def calculate_outliers(spec: DatasetSpec, data: pd.DataFrame) -> OutlierAnalysis:
    """Calculate robust distances, flags, and a display-only PCA projection."""

    measurements = data[list(spec.variables)].astype(float)
    scaled = RobustScaler().fit_transform(measurements)

    detector = MinCovDet(
        support_fraction=SUPPORT_FRACTION,
        random_state=RANDOM_STATE,
    ).fit(scaled)
    robust_distance = np.sqrt(detector.mahalanobis(scaled))
    cutoff = float(np.sqrt(chi2.ppf(CUTOFF_PROBABILITY, df=len(spec.variables))))
    outlier_flag = robust_distance > cutoff

    pca = PCA(n_components=2).fit(scaled)
    projection = pca.transform(scaled)
    record_ids = (
        data[spec.id_column].to_numpy()
        if spec.id_column
        else np.arange(1, len(data) + 1)
    )
    known_mask = known_outlier_mask(spec, data)

    result = data.copy()
    result.insert(0, "RecordID", record_ids)
    result[DISTANCE_COLUMN] = robust_distance
    result["DistanceCutoff"] = cutoff
    result[FLAG_COLUMN] = outlier_flag
    result[STATUS_COLUMN] = np.where(outlier_flag, "Outlier", "Inlier")
    if known_mask is not None:
        result["KnownOutlier"] = known_mask

    return OutlierAnalysis(
        spec=spec,
        data=data,
        result=result,
        record_ids=record_ids,
        projection=projection,
        explained_variance=pca.explained_variance_ratio_,
        distance=robust_distance,
        cutoff=cutoff,
        flag=outlier_flag,
        known=known_mask,
    )
