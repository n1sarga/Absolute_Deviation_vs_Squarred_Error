"""Dataset definitions and shared analysis settings."""

from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
SUPPORT_FRACTION = 0.70
CUTOFF_PROBABILITY = 0.975

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"

STATUS_COLUMN = "OutlierStatus"
DISTANCE_COLUMN = "RobustDistance"
FLAG_COLUMN = "OutlierFlag"
COLORS = {"Inlier": "#4C78A8", "Outlier": "#E45756"}


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one input dataset and the columns used by the detector."""

    name: str
    slug: str
    path: Path
    variables: tuple[str, ...]
    id_column: str | None = None
    read_kwargs: dict | None = None
    known_label: str | None = None


DATASETS = (
    DatasetSpec(
        name="Boston Housing",
        slug="boston_housing",
        path=PROJECT_ROOT / "data" / "processed" / "boston_housing.csv",
        # B excluded because it encodes an ethically problematic racial assumption.
        variables=(
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "LSTAT", "MEDV",
        ),
        read_kwargs={"skiprows": 1},
    ),
    DatasetSpec(
        name="Concrete Strength",
        slug="concrete_strength",
        path=PROJECT_ROOT / "data" / "processed" / "concrete_strength.csv",
        variables=(
            "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
            "Superplasticizer", "CoarseAggregate", "FineAggregate",
            "Age", "Strength",
        ),
    ),
    DatasetSpec(
        name="Hawkins-Bradu-Kass (HBK)",
        slug="hbk",
        path=PROJECT_ROOT / "data" / "processed" / "hbk.csv",
        variables=("X1", "X2", "X3", "Y"),
        id_column="Observation",
        known_label="hbk",
    ),
    DatasetSpec(
        name="Synthetic OLS-LAD",
        slug="synthetic",
        path=PROJECT_ROOT / "data" / "processed" / "synthetic_ols_lad_outliers.csv",
        variables=("X1", "X2", "X3", "y"),
        id_column="ID",
    ),
)
