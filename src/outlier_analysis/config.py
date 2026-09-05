"""Dataset definitions and shared analysis settings."""

from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
SUPPORT_FRACTION = 0.70
CUTOFF_PROBABILITY = 0.975
RESIDUAL_Z_CUTOFF = 3.5
CONTAMINATION_RATES = (0.0, 0.05, 0.10, 0.20)
CONTAMINATION_REPETITIONS = 30
CONTAMINATION_SCALE_MULTIPLIER = 10.0
RUNTIME_REPETITIONS = 30
RUNTIME_WARMUPS = 3
CV_SPLITS = 5
CV_REPEATS = 10
BOOTSTRAP_REPETITIONS = 5000
RANDOMIZATION_REPETITIONS = 20000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"

STATUS_COLUMN = "OutlierStatus"
DISTANCE_COLUMN = "RobustDistance"
FLAG_COLUMN = "OutlierFlag"
COLORS = {"Inlier": "#4C78A8", "Outlier": "#E45756"}
CLASS_COLORS = {
    "Regular": "#4C78A8",
    "Leverage only": "#F2CF5B",
    "Response only": "#E45756",
    "Leverage + response": "#B279A2",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one input dataset and the columns used by the detector."""

    name: str
    slug: str
    path: Path
    predictors: tuple[str, ...]
    response: str
    id_column: str | None = None
    read_kwargs: dict | None = None
    known_label: str | None = None

    @property
    def variables(self) -> tuple[str, ...]:
        """All numeric columns used by the joint distribution analysis."""

        return self.predictors + (self.response,)


DATASETS = (
    DatasetSpec(
        name="Boston Housing",
        slug="boston_housing",
        path=PROJECT_ROOT / "data" / "processed" / "boston_housing.csv",
        # B excluded because it encodes an ethically problematic racial assumption.
        predictors=(
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "LSTAT",
        ),
        response="MEDV",
        read_kwargs={"skiprows": 1},
    ),
    DatasetSpec(
        name="Concrete Strength",
        slug="concrete_strength",
        path=PROJECT_ROOT / "data" / "processed" / "concrete_strength.csv",
        predictors=(
            "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
            "Superplasticizer", "CoarseAggregate", "FineAggregate",
            "Age",
        ),
        response="Strength",
    ),
    DatasetSpec(
        name="Hawkins-Bradu-Kass (HBK)",
        slug="hbk",
        path=PROJECT_ROOT / "data" / "processed" / "hbk.csv",
        predictors=("X1", "X2", "X3"),
        response="Y",
        id_column="Observation",
        known_label="hbk",
    ),
    DatasetSpec(
        name="Synthetic OLS-LAD",
        slug="synthetic",
        path=PROJECT_ROOT / "data" / "processed" / "synthetic_ols_lad_outliers.csv",
        predictors=("X1", "X2", "X3"),
        response="y",
        id_column="ID",
    ),
)
