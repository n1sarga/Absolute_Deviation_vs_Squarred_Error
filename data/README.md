# Data

The active literature-aligned empirical analysis uses only the three files listed below. Files under `raw/` preserve available source downloads and are not modified by the analysis.

## Active processed datasets

| File | Purpose |
|---|---|
| `boston_housing.csv` | Boston Housing regression data |
| `concrete_strength.csv` | Concrete Compressive Strength regression data |
| `hbk.csv` | Hawkins-Bradu-Kass regression data |

The previously included `synthetic_ols_lad_outliers.csv` file is no longer part of the project experiment.

## Minimal preparation

The project deliberately avoids a separate outlier-detection pipeline. Identifier or descriptive columns are excluded when they are not regression variables, missing rows are removed transparently, and numeric predictors are passed directly to OLS and LAD.

Extreme observations are not automatically deleted because the project studies how squared-error and absolute-error regression respond to large errors.

Controlled simulations used for the contamination and error-distribution experiments are generated in code and are not treated as empirical datasets.

See [`docs/experimental_design.md`](../docs/experimental_design.md) and [`docs/literature_mapping.md`](../docs/literature_mapping.md) for the active methodology.
