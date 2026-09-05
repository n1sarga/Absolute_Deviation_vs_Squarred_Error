# Data

The active literature-aligned analysis reads from `processed/`. Files under `raw/` preserve available source downloads and are not modified by the analysis.

## Processed datasets

| File | Purpose |
|---|---|
| `boston_housing.csv` | Boston Housing regression data |
| `concrete_strength.csv` | Concrete Compressive Strength regression data |
| `hbk.csv` | Hawkins-Bradu-Kass regression data |
| `synthetic_ols_lad_outliers.csv` | Synthetic multivariate OLS/LAD regression data |

## Minimal preparation

The rebuilt project deliberately avoids a separate outlier-detection pipeline. Identifier or descriptive columns are excluded when they are not regression variables, missing rows are removed transparently, and numeric predictors are passed directly to OLS and LAD.

Extreme observations are not automatically deleted because the project studies how squared-error and absolute-error regression respond to large errors.

See [`docs/experimental_design.md`](../docs/experimental_design.md) and [`docs/literature_mapping.md`](../docs/literature_mapping.md) for the active methodology.
