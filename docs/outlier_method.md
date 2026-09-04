# Outlier Analysis

Unified row-level outlier detection for Boston Housing, Concrete Strength, HBK,
and supplied synthetic data.

## Method

1. Exclude identifiers and text labels.
2. Robust-scale all measurement and response columns.
3. Fit Minimum Covariance Determinant with 70% support.
4. Calculate robust Mahalanobis distance in full-dimensional space.
5. Flag rows beyond 97.5% chi-square cutoff for dataset dimensionality.
6. Project robust-scaled data to two PCA dimensions for display only.

PCA does not determine flags. Red points are robust-distance outliers; blue
points are inliers. Flags mean unusual relative to central multivariate pattern,
not necessarily data errors. Skewed or multi-cluster datasets can produce many
flags because ellipsoidal-distance assumptions fit them poorly.

Boston variable `B` is excluded because it encodes an ethically problematic
racial assumption. All other Boston predictor and response columns are used.

For the updated synthetic dataset, `X1`, `X2`, `X3`, and `y` are analyzed
jointly; `ID` is used only as a row identifier.

## Run

```powershell
python scripts/analyze_outliers.py
```

## Code layout

```text
scripts/analyze_outliers.py       # Small command-line entry point
src/outlier_analysis/
|-- config.py                    # Settings and dataset definitions
|-- detection.py                 # Loading, validation, and calculation
|-- plotting.py                  # Individual and merged diagrams
`-- pipeline.py                  # Execution, CSV output, and summaries
```

Outputs:

- four individual distribution diagrams in `outputs/figures/`;
- `outputs/figures/merged_outlier_distributions.png`;
- one row-level flagged CSV per dataset in `outputs/results/`;
- `outputs/results/outlier_summary.csv`.
