# Regression by Minimizing Absolute Deviation

Reproducible research code and datasets for studying robust regression and
multivariate outliers in Boston Housing, Concrete Strength, Hawkins-Bradu-Kass
(HBK), and a synthetic OLS/LAD dataset.

The current workflow robust-scales each dataset, estimates its multivariate
center and covariance with Minimum Covariance Determinant (MCD), and flags rows
whose robust Mahalanobis distance exceeds the 97.5% chi-square cutoff. PCA is
used only to visualize the distributions.

## Joint data-distribution outliers

![Merged robust multivariate outlier distributions for all four datasets](outputs/figures/merged_outlier_distributions.png)

*Figure 1. Robust-scaled PCA distributions for the four datasets. Blue points
are inliers; red points are observations flagged as multivariate outliers.*

## Regression outlier classification

![Merged leverage and response outlier classifications](outputs/figures/merged_outlier_classifications.png)

*Figure 2. Predictor-space leverage versus conditional response deviation.
Blue = regular, yellow = leverage only, red = response only, and purple = both.*

## LP-based LAD solver

![LP-based LAD predictions compared with reference median regression](outputs/figures/lad_solver_validation.png)

*Figure 3. Custom sparse linear-programming LAD predictions versus unpenalized
median-regression reference predictions. All four objectives agree within
floating-point error.*

## Full-data OLS and LAD baselines

![Full-data OLS and LAD baseline fits for all four datasets](outputs/figures/merged_baseline_ols_lad_fits.png)

*Figure 4. Actual responses versus full-data OLS and LAD predictions. OLS
minimizes squared error; LAD minimizes absolute error. These in-sample fits are
descriptive baselines, not estimates of performance on unseen data.*

## Sample-condition and contamination experiments

![OLS and LAD sensitivity to observed and artificial outliers](outputs/figures/merged_condition_contamination_comparison.png)

*Figure 5. Clean-reference MAE after contaminating 0%, 5%, 10%, or 20% of
regular-row responses. Solid lines are means across 30 repetitions with
one-standard-deviation error bars. Dotted lines show complete-data fits on the
same regular rows.*

## Formal model comparison

![Formal OLS versus LAD comparison](outputs/figures/model_comparison_dashboard.png)

*Figure 6. LAD reductions in clean-reference error, prediction drift, and
standardized slope shift at 20% response contamination, plus measured runtime
cost relative to OLS. Positive reductions favor LAD.*

## Final out-of-fold validation

![Paired-bootstrap MAE differences for OLS and LAD](outputs/figures/bootstrap_mae_difference.png)

*Figure 7. LAD-minus-OLS MAE from repeat-averaged out-of-fold predictions with
paired-bootstrap 95% confidence intervals. Negative values favor LAD.*

LAD had lower out-of-fold MAE point estimates for all datasets (1.0%–13.9%
reduction). Only Boston's unadjusted bootstrap interval excluded zero; no
dataset passed Holm-adjusted family-wise alpha 0.05. Main conclusion: LAD gives
strong vertical-outlier stability, while universal out-of-sample superiority is
not established. OLS remains faster and has lower out-of-fold RMSE.

## Repository layout

```text
.
|-- .github/workflows/       # Automated tests
|-- data/
|   |-- raw/                 # Original downloaded files
|   `-- processed/           # Analysis-ready CSV files
|-- docs/                    # Method documentation
|-- outputs/
|   |-- figures/             # Individual and merged diagrams
|   `-- results/             # Row-level flags and summary CSV
|-- scripts/                 # User-facing command-line scripts
|-- src/outlier_analysis/    # Reusable Python package
|-- tests/                   # Automated regression tests
|-- pyproject.toml           # Package and test configuration
`-- requirements*.txt        # Runtime and development setup
```

## Quick start

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/analyze_outliers.py
python scripts/validate_lad.py
python scripts/fit_baseline_models.py
python scripts/run_robustness_experiments.py
python scripts/evaluate_models.py
python scripts/run_final_analysis.py
```

After installation, this equivalent command is also available:

```powershell
analyze-outliers
validate-lad
fit-baseline-models
run-robustness-experiments
evaluate-models
run-final-analysis
```

## Test

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Outputs

- `outputs/figures/`: one diagram per dataset and one merged diagram;
- `outputs/results/`: joint outlier results plus regression-specific leverage and
  response classifications, LAD validation results, and full-data OLS/LAD
  metrics, coefficients, predictions, and contamination sensitivity results.

Blue points are inliers. Red points are robust-distance outliers. A flag means
that a row is unusual relative to the central multivariate pattern; it does not
automatically mean the observation is erroneous.

See [outlier method notes](docs/outlier_method.md) and
[LAD solver notes](docs/lad_solver.md) for calculation details. The
[baseline model notes](docs/baseline_models.md) describe the full-data OLS/LAD
comparison. The [robustness experiment notes](docs/robustness_experiments.md)
define the full-data, inlier-only, and controlled-contamination protocol. See
[formal comparison notes](docs/formal_comparison.md) for paired effects and
runtime definitions. [Final validation notes](docs/final_validation.md) define
cross-validation, bootstrap intervals, multiplicity correction, and limits.
[Final research report](docs/final_research_report.md) provides full synthesis.
