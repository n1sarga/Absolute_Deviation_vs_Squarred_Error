# Regression by Minimizing Absolute Deviation

This repository is a literature-aligned computational study of ordinary least squares (OLS) and least absolute deviations (LAD) regression. The methodology is intentionally restricted to ideas supported by six core LAD papers that directly match the project title and implemented workflow.

## Project objective

The central research question is:

> How does regression obtained by minimizing absolute deviations differ from regression obtained by minimizing squared deviations, particularly when observations contain large or extreme errors?

The project studies the mathematical difference between squared and absolute loss, the linear-programming formulation of LAD, sensitivity to large vertical errors and long-tailed error distributions, and the computational cost of LAD relative to OLS.

### Actual multivariate visualizations

All three empirical datasets are visualized using the full multivariate regression fits rather than by plotting only one predictor against the response.

#### Hawkins-Bradu-Kass (HBK)

The HBK dataset already provides case-group labels: cases 1-10 are marked as **bad leverage**, cases 11-14 as **good leverage**, and cases 15-75 as **regular**. The figure therefore distinguishes actual regular and non-regular observations without introducing a new outlier-detection method. Both OLS and LAD use `X1`, `X2`, and `X3` simultaneously, and each point is plotted by its absolute OLS residual and absolute LAD residual.

![HBK multivariate inlier/outlier visualization](outputs/figures/hbk_multivariate_inlier_outlier.png)

**Figure:** Actual HBK observations visualized through residuals from the full multivariate OLS and LAD fits. The case labels come directly from the dataset.

#### Boston Housing

Boston Housing does not contain supplied inlier/outlier labels. The figure therefore shows the absolute residual from the full multivariate OLS fit against the absolute residual from the full multivariate LAD fit. The five observations at the largest displayed residual magnitudes are annotated only to make the extreme end of the residual cloud easy to inspect; this is a descriptive visualization choice, not a formal outlier-detection rule.

![Boston Housing multivariate residual visualization](outputs/figures/boston_housing_multivariate_residuals.png)

**Figure:** Boston Housing residual-space visualization using all active predictors simultaneously. Highlighted observations are described only as large-residual observations, not formally detected outliers.

#### Concrete Compressive Strength

Concrete Strength likewise has no supplied case labels identifying inliers and outliers. The full predictor set is used for both OLS and LAD, and the plot compares the resulting absolute residuals observation by observation. As with Boston Housing, the five largest displayed residual observations are annotated for readability only.

![Concrete Strength multivariate residual visualization](outputs/figures/concrete_strength_multivariate_residuals.png)

**Figure:** Concrete Strength residual-space visualization using the complete multivariate regression model. No external outlier detector is used.

The distinction is deliberate: HBK can be shown with its existing regular/non-regular labels, whereas Boston Housing and Concrete Strength can only be shown as residual-space visualizations unless an additional outlier-detection methodology is introduced. To preserve the literature-only constraint, this repository does not add Mahalanobis distance, MCD, PCA, IQR, Cook's distance, or another external classification rule.

## Literature basis

The project now uses only the six papers that are directly relevant to standard regression by minimizing absolute deviation:

1. Wagner, H. M. (1959), *Linear Programming Techniques for Regression Analysis*.
2. Barrodale, I. and Roberts, F. D. K. (1973), *An Improved Algorithm for Discrete l1 Linear Approximation*.
3. Bassett, G. and Koenker, R. (1978), *Asymptotic Theory of Least Absolute Error Regression*.
4. Bloomfield, P. and Steiger, W. (1980), *Least Absolute Deviations Curve-Fitting*.
5. Narula, S. C. and Wellington, J. F. (1982), *The Minimum Sum of Absolute Errors Regression: A State of the Art Survey*.
6. Pollard, D. (1991), *Asymptotics for Least Absolute Deviation Regression Estimators*.

Wagner and Barrodale-Roberts provide the linear-programming and computational foundations. Bassett-Koenker, Bloomfield-Steiger, Narula-Wellington and Pollard provide the statistical, robustness, computational, and asymptotic basis for standard LAD regression.

## OLS and LAD objectives

OLS minimizes:

```text
sum_i (y_i - x_i^T beta)^2
```

LAD minimizes:

```text
sum_i |y_i - x_i^T beta|
```

For LAD, residuals are represented by non-negative positive and negative parts:

```text
y_i - x_i^T beta = e_i^+ - e_i^-
e_i^+ >= 0
e_i^- >= 0
```

and the LP objective is:

```text
minimize sum_i (e_i^+ + e_i^-)
```

The custom solver in `src/absolute_deviation/models.py` implements this literature-supported LP formulation with `scipy.optimize.linprog`.

## Active empirical datasets

Only three empirical datasets are included in the active experiment:

- Boston Housing
- Concrete Compressive Strength
- Hawkins-Bradu-Kass (HBK)

The previously included `synthetic_ols_lad_outliers.csv` dataset has been excluded from the experiment and removed from the active repository data.

The project still contains controlled simulations for the response-contamination and error-distribution studies. These are generated internally to test literature-supported claims and are not treated as empirical datasets.

No MCD, robust Mahalanobis distance, PCA outlier screening, bootstrap inference, Holm correction, repeated cross-validation, or external outlier-classification framework is used.

## Experiments

### 1. Original-data OLS/LAD comparison

Boston Housing, Concrete Strength and HBK are fitted with OLS and LAD using the same rows and predictors. The repository reports coefficients, SSE, SAE, MAE, RMSE and runtime.

### 2. Large-response-error experiment

A controlled regression design is generated internally. Deliberately large response errors are introduced, then OLS and LAD are refitted and compared using coefficient change, prediction change, MAE and RMSE relative to the uncontaminated response. The exact contamination percentages are project-selected illustration settings rather than values claimed from the papers.

### 3. Error-distribution experiment

OLS and LAD are compared under Normal, Laplace/double-exponential, Cauchy/long-tailed and contaminated-normal errors, reflecting distributions discussed in the six core papers. Only descriptive summaries are used.

### 4. Computational comparison

OLS and the LP-based LAD solver are timed over several sample sizes and predictor counts. Timings are implementation- and hardware-specific.

## Reproduce the study

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/generate_all_results.py
```

Run tests with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

Generated CSV files are written to `outputs/results/` and figures to `outputs/figures/`.

## Interpretation

The project does not assume LAD is universally superior. OLS directly optimizes squared error and remains attractive when squared loss, approximately Gaussian light-tailed errors and computational simplicity are central. LAD directly optimizes absolute error and is particularly relevant when large vertical response errors or long-tailed distributions make squared-error fitting unstable.

## Limitations

- The project studies standard linear LAD only.
- LAD is not claimed to be immune to all unusual observations, especially extreme predictor configurations.
- The HBK inlier/outlier visualization uses the dataset's existing case labels; it is not a new outlier-detection algorithm.
- Boston Housing and Concrete Strength do not have supplied case labels, so their residual plots are descriptive and do not formally classify outliers.
- Controlled contamination and distribution simulations depend on their selected design settings.
- Runtime depends on hardware, solver and data dimensions.
- The absolute-value objective is non-differentiable, which motivates the computational approaches emphasized in the literature.

See `docs/literature_mapping.md`, `docs/theoretical_background.md`, `docs/experimental_design.md`, and `docs/literature_compliance_audit.md` for the methodological mapping.
