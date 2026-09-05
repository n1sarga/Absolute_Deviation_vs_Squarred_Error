# Regression by Minimizing Absolute Deviation

This project compares Ordinary Least Squares (OLS) and Least Absolute Deviations (LAD) regression using Boston Housing, Concrete Compressive Strength, and Hawkins-Bradu-Kass (HBK) datasets.

The complete executed workflow is available in [`notebooks/absolute_deviation_full_workflow.ipynb`](notebooks/absolute_deviation_full_workflow.ipynb).

## Experimental workflow

### Task 4 — Prepare the datasets

The processed datasets are loaded from `data/processed/`. Rows with missing values are removed and only numeric predictors are used. Boston Housing uses `MEDV` as the target, skips the first metadata row, and excludes `B` from the predictors. Concrete Strength uses `Strength` as the target and all eight numeric predictors. HBK uses `Y` as the target and `X1`, `X2`, and `X3` as predictors; `Observation` and `CaseGroup` are not used as regression inputs. The same prepared observations and predictors are then supplied to both OLS and LAD.

### Task 5 — Implement OLS

OLS is fitted with an intercept using NumPy's `lstsq` solver. After estimating the coefficients, the implementation calculates fitted values, residuals, the sum of squared errors (SSE), and fitting time.

### Task 6 — Implement LAD

LAD is fitted with the same design matrix and intercept as OLS. The regression is solved as a linear-programming problem with SciPy's `linprog` using the HiGHS solver. The coefficient variables are unrestricted, while positive and negative residual components are constrained to be non-negative. The implementation returns the coefficients, fitted values, residuals, sum of absolute errors (SAE), and fitting time.

### Remaining workflow

7. Fit OLS and LAD on the three empirical datasets using the same observations and predictors.
8. Compare SSE, SAE, MAE, RMSE, coefficients, and runtime.
9. Visualize fitted values and multivariate residual behavior.
10. Run the controlled large-response-error experiment.
11. Compare OLS and LAD under Normal, Laplace, Cauchy, and contaminated-normal errors.
12. Benchmark runtime across sample sizes and predictor counts.
13. Validate the OLS and LAD implementations.
14. Generate all result tables, figures, and the executed notebook.

## Empirical visualizations

### Hawkins-Bradu-Kass (HBK)

![HBK multivariate inlier/outlier visualization](outputs/figures/hbk_multivariate_inlier_outlier.png)

### Boston Housing

![Boston Housing multivariate residual visualization](outputs/figures/boston_housing_multivariate_residuals.png)

### Concrete Compressive Strength

![Concrete Strength multivariate residual visualization](outputs/figures/concrete_strength_multivariate_residuals.png)

## Run

```powershell
python -m pip install -r requirements-dev.txt
python scripts/generate_all_results.py
python -m pytest
```

Generated results are stored in `outputs/results/` and figures in `outputs/figures/`.