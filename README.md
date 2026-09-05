# Regression by Minimizing Absolute Deviation

This project compares Ordinary Least Squares (OLS) and Least Absolute Deviations (LAD) regression using Boston Housing, Concrete Compressive Strength, and Hawkins-Bradu-Kass (HBK) datasets.

The complete executed workflow is available in [`notebooks/absolute_deviation_full_workflow.ipynb`](notebooks/absolute_deviation_full_workflow.ipynb).

## Experimental workflow

1. [Load and prepare the three empirical datasets.](src/absolute_deviation/data.py)
2. [Fit OLS and LAD models using the same observations and predictors.](src/absolute_deviation/models.py)
3. [Compare SSE, SAE, MAE, RMSE, coefficients, and runtime.](src/absolute_deviation/experiments.py)
4. [Visualize fitted values and multivariate residual behavior.](src/absolute_deviation/plotting.py)
5. [Run the controlled large-response-error experiment.](src/absolute_deviation/experiments.py)
6. [Compare OLS and LAD under Normal, Laplace, Cauchy, and contaminated-normal errors.](src/absolute_deviation/experiments.py)
7. [Benchmark runtime across sample sizes and predictor counts.](src/absolute_deviation/experiments.py)
8. [Validate the OLS and LAD implementations.](src/absolute_deviation/experiments.py)
9. [Generate all result tables and figures.](scripts/generate_all_results.py)

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