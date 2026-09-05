# Regression by Minimizing Absolute Deviation

This project compares Ordinary Least Squares (OLS) and Least Absolute Deviations (LAD) regression using Boston Housing, Concrete Compressive Strength, and Hawkins-Bradu-Kass (HBK) datasets.

The complete executed workflow is available in [`notebooks/absolute_deviation_full_workflow.ipynb`](notebooks/absolute_deviation_full_workflow.ipynb).

## Objective

> How does regression obtained by minimizing absolute deviations differ from regression obtained by minimizing squared deviations, particularly when observations contain large or extreme errors?

For residuals

$$
r_i = y_i - x_i^T\beta,
$$

OLS minimizes

$$
\min_{\beta}\sum_{i=1}^{n} r_i^2,
$$

while LAD minimizes

$$
\min_{\beta}\sum_{i=1}^{n} |r_i|.
$$

For the linear-programming formulation of LAD,

$$
r_i = e_i^+ - e_i^-, \qquad e_i^+ \ge 0, \quad e_i^- \ge 0,
$$

with objective

$$
\min \sum_{i=1}^{n}\left(e_i^+ + e_i^-\right).
$$

## Workflow

1. Load and prepare the three empirical datasets.
2. Fit OLS and LAD models using the same observations and predictors.
3. Compare SSE, SAE, MAE, RMSE, coefficients, and runtime.
4. Visualize fitted values and multivariate residual behavior.
5. Run the controlled large-response-error experiment.
6. Compare OLS and LAD under Normal, Laplace, Cauchy, and contaminated-normal errors.
7. Benchmark runtime across sample sizes and predictor counts.
8. Validate that OLS minimizes squared error and LAD minimizes absolute error on deterministic examples.
9. Generate all result tables, figures, and the executed notebook.

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
