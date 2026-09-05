# Regression by Minimizing Absolute Deviation

This project compares Ordinary Least Squares (OLS) and Least Absolute Deviations (LAD) regression using Boston Housing, Concrete Compressive Strength, and Hawkins-Bradu-Kass (HBK) datasets.

The complete executed workflow is available in [`absolute_deviation_full_workflow.ipynb`](notebooks/absolute_deviation_full_workflow.ipynb).

## Experimental workflow

1. Load and prepare the three empirical datasets. [`data.py`](src/absolute_deviation/data.py)
2. Fit OLS and LAD models using the same observations and predictors. [`models.py`](src/absolute_deviation/models.py)
3. Compare SSE and SAE on the empirical datasets. [`experiments.py`](src/absolute_deviation/experiments.py)
4. Run the controlled large-response-error experiment. [`experiments.py`](src/absolute_deviation/experiments.py)
5. Compare OLS and LAD under Normal, Laplace, Cauchy, and contaminated-normal errors. [`experiments.py`](src/absolute_deviation/experiments.py)
6. Benchmark runtime across sample sizes and predictor counts. [`experiments.py`](src/absolute_deviation/experiments.py)
7. Validate the OLS and LAD implementations. [`experiments.py`](src/absolute_deviation/experiments.py)
8. Generate the four focused visualizations used for the main comparison. [`plotting.py`](src/absolute_deviation/plotting.py)
9. Save result tables and execute the complete notebook.

## Key visualizations

### OLS vs LAD with large response errors

![OLS and LAD with contaminated responses](outputs/figures/ols_lad_contaminated_fit.png)

### SSE vs contamination

![SSE versus contamination](outputs/figures/sse_vs_contamination.png)

### SAE vs contamination

![SAE versus contamination](outputs/figures/sae_vs_contamination.png)

### Hawkins-Bradu-Kass (HBK) multivariate residual comparison

![HBK multivariate residual comparison](outputs/figures/hbk_multivariate_inlier_outlier.png)

## Run

```powershell
python -m pip install -r requirements-dev.txt
python scripts/generate_all_results.py
python -m pytest
```

Generated result tables are stored in `outputs/results/`. The four retained figures are stored in `outputs/figures/`.
