# Full-Data OLS and LAD Baselines

## Purpose

This stage fits ordinary least squares (OLS) and least absolute deviations
(LAD) to every complete dataset. These are descriptive, in-sample baselines.
They do not yet measure performance on unseen observations.

## Models

For response values `y_i` and fitted values `yhat_i`:

- OLS minimizes the sum of squared errors, `sum((y_i - yhat_i)^2)`;
- LAD minimizes the sum of absolute errors, `sum(|y_i - yhat_i|)`.

Both models include an intercept. The implementation robust-scales predictors
and the response using medians and interquartile ranges for numerical stability,
then returns coefficients and predictions in original measurement units.

## Dataset roles

| Dataset | Response | Predictors |
|---|---|---|
| Boston Housing | `MEDV` | 12 housing and neighborhood variables; `B` is excluded |
| Concrete Strength | `Strength` | 8 mixture and age variables |
| Hawkins-Bradu-Kass | `Y` | `X1`, `X2`, `X3` |
| Synthetic OLS-LAD | `y` | `X1`, `X2`, `X3` |

## Reported quantities

The workflow saves SAE, SSE, MAE, RMSE, median absolute error, and R-squared.
It also saves original-unit coefficients, robust-standardized slope
coefficients, row-level predictions, and residuals.

The objective checks are implementation invariants:

- OLS SSE must not exceed LAD SSE;
- LAD SAE must not exceed OLS SAE.

## Run

```powershell
python scripts/fit_baseline_models.py
```

Generated files:

- `outputs/results/baseline_model_metrics.csv`;
- `outputs/results/baseline_model_coefficients.csv`;
- `outputs/results/baseline_optimality_checks.csv`;
- `outputs/results/*_baseline_predictions.csv`;
- `outputs/figures/*_baseline_ols_lad_fit.png`;
- `outputs/figures/merged_baseline_ols_lad_fits.png`.

Controlled contamination is documented in `docs/robustness_experiments.md`.
Cross-validation and inference are documented in `docs/final_validation.md`.
