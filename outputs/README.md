# Generated Outputs

These files are reproducible products of:

```powershell
python scripts/analyze_outliers.py
python scripts/validate_lad.py
python scripts/fit_baseline_models.py
python scripts/run_robustness_experiments.py
```

- `figures/` contains individual and merged diagrams for joint distributions
  and regression-specific outlier classes.
- `results/` contains row-level joint distances, leverage/response classes, and
  combined summary tables. It also contains LP-versus-reference LAD validation
  results plus full-data OLS/LAD metrics, coefficients, predictions, residuals,
  objective checks, and controlled-contamination sensitivity results.
