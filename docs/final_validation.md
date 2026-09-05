# Final Validation and Statistical Inference

## Predictive protocol

Primary validation uses repeated 5-fold cross-validation with 10 independent
shuffles and fixed random seed 42. Within each split:

1. fit OLS and LAD only on training rows;
2. estimate every scaling value only from training rows;
3. predict untouched test rows;
4. use identical partitions for both models;
5. keep all rows—no full-sample outlier labels influence fitting or testing.

Every observation receives one out-of-fold prediction per repeat. Repeat-level
MAE, RMSE, median absolute error, and R-squared provide paired descriptive
comparisons.

## Inference

For each observation, out-of-fold predictions are averaged over 10 repeats.
Rows are then sampled with replacement in 5,000 paired bootstrap replicates.
This yields percentile 95% confidence intervals for LAD-minus-OLS differences
in MAE, RMSE, and median absolute error. Negative differences favor LAD.

Primary MAE tests use 20,000 two-sided paired randomizations. Each randomization
swaps OLS/LAD labels within observations by sign-flipping absolute-error
differences, directly testing zero mean difference. Holm adjustment controls
family-wise error across four datasets at alpha 0.05.

## Current result

LAD produced lower out-of-fold MAE point estimates for all four datasets.
Bootstrap intervals excluded zero only for Boston Housing. After Holm correction,
no paired randomization test crossed family-wise alpha 0.05; smallest adjusted
p-value was 0.056 for Boston. Result favors LAD descriptively but remains
insufficient for universal out-of-sample superiority claim.

## Interpretation limits

- Row bootstrap assumes observations are independent and representative.
- Repeated folds overlap; repeat metrics are not treated as independent tests.
- Averaging predictions across repeats reduces partition noise but is not an
  external validation set.
- Bootstrap conditions on fixed out-of-fold predictions rather than refitting
  full training pipeline in every bootstrap replicate.
- Outlier screening remains descriptive and is not applied inside primary CV.
- Results cover linear models and selected datasets only.

## Run

Step 5 results must exist first.

```powershell
python scripts/run_final_analysis.py
```

## Outputs

- `outputs/results/cross_validation_predictions.csv`;
- `outputs/results/cross_validation_average_predictions.csv`;
- `outputs/results/cross_validation_repeat_metrics.csv`;
- `outputs/results/cross_validation_summary.csv`;
- `outputs/results/cross_validation_paired_comparison.csv`;
- `outputs/results/bootstrap_inference.csv`;
- `outputs/results/final_research_findings.csv`;
- `outputs/figures/cross_validation_comparison.png`;
- `outputs/figures/bootstrap_mae_difference.png`.

See `docs/final_research_report.md` for complete synthesis and conclusion.
