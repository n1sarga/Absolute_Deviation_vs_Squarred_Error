# Observed-Condition and Contamination Experiments

## Research question

How differently do OLS and LAD react when regression outliers are removed or
when increasingly severe response contamination is introduced?

This stage is a sensitivity experiment. It is not yet cross-validation or a
claim about performance on unseen data.

## Observed sample conditions

The outlier classifier from Stage 1 defines a row as **regular** only when it is
neither a predictor-leverage point nor a conditional response outlier. OLS and
LAD are fitted under two conditions:

1. all observed rows;
2. regular rows only.

Each fit is evaluated on both the complete observed sample and the regular-row
sample. The regular-row evaluation gives both training conditions a common
comparison set.

## Controlled response contamination

The regular rows form a fixed clean reference. For every dataset:

1. contaminate 0%, 5%, 10%, or 20% of response values;
2. sample affected rows without replacement;
3. add independent zero-mean Gaussian noise with standard deviation equal to
   10 times the clean response IQR;
4. give OLS and LAD the identical contaminated sample;
5. repeat each rate 30 times using the deterministic seed schedule in the code;
6. evaluate fitted predictions against the unchanged regular responses.

This is an inflated-variance contaminated-error experiment motivated by the
contaminated-normal robustness setting discussed in the project's LAD
literature. It is not intended to reproduce one paper's exact simulation.

The primary plotted quantity is clean-reference MAE. The raw tables also retain
clean-reference RMSE, prediction drift from the uncontaminated fit, standardized
slope displacement, contaminated-training error, coefficients, selected row
identifiers, and added noise.

## Interpretation

- The point at 0% is the inlier-only fit.
- Solid lines show mean clean-reference MAE under artificial contamination.
- Error bars show one standard deviation over 30 repetitions.
- Dotted horizontal lines show models fitted to the complete observed dataset,
  evaluated on the same regular rows.
- Lower and flatter curves indicate less sensitivity under this protocol.

Outlier labels were calculated once from the complete observed dataset. That is
appropriate for this descriptive sensitivity stage but would leak information
in predictive cross-validation. Later validation must refit diagnostics inside
each training fold.

## Current result

At 20% artificial response contamination, mean clean-reference MAE was:

| Dataset | OLS | LAD |
|---|---:|---:|
| Boston Housing | 4.966 | 2.340 |
| Concrete Strength | 9.744 | 5.895 |
| Hawkins-Bradu-Kass | 0.915 | 0.479 |
| Synthetic OLS-LAD | 14.913 | 1.915 |

Under this deliberately severe vertical-contamination protocol, LAD remained
close to its uncontaminated error while OLS degraded in every dataset.

## Run

```powershell
python scripts/run_robustness_experiments.py
```

Implementation is separated into condition fitting, contamination mechanics,
plotting, and orchestration modules under `src/outlier_analysis/`.

## Outputs

- `outputs/results/sample_condition_metrics.csv`;
- `outputs/results/sample_condition_coefficients.csv`;
- `outputs/results/contamination_experiment_metrics.csv`;
- `outputs/results/contamination_experiment_coefficients.csv`;
- `outputs/results/contamination_experiment_design.csv`;
- `outputs/results/contamination_experiment_summary.csv`;
- `outputs/figures/*_condition_contamination_comparison.png`;
- `outputs/figures/merged_condition_contamination_comparison.png`.
