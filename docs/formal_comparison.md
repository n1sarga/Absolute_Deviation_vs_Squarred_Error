# Formal OLS-versus-LAD Comparison

## Scope

Stage 5 converts prior experiment outputs into direct paired comparisons. It
measures error, prediction stability, coefficient stability, and fit-only
runtime. Stage 6 adds statistical inference and cross-validation.

## Paired effects

Each OLS result is paired with LAD fitted using identical rows and artificial
noise. For loss or instability measure `M`, reported LAD reduction is

```text
100 * (M_OLS - M_LAD) / M_OLS
```

Positive values favor LAD. Negative values favor OLS. Win rate is fraction of
30 repetitions where LAD has smaller measure than OLS.

Reduction percentages for prediction drift and slope shift are undefined at
0% contamination because both models have zero displacement from themselves.

Measures:

- clean-reference MAE and RMSE: prediction error on unchanged regular rows;
- prediction drift: RMSE between contaminated-fit and clean-fit predictions;
- slope shift: L2 distance between contaminated and clean robust-standardized
  slope vectors;
- runtime: elapsed fit call only, excluding loading, classification, metrics,
  file output, and plotting.

## Runtime protocol

Each dataset and sample condition receives three untimed warm-ups followed by
30 timed fits per model. Tables retain every measurement, distribution summary,
and software/host context. Timings are machine- and load-specific; ratios show
computational cost on this run, not universal constants.

## Current result

At 20% artificial response contamination:

| Dataset | MAE reduction | Prediction-drift reduction | Slope-shift reduction | LAD runtime multiple |
|---|---:|---:|---:|---:|
| Boston Housing | 52.9% | 88.2% | 88.4% | 51x |
| Concrete Strength | 39.5% | 91.5% | 91.2% | 133x |
| Hawkins-Bradu-Kass | 47.7% | 82.6% | 83.4% | 8x |
| Synthetic OLS-LAD | 87.2% | 97.3% | 97.5% | 9x |

LAD had lower clean-reference MAE in all 120 paired 20%-contamination fits.
Runtime multiples use inlier-only samples on environment recorded in
`outputs/results/runtime_environment.csv`.

## Run

Stage 4 outputs must exist first.

```powershell
python scripts/evaluate_models.py
```

## Outputs

- `outputs/results/sample_condition_model_comparison.csv`;
- `outputs/results/contamination_model_comparison.csv`;
- `outputs/results/contamination_model_comparison_summary.csv`;
- `outputs/results/runtime_benchmark.csv`;
- `outputs/results/runtime_summary.csv`;
- `outputs/results/runtime_environment.csv`;
- `outputs/results/stage5_key_findings.csv`;
- `outputs/figures/model_comparison_dashboard.png`;
- `outputs/figures/runtime_benchmark.png`.

## Interpretation boundary

These are descriptive paired effects for fixed datasets and simulated
contamination. They do not establish out-of-sample superiority or uncertainty.
Stage 6 results are reported in `docs/final_validation.md`.
