# Outlier Analysis

Unified diagnostics for Boston Housing, Concrete Strength,
Hawkins-Bradu-Kass (HBK), and supplied synthetic data.

## 1. Joint data-distribution screening

This exploratory view finds rows unusual in the combined predictor-response
space:

1. Exclude identifiers and text labels.
2. Robust-scale all selected predictors and the response.
3. Fit Minimum Covariance Determinant (MCD) with 70% support.
4. Calculate robust Mahalanobis distance in full-dimensional space.
5. Flag distances beyond the 97.5% chi-square cutoff.
6. Use PCA only to display the robust-scaled data in two dimensions.

These flags are not used to distinguish regression-outlier mechanisms.

## 2. Regression-specific classification

Each dataset declares predictor matrix `X` and response `y` separately.

### Predictor leverage

Predictors are robust-scaled. MCD is fitted in predictor space only. Observation
`i` is a leverage point when

```text
sqrt((x_i - robust_center)' robust_covariance^-1
     (x_i - robust_center)) > sqrt(chi2_0.975,p)
```

where `p` is predictor count.

### Conditional response outlier

An unpenalized median regression (`quantile=0.5`) provides a robust diagnostic
fit. Its residuals are centered at their median and scaled using
`1.4826 * MAD`. Observation `i` is a conditional response outlier when its
absolute robust residual z-score exceeds `3.5`.

Median regression here supports classification only. It is not the later
OLS-versus-LAD performance experiment.

### Mutually exclusive classes

| Leverage flag | Response flag | Class |
|---|---|---|
| No | No | Regular |
| Yes | No | Leverage only |
| No | Yes | Response only |
| Yes | Yes | Leverage + response |

Blue, yellow, red, and purple represent these four classes respectively.

## Interpretation limits

- A flag means unusual, not necessarily erroneous.
- MCD assumes an approximately ellipsoidal central predictor distribution.
  Skewed or multi-cluster data can produce many leverage flags.
- Classification describes each complete dataset. Final predictive validation
  avoids leakage by not using these labels. Any future filtered CV experiment
  must refit every detector within each training fold.
- Boston variable `B` is excluded because it encodes an ethically problematic
  racial assumption.
- Synthetic `ID` and HBK `Observation` are identifiers, not predictors.

## Run

```powershell
python scripts/analyze_outliers.py
```

## Code layout

```text
scripts/analyze_outliers.py
src/outlier_analysis/
|-- config.py                    # Dataset roles and thresholds
|-- detection.py                 # Joint distribution screening
|-- plotting.py                  # Joint distribution figures
|-- classification.py            # Leverage/response classification
|-- classification_plotting.py   # Classification figures
`-- pipeline.py                  # Execution and saved outputs
```

## Outputs

- `outputs/figures/*_outlier_distribution.png`;
- `outputs/figures/merged_outlier_distributions.png`;
- `outputs/figures/*_outlier_classification.png`;
- `outputs/figures/merged_outlier_classifications.png`;
- `outputs/results/*_outliers.csv`;
- `outputs/results/outlier_summary.csv`;
- `outputs/results/*_outlier_classification.csv`;
- `outputs/results/outlier_classification_summary.csv`.
