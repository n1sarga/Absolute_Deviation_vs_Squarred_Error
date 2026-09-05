# Regression by Minimizing Absolute Deviation: Final Research Summary

## Research objective

Compare ordinary least squares (OLS), which minimizes squared residuals, with
least absolute deviations (LAD), which minimizes absolute residuals. Main focus:
fit quality, response-outlier resistance, coefficient and prediction stability,
runtime, and out-of-sample absolute error.

## Data and model

Four linear-regression datasets were analyzed: Boston Housing, Concrete
Strength from Yeh (1998), Hawkins-Bradu-Kass (HBK), and supplied synthetic
OLS-LAD data. Identifier fields were excluded. Boston variable `B` was excluded
because it encodes an ethically problematic racial assumption.

Custom LAD solves sparse linear program derived from Wagner's formulation.
Independent validation against unpenalized median regression gave matching
absolute-error objectives within floating-point precision. Both OLS and LAD
use median/IQR scaling for numerical conditioning and return parameters in
original units.

## Experimental sequence

1. Robust multivariate screening and regression-specific classification.
2. LP-based LAD implementation and reference validation.
3. Full-data OLS/LAD descriptive baselines.
4. Full-data, regular-row, and repeated artificial-contamination fits.
5. Paired error, prediction drift, slope shift, and runtime comparisons.
6. Repeated cross-validation, paired bootstrap intervals, and multiplicity-
   adjusted randomization tests.

## Controlled-contamination finding

At 20% response contamination, LAD reduced clean-reference MAE relative to OLS
by 39.5% to 87.2%. Prediction drift fell 82.6% to 97.3%; standardized slope
shift fell 83.4% to 97.5%. LAD won clean MAE in all 120 paired repetitions.

This result supports LAD resistance to severe vertical response outliers under
the chosen contaminated-normal protocol. It does not imply equal protection
from high-leverage predictor contamination.

## Out-of-fold validation

Reported values below use each observation's out-of-fold predictions averaged
across 10 repetitions of 5-fold cross-validation.

| Dataset | OLS MAE | LAD MAE | LAD reduction | LAD − OLS MAE | 95% bootstrap CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Boston Housing | 3.450 | 3.295 | 4.49% | -0.155 | [-0.278, -0.033] | 0.056 |
| Concrete Strength | 8.306 | 8.223 | 1.00% | -0.083 | [-0.259, 0.098] | 0.741 |
| Hawkins-Bradu-Kass | 1.519 | 1.308 | 13.91% | -0.211 | [-0.485, 0.083] | 0.454 |
| Synthetic OLS-LAD | 3.347 | 3.309 | 1.13% | -0.038 | [-0.418, 0.395] | 0.860 |

LAD point estimates have lower MAE in every dataset. Boston's ordinary 95%
paired-bootstrap interval excludes zero, but no dataset passes Holm-adjusted
family-wise alpha 0.05; Boston is borderline at 0.056. Evidence therefore does
not establish general out-of-sample MAE superiority across all four datasets.

OLS has lower out-of-fold RMSE in every dataset. Concrete's RMSE difference is
clearly separated from zero; other RMSE intervals include zero. This agrees
with each method's loss: LAD targets typical absolute error, while OLS places
more weight on large errors.

## Computational trade-off

Custom LP-based LAD took roughly 8 to 133 times OLS median fit time on inlier
samples in recorded environment. Absolute fits remained below about 100 ms for
these dataset sizes, but runtime ratios are hardware-, solver-, and load-specific.

## Conclusion

LAD is preferred when response outliers, heavy-tailed errors, median behavior,
or absolute-error loss dominate research goals. It showed much stronger
parameter and prediction stability under severe response contamination.

OLS remains preferred when squared-error performance, computational speed, or
approximately Gaussian light-tailed errors dominate. Current cross-validation
does not justify claiming universal LAD superiority. Best conclusion: LAD gives
robustness benefit with computational and RMSE trade-offs; choice must follow
loss function and contamination risk.

## Limitations

- No independent external test datasets.
- Linear specification only; nonlinear structure remains untreated.
- Bootstrap resamples fixed out-of-fold predictions and conditions on fitted
  CV models; it does not refit entire pipeline within every bootstrap sample.
- Row bootstrap and randomization assume observations form exchangeable pairs.
- HBK and synthetic datasets are small; interval uncertainty is wide.
- Boston target is capped and dataset carries known ethical/historical limits.
- LAD resists vertical response outliers but can remain vulnerable to leverage.
- Artificial noise scale is deliberately severe and represents one scenario.
- Runtime results are machine-specific.

## Reproduction

```powershell
python scripts/analyze_outliers.py
python scripts/validate_lad.py
python scripts/fit_baseline_models.py
python scripts/run_robustness_experiments.py
python scripts/evaluate_models.py
python scripts/run_final_analysis.py
python -m pytest
```

## Literature basis

- Wagner (1959), *Linear Programming Techniques for Regression Analysis*.
- Barrodale and Roberts (1973), *An Improved Algorithm for Discrete L1 Linear
  Approximation*.
- Bassett and Koenker (1978), *Asymptotic Theory of Least Absolute Error
  Regression*.
- Narula and Wellington (1982), *The Minimum Sum of Absolute Errors Regression:
  A State of the Art Survey*.
- Yeh (1998), Concrete Strength dataset publication.
