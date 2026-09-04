# LP-Based Least-Absolute-Deviation Solver

## Objective

For response vector `y` and design matrix `X`, LAD estimates coefficients by

```text
minimize    sum(u_i)
subject to  X beta - u <= y
           -X beta - u <= -y
            u >= 0
```

At the optimum, each auxiliary variable `u_i` equals the absolute residual
`|y_i - x_i' beta|`. The intercept and regression coefficients are unrestricted.
This is the linear-programming construction introduced for regression by
Wagner (1959).

## Numerical implementation

`src/outlier_analysis/lad.py`:

1. validates finite `X` and `y` arrays;
2. robust-scales predictors and response using medians and interquartile ranges;
3. constructs sparse linear constraints;
4. solves with SciPy's HiGHS linear-programming backend;
5. transforms coefficients and predictions back to original units;
6. reports the sum of absolute errors and solver diagnostics.

Scaling changes numerical conditioning, not the unpenalized LAD optimum.

## Independent interface validation

The custom solver is compared with scikit-learn `QuantileRegressor` using
`quantile=0.5`, `alpha=0`, and the same scaled data. LAD solutions can be
nonunique, so the primary validation criterion is equality of minimized
absolute-error objectives. Parameter and prediction differences are reported as
secondary diagnostics.

Run:

```powershell
python scripts/validate_lad.py
```

Outputs:

- `outputs/results/lad_solver_validation.csv`;
- `outputs/results/*_lad_validation_predictions.csv`;
- `outputs/figures/lad_solver_validation.png`.

## Current result

All four datasets match the reference objective within floating-point error.
Largest relative objective difference is below `3e-16`.

## Historical basis

- Wagner, H. M. (1959), *Linear Programming Techniques for Regression
  Analysis*.
- Barrodale, I. and Roberts, F. D. K. (1973), *An Improved Algorithm for
  Discrete L1 Linear Approximation*.
- Narula, S. C. and Wellington, J. F. (1982), *The Minimum Sum of Absolute
  Errors Regression: A State of the Art Survey*.
