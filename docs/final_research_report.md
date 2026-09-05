# Regression by Minimizing Absolute Deviation

## 1. Introduction

Least squares and least absolute deviations are two long-established criteria for fitting linear regression models. Least squares minimizes squared residuals, while least absolute deviations (LAD) minimizes absolute residuals. The supplied literature motivates LAD as an alternative when extreme errors or long-tailed error distributions make squared-error fitting sensitive to a small number of large residuals.

This project is deliberately restricted to the ideas and methods contained in the supplied LAD literature. The repository therefore focuses on the OLS/LAD objective functions, the linear-programming formulation of LAD, direct response-error robustness experiments, error distributions discussed by the papers, and computational comparison. Modern outlier-screening and validation techniques that are not part of the supplied methodological basis are excluded.

## 2. Research objective

The primary research question is:

> How does regression obtained by minimizing absolute deviations differ from regression obtained by minimizing squared deviations, particularly when observations contain large or extreme errors?

The secondary objectives are to explain the LP formulation of LAD, compare OLS and LAD on the same empirical datasets, examine changes under large vertical response errors, study Normal versus long-tailed error distributions, and compare computational cost.

## 3. Literature basis

Wagner (1959) establishes the linear-programming treatment of absolute-deviation regression. Barrodale and Roberts (1973) develop an efficient algorithm for discrete L1 linear approximation and emphasize specialized simplex computation. Bassett and Koenker (1978) give asymptotic theory for least absolute error regression and discuss its relevance for longer-tailed error distributions. Bloomfield and Steiger (1980) study LAD curve fitting, computation, and reduced sensitivity to extreme errors. Narula and Wellington (1982) survey minimum-sum-of-absolute-errors regression, including robustness, algorithms, and non-Gaussian settings. Pollard (1991) develops further asymptotic theory using convexity arguments. Amemiya (1982) extends LAD ideas to simultaneous-equation models, and Powell (1984) extends them to censored regression. The latter two are treated as literature extensions rather than implemented models in this standard linear-regression project.

## 4. Mathematical formulation

For observations `(x_i, y_i)`, OLS estimates `beta` by minimizing

```text
sum_i (y_i - x_i^T beta)^2.
```

LAD estimates `beta` by minimizing

```text
sum_i |y_i - x_i^T beta|.
```

The square grows faster than the absolute value, so a large residual contributes more strongly to the OLS objective than to the LAD objective. This is the central mathematical reason the supplied literature discusses LAD as less sensitive to extreme errors.

## 5. Linear-programming formulation of LAD

Following the literature-supported LP formulation, each residual is written as

```text
y_i - x_i^T beta = e_i^+ - e_i^-
```

with

```text
e_i^+ >= 0
e_i^- >= 0.
```

The LAD problem then becomes

```text
minimize sum_i (e_i^+ + e_i^-).
```

The repository implements this formulation directly with a modern linear-programming solver. This preserves the mathematical program described in the literature without claiming to reproduce a specific historical simplex implementation.

## 6. Empirical datasets

The active empirical datasets are:

- Boston Housing
- Concrete Compressive Strength
- Hawkins-Bradu-Kass (HBK)

The previously included synthetic multivariate OLS/LAD CSV dataset has been removed from the active experiment. The empirical comparison therefore uses only these three real datasets.

Data preparation is intentionally minimal. Identifier or non-numeric descriptive fields are excluded where needed, missing rows are removed transparently, and the same usable observations are passed to OLS and LAD. Extreme observations are not automatically removed because the project is specifically interested in how the two loss criteria respond to large errors.

Controlled data generated inside the contamination and error-distribution studies are used only as simulations to operationalize literature-supported claims. They are not treated as empirical datasets.

## 7. Experimental method

### 7.1 Original-data comparison

OLS and LAD are fitted to the same rows and predictors for Boston Housing, Concrete Strength, and HBK. Coefficients, SSE, SAE, MAE, RMSE, and runtime are reported. The comparison is descriptive and tied directly to the two objective functions: OLS is expected to minimize SSE, while LAD is expected to minimize SAE.

### 7.2 Large response-error experiment

A controlled regression design is generated internally and fitted in an uncontaminated state and after deliberately adding large errors to selected response values. The project records coefficient change, prediction change, MAE and RMSE relative to the uncontaminated response, and runtime. The exact contamination levels and magnitude are project-selected illustration settings and are not claimed to be prescribed by the papers.

### 7.3 Error-distribution experiment

The controlled simulation compares Normal, Laplace, Cauchy, and contaminated-normal errors because these settings are directly motivated by the supplied literature's discussion of Gaussian, double-exponential, long-tailed, and contaminated distributions. OLS and LAD are compared using coefficient estimation error and both squared- and absolute-error measures.

### 7.4 Computational comparison

OLS and LP-based LAD are timed for several sample sizes and numbers of predictors. The purpose is to illustrate the computational trade-off discussed in Wagner, Barrodale-Roberts, Bloomfield-Steiger, and Narula-Wellington. Runtime is treated as solver-, hardware-, and data-dependent.

## 8. Results

All numerical results are generated reproducibly by:

```text
python scripts/generate_all_results.py
```

The script creates empirical-data metrics and coefficients for the three active datasets, contamination results, error-distribution results, runtime results, solver validation results, and accompanying figures.

A defining correctness check is included: for deterministic examples, the OLS solution must have SSE no greater than the SSE evaluated at the LAD coefficients, and the LAD solution must have SAE no greater than the SAE evaluated at the OLS coefficients. These checks follow directly from the two optimization criteria.

## 9. Discussion

The literature-supported expectation is not that LAD always dominates OLS. Rather, each estimator directly optimizes a different loss function. OLS is naturally favored when squared error and approximately Gaussian light-tailed errors are central. LAD is naturally favored when absolute error, median behavior, large vertical response errors, or long-tailed distributions are more important.

The response-error experiment illustrates the robustness argument made throughout the supplied literature: when a small number of response values are made very large, the squared-loss fit can move substantially because the largest residuals receive very high weight, while the absolute-loss fit is generally less affected. The error-distribution experiment complements this by examining Normal, Laplace, Cauchy, and contaminated-normal settings without adding external inferential machinery.

The computational comparison also reflects a recurring literature theme. Least squares has a particularly simple numerical solution, whereas LAD historically motivated linear-programming and specialized L1 algorithms. Runtime is therefore treated as a computational trade-off rather than evidence that one loss function is universally preferable.

## 10. Limitations

The project is restricted to standard linear LAD. Amemiya's two-stage LAD and Powell's censored LAD are not implemented. LAD should not be described as immune to every kind of unusual observation; extreme predictor configurations can still be problematic. The controlled contamination experiment is illustrative and its exact percentages and magnitudes are project settings. Error-distribution simulation results depend on sample size and design. The absolute-value objective is non-differentiable, and the modern LP solver used here is not the same as the historical algorithms studied in the papers. Runtime is machine-specific.

## 11. Conclusion

Regression by minimizing absolute deviation provides a principled alternative to least squares when the research problem is better represented by absolute loss or when large vertical errors and long-tailed distributions make squared-error fitting unstable. The supplied literature supports this interpretation through statistical theory, robust-error motivation, linear-programming formulations, and computational algorithms. Least squares remains important when squared-error performance, computational simplicity, and approximately Gaussian light-tailed errors dominate. The appropriate choice therefore follows the loss function and error behavior rather than a claim that one method is universally superior.

## References

- Amemiya, T. (1982). *Two Stage Least Absolute Deviations Estimators*.
- Barrodale, I. and Roberts, F. D. K. (1973). *An Improved Algorithm for Discrete l1 Linear Approximation*.
- Bassett, G. and Koenker, R. (1978). *Asymptotic Theory of Least Absolute Error Regression*.
- Bloomfield, P. and Steiger, W. (1980). *Least Absolute Deviations Curve-Fitting*.
- Narula, S. C. and Wellington, J. F. (1982). *The Minimum Sum of Absolute Errors Regression: A State of the Art Survey*.
- Pollard, D. (1991). *Asymptotics for Least Absolute Deviation Regression Estimators*.
- Powell, J. L. (1984). *Least Absolute Deviations Estimation for the Censored Regression Model*.
- Wagner, H. M. (1959). *Linear Programming Techniques for Regression Analysis*.
