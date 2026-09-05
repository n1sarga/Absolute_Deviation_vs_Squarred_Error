# Theoretical Background

## Least squares and least absolute deviations

The supplied literature treats least squares and least absolute deviations as alternative criteria for fitting a linear regression model. Least squares minimizes the sum of squared residuals, whereas LAD minimizes the sum of absolute residuals. The latter is also described as least absolute error regression, minimum-sum-of-absolute-errors regression, L1 regression, and related terminology in the literature.

For a linear model with response vector `y`, predictor matrix `X`, and coefficient vector `beta`, the two criteria are:

```text
OLS:  minimize sum_i (y_i - x_i^T beta)^2
LAD:  minimize sum_i |y_i - x_i^T beta|
```

The distinction matters because the square function grows more rapidly than the absolute-value function. Consequently, a large residual contributes much more heavily to the OLS objective than to the LAD objective. Narula and Wellington (1982), Bassett and Koenker (1978), and Bloomfield and Steiger (1980) use this contrast to motivate absolute-error regression when errors are long-tailed or observations contain large errors.

## LAD as an L1 problem

Barrodale and Roberts (1973) formulate discrete L1 approximation as minimization of the sum of absolute approximation errors. The regression form used in this repository is a direct special case of that general L1 approximation problem.

## Linear-programming formulation

Wagner (1959) shows that minimizing absolute vertical deviations in linear regression can be converted to a linear-programming problem. Let

```text
r_i = y_i - x_i^T beta.
```

Write each residual as the difference of two non-negative variables:

```text
r_i = e_i^+ - e_i^-
e_i^+ >= 0
e_i^- >= 0.
```

Then LAD is solved by minimizing

```text
sum_i (e_i^+ + e_i^-)
```

subject to the regression equality constraints. Narula and Wellington (1982) summarize the same formulation. The repository implements this LP directly using a modern numerical linear-programming solver.

## Computation and non-differentiability

Unlike the quadratic least-squares objective, the absolute-value objective is not differentiable at zero. The supplied computational literature therefore emphasizes linear programming, simplex variants, weighted-median ideas, and specialized L1 algorithms. Barrodale and Roberts (1973) develop an efficient simplex-based algorithm for discrete L1 approximation, while Bloomfield and Steiger (1980) propose a computational procedure for LAD curve fitting and compare its cost with least squares and other LAD algorithms.

The repository does not claim to reproduce the Barrodale-Roberts algorithm exactly. It implements the literature-supported LP formulation and records runtime as an implementation-specific computational comparison.

## Long-tailed and non-Gaussian errors

Bassett and Koenker (1978) describe least absolute error regression as particularly well suited to longer-tailed error distributions. Narula and Wellington (1982) summarize evidence supporting absolute-error regression for Laplace, Cauchy, contaminated-normal, and other thick-tailed situations. Bloomfield and Steiger (1980) note the maximum-likelihood connection under double-exponential (Laplace) errors.

These points motivate the repository's simulation study with Normal, Laplace, Cauchy, and contaminated-normal errors. No external inferential method is added to the experiment.

## Asymptotic properties

Bassett and Koenker (1978) establish consistency and asymptotic Gaussian behavior for least absolute error regression under stated regularity conditions for the general linear model. Pollard (1991) gives a direct asymptotic argument based on convexity of the LAD criterion and extends the discussion to additional settings, including autoregressive examples.

These results are treated as theoretical background. The repository does not use them as a reason to introduce bootstrap, cross-validation, multiplicity correction, or other methods not found in the supplied literature.

## Extensions in the supplied papers

Amemiya (1982) develops two-stage least absolute deviations estimators for simultaneous-equation models. Powell (1984) extends LAD ideas to censored regression and studies consistency and asymptotic normality in that setting. These papers demonstrate broader applicability of LAD, but the current project remains focused on standard linear regression and therefore does not implement 2SLAD or censored LAD.
