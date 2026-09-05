# Theoretical Background

## Least squares and least absolute deviations

The six core papers retained for this project treat least squares and least absolute deviations as alternative criteria for fitting a linear regression model. Least squares minimizes the sum of squared residuals, whereas LAD minimizes the sum of absolute residuals. LAD is also described as least absolute error regression, minimum-sum-of-absolute-errors regression, and $L_1$ regression.

For residuals

$$
r_i = y_i - x_i^T\beta,
$$

OLS minimizes

$$
\min_{\beta}\sum_{i=1}^{n} r_i^2,
$$

whereas LAD minimizes

$$
\min_{\beta}\sum_{i=1}^{n}|r_i|.
$$

The distinction matters because the square function grows more rapidly than the absolute-value function. Consequently, a large residual contributes more heavily to the OLS objective than to the LAD objective. Narula and Wellington (1982), Bassett and Koenker (1978), and Bloomfield and Steiger (1980) use this contrast to motivate absolute-error regression when errors are long-tailed or observations contain large errors.

## LAD as an $L_1$ problem

Barrodale and Roberts (1973) formulate discrete $L_1$ approximation as minimization of the sum of absolute approximation errors. The regression form used in this repository is a direct special case of that general $L_1$ approximation problem.

## Linear-programming formulation

Wagner (1959) shows that minimizing absolute vertical deviations in linear regression can be converted to a linear-programming problem. Each residual is represented as

$$
r_i = e_i^+ - e_i^-,
$$

with

$$
e_i^+ \ge 0, \qquad e_i^- \ge 0.
$$

The LAD objective is then

$$
\min \sum_{i=1}^{n}\left(e_i^+ + e_i^-\right),
$$

subject to the regression equality constraints. Narula and Wellington (1982) summarize the same formulation. The repository implements this LP using a modern numerical linear-programming solver.

## Computation and non-differentiability

Unlike the quadratic least-squares objective, the absolute-value objective is not differentiable at zero. The retained computational literature therefore emphasizes linear programming, simplex variants, weighted-median ideas, and specialized $L_1$ algorithms. Barrodale and Roberts (1973) develop an efficient simplex-based algorithm for discrete $L_1$ approximation, while Bloomfield and Steiger (1980) propose a computational procedure for LAD curve fitting and compare its cost with least squares and other LAD algorithms.

The repository does not claim to reproduce the Barrodale-Roberts algorithm exactly. It implements the literature-supported LP formulation and records runtime as an implementation-specific computational comparison.

## Long-tailed and non-Gaussian errors

Bassett and Koenker (1978) describe least absolute error regression as particularly well suited to longer-tailed error distributions. Narula and Wellington (1982) summarize evidence supporting absolute-error regression for Laplace, Cauchy, contaminated-normal, and other thick-tailed situations. Bloomfield and Steiger (1980) discuss LAD computation under long-tailed and normal settings.

These points motivate the repository's simulation study with Normal, Laplace, Cauchy, and contaminated-normal errors. No external inferential method is added to the experiment.

## Asymptotic properties

Bassett and Koenker (1978) establish asymptotic results for least absolute error regression under stated regularity conditions for the general linear model. Pollard (1991) gives a direct asymptotic argument based on convexity of the LAD criterion and further develops the large-sample theory of LAD regression estimators.

## Scope

The literature review is limited to Wagner (1959), Barrodale and Roberts (1973), Bassett and Koenker (1978), Bloomfield and Steiger (1980), Narula and Wellington (1982), and Pollard (1991).
