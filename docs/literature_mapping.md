# Literature Mapping

This document maps the active methodology to the six core papers used by the project.

| Repository component | Supporting literature | Role in this project |
|---|---|---|
| OLS versus absolute-error regression | Narula & Wellington (1982); Bassett & Koenker (1978) | Establishes the contrast between squared and absolute loss. |
| LAD objective $\sum_i |r_i|$ | Bassett & Koenker (1978); Pollard (1991) | Defines the least-absolute-deviation estimator. |
| $L_1$ / absolute-error terminology | Bassett & Koenker (1978); Narula & Wellington (1982) | Connects LAD, LAE, $L_1$, and minimum-sum-of-absolute-errors terminology. |
| LP formulation of LAD | Wagner (1959) | Shows absolute-deviation regression can be solved by linear programming. |
| Positive/negative residual variables | Wagner (1959); Narula & Wellington (1982) | Supports the implemented LP constraints and objective. |
| Specialized $L_1$ computation | Barrodale & Roberts (1973) | Supports simplex-based $L_1$ computation. |
| LAD curve-fitting computation | Bloomfield & Steiger (1980) | Supports computational comparison and non-differentiability discussion. |
| Sensitivity of least squares to extreme errors | Narula & Wellington (1982); Bloomfield & Steiger (1980) | Motivates the large-response-error experiment. |
| LAD under long-tailed errors | Bassett & Koenker (1978); Narula & Wellington (1982) | Motivates Laplace, Cauchy, and contaminated-error experiments. |
| Asymptotic theory of LAD | Bassett & Koenker (1978); Pollard (1991) | Provides theoretical background. |

## Core literature

1. Wagner (1959)
2. Barrodale & Roberts (1973)
3. Bassett & Koenker (1978)
4. Bloomfield & Steiger (1980)
5. Narula & Wellington (1982)
6. Pollard (1991)

Methods outside this six-paper basis, including MCD, Mahalanobis-distance screening, PCA outlier detection, cross-validation, bootstrap inference, permutation testing, and alternative robust regressors, are excluded from the active workflow.
