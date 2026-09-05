# Literature Mapping

This document maps every retained methodological component of the rebuilt repository to the supplied literature. If a method cannot be mapped to these papers, it is excluded from the active workflow.

| Repository component | Supporting literature | Role in this project |
|---|---|---|
| OLS versus absolute-error regression | Narula & Wellington (1982); Bassett & Koenker (1978) | Establishes the contrast between squared and absolute loss. |
| LAD objective `sum |residual|` | Bassett & Koenker (1978); Pollard (1991) | Defines the least-absolute-deviation estimator. |
| L1 / absolute-error terminology | Bassett & Koenker (1978); Narula & Wellington (1982) | Connects LAD, LAE, L1 and minimum-sum-of-absolute-errors terminology. |
| LP formulation of LAD | Wagner (1959) | Shows absolute-deviation regression can be solved by linear programming. |
| Positive/negative residual variables | Wagner (1959); Narula & Wellington (1982) | Basis of the implemented LP constraints and objective. |
| Specialized L1 computation | Barrodale & Roberts (1973) | Establishes the computational importance of simplex-based L1 algorithms. |
| LAD curve-fitting computation | Bloomfield & Steiger (1980) | Supports computational comparison and non-differentiability discussion. |
| Sensitivity of least squares to extreme errors | Narula & Wellington (1982); Bloomfield & Steiger (1980) | Motivates the large-response-error experiment. |
| LAD robustness under long tails | Bassett & Koenker (1978); Narula & Wellington (1982) | Motivates Laplace/Cauchy/contaminated-error experiments. |
| Laplace / double-exponential setting | Bloomfield & Steiger (1980); Narula & Wellington (1982) | Supports including Laplace errors in simulation. |
| Cauchy / thick-tailed setting | Narula & Wellington (1982); Pollard (1991) | Supports long-tailed simulation examples. |
| Asymptotic normality of LAD | Bassett & Koenker (1978); Pollard (1991) | Theoretical background only; not used to add unsupported inference. |
| Convex criterion perspective | Pollard (1991) | Explains theoretical treatment of LAD minimization. |
| Two-stage LAD | Amemiya (1982) | Discussed as an extension; not implemented. |
| Censored LAD | Powell (1984) | Discussed as an extension; not implemented. |

## Explicit exclusions

The following methods are not part of the rebuilt workflow because they are not part of the supplied methodological basis for this project:

- Minimum Covariance Determinant (MCD)
- robust Mahalanobis distance
- PCA-based outlier screening
- externally defined leverage/outlier classification frameworks
- repeated k-fold cross-validation
- bootstrap confidence intervals
- paired bootstrap inference
- permutation/randomization testing
- Holm or other multiplicity correction
- Huber regression as an implemented comparator
- RANSAC
- Theil-Sen regression
- ridge, lasso or elastic-net regularization

The repository may retain historical Git history containing earlier versions, but the active `main` workflow does not rely on these methods after the rebuild.
