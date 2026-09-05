# Literature Compliance Audit

This audit applies to the rebuilt active workflow on `main`.

## Retained methods

### Ordinary least squares
- Supporting literature: Narula & Wellington (1982); Bassett & Koenker (1978).
- Literature concept: squared-deviation regression as the standard comparison criterion.
- Implementation: NumPy least-squares solution.
- Deviation from paper: numerical library implementation only; the statistical criterion is unchanged.

### Least absolute deviations
- Supporting literature: Bassett & Koenker (1978); Pollard (1991); Narula & Wellington (1982).
- Literature concept: minimize the sum of absolute residuals.
- Implementation: custom LP formulation.
- Deviation from paper: none in the optimization criterion.

### Linear-programming LAD formulation
- Supporting literature: Wagner (1959); Narula & Wellington (1982); Barrodale & Roberts (1973).
- Literature concept: represent absolute residuals with non-negative positive and negative parts and minimize their sum.
- Implementation: `scipy.optimize.linprog` with unrestricted regression coefficients and non-negative residual-part variables.
- Deviation from paper: a modern LP solver is used rather than a historical simplex implementation. The LP itself follows the literature-supported formulation.

### HBK multivariate case visualization
- Supporting basis: the regression residual concepts come from the LAD/OLS literature, while the regular, bad-leverage, and good-leverage labels are already present as `CaseGroup` values in the HBK dataset.
- Literature concept: compare squared-error and absolute-error regression on observations that include unusual/extreme cases.
- Implementation: fit OLS and LAD using all three HBK predictors (`X1`, `X2`, `X3`) and plot each observation's absolute OLS residual against its absolute LAD residual. Cases 15-75 are displayed as regular/inlier cases, cases 1-10 as bad-leverage/outlier cases, and cases 11-14 as good-leverage cases.
- Deviation from paper: no new outlier detector is introduced. The figure preserves the dataset-provided case labels instead of deriving labels through an external statistical procedure.

### Large-response-error experiment
- Supporting literature: Narula & Wellington (1982); Bloomfield & Steiger (1980); Barrodale & Roberts (1973).
- Literature concept: least squares can be strongly affected by extreme errors, while L1 fitting is less sensitive.
- Implementation: deliberately alter a selected fraction of response values and compare changes in OLS and LAD fits.
- Deviation from paper: exact percentages and error magnitude are project-selected illustrative settings and are explicitly labeled as such.

### Error-distribution experiment
- Supporting literature: Bassett & Koenker (1978); Bloomfield & Steiger (1980); Narula & Wellington (1982); Pollard (1991).
- Literature concept: contrast Gaussian behavior with Laplace, Cauchy, contaminated, and other long-tailed error settings.
- Implementation: repeated simulation using Normal, Laplace, Cauchy, and contaminated-normal errors.
- Deviation from paper: sample sizes and repetition counts are implementation settings; no unsupported inference is attached to them.

### Computational comparison
- Supporting literature: Wagner (1959); Barrodale & Roberts (1973); Bloomfield & Steiger (1980); Narula & Wellington (1982).
- Literature concept: LAD computation is a central practical issue and specialized algorithms can improve efficiency.
- Implementation: record wall-clock time for OLS and LP-based LAD over several problem sizes.
- Deviation from paper: timings are modern and machine-specific; no claim of reproducing historical CPU results is made.

## Literature extensions not implemented

### Two-stage LAD
- Source: Amemiya (1982).
- Status: theoretical/literature discussion only because the project studies standard linear regression rather than simultaneous equations.

### Censored LAD
- Source: Powell (1984).
- Status: theoretical/literature discussion only because the project does not study censored dependent variables.

## Removed from the active workflow

The rebuild excludes the following methods because they are not part of the supplied methodological basis for this project:

- Minimum Covariance Determinant
- robust Mahalanobis distance
- chi-square outlier thresholds
- PCA outlier screening
- externally computed leverage/response-outlier classification
- repeated cross-validation
- bootstrap confidence intervals
- paired bootstrap
- permutation/randomization inference
- Holm multiple-testing adjustment
- modern alternative robust regressors

The HBK `CaseGroup` field is not an externally computed classification in this repository; it is existing dataset metadata and is used only for visualization.

## Compliance conclusion

The active workflow is limited to squared-loss regression, absolute-loss regression, the literature-supported linear-programming formulation of LAD, descriptive visualization of the existing HBK case groups, direct experiments with extreme response errors and literature-discussed error distributions, and computational comparisons. Any project-specific numerical settings are labeled as experiment settings rather than literature-derived prescriptions.
