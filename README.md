# Regression by Minimizing Absolute Deviation

This repository is a literature-aligned computational study of ordinary least squares (OLS) and least absolute deviations (LAD) regression. The project is intentionally restricted to methods and ideas supported by the supplied LAD literature.

## Project objective

The central research question is:

> How does regression obtained by minimizing absolute deviations differ from regression obtained by minimizing squared deviations, particularly when observations contain large or extreme errors?

The project studies four connected themes: the mathematical difference between squared and absolute loss, the linear-programming formulation of LAD, the response of OLS and LAD to large vertical errors and long-tailed error distributions, and the computational cost of LAD relative to OLS.

## Literature basis

The methodology is based on the following supplied papers:

1. Wagner, H. M. (1959), *Linear Programming Techniques for Regression Analysis*.
2. Barrodale, I. and Roberts, F. D. K. (1973), *An Improved Algorithm for Discrete l1 Linear Approximation*.
3. Bassett, G. and Koenker, R. (1978), *Asymptotic Theory of Least Absolute Error Regression*.
4. Bloomfield, P. and Steiger, W. (1980), *Least Absolute Deviations Curve-Fitting*.
5. Narula, S. C. and Wellington, J. F. (1982), *The Minimum Sum of Absolute Errors Regression: A State of the Art Survey*.
6. Amemiya, T. (1982), *Two Stage Least Absolute Deviations Estimators*.
7. Powell, J. L. (1984), *Least Absolute Deviations Estimation for the Censored Regression Model*.
8. Pollard, D. (1991), *Asymptotics for Least Absolute Deviation Regression Estimators*.

Wagner and Barrodale-Roberts provide the linear-programming and computational foundations. Bassett-Koenker, Narula-Wellington, Bloomfield-Steiger and Pollard provide the statistical and robustness motivation. Amemiya and Powell are treated as extensions showing broader LAD applicability; this repository does not implement 2SLAD or censored LAD.

## OLS objective

For observations `(x_i, y_i)`, OLS chooses coefficients `beta` to minimize

```text
sum_i (y_i - x_i^T beta)^2
```

Because residuals are squared, large errors receive disproportionately large weight.

## LAD objective

LAD chooses coefficients `beta` to minimize

```text
sum_i |y_i - x_i^T beta|
```

This is the L1 / minimum-sum-of-absolute-errors criterion used throughout the supplied literature.

## Linear-programming formulation

The LAD residual is decomposed into non-negative positive and negative parts:

```text
y_i - x_i^T beta = e_i^+ - e_i^-
e_i^+ >= 0
e_i^- >= 0
```

and the optimization problem is

```text
minimize sum_i (e_i^+ + e_i^-)
```

The custom solver in `src/absolute_deviation/models.py` implements this formulation with `scipy.optimize.linprog`. It is a modern numerical solution of the literature-supported LP formulation; it is not claimed to reproduce the specialized Barrodale-Roberts algorithm.

## Datasets

The repository retains four experimental datasets:

- Boston Housing
- Concrete Compressive Strength
- Hawkins-Bradu-Kass (HBK)
- Synthetic OLS/LAD data

The analytical methods applied to these datasets are restricted to the supplied literature. No MCD, robust Mahalanobis distance, PCA outlier screening, bootstrap inference, Holm correction, repeated cross-validation, or external outlier-classification framework is used in the rebuilt workflow.

## Experiments

### 1. Original-data OLS/LAD comparison

Each dataset is fitted with OLS and LAD using the same rows and predictors. The repository reports coefficients, sum of squared errors (SSE), sum of absolute errors (SAE), MAE, RMSE and runtime.

### 2. Large-response-error experiment

A controlled synthetic experiment introduces deliberately large errors into selected response values. OLS and LAD are refitted and compared using coefficient change, prediction change, MAE and RMSE relative to the uncontaminated response. The selected contamination percentages are project-level experimental settings; they are not presented as values prescribed by the papers.

### 3. Error-distribution experiment

OLS and LAD are compared under error distributions explicitly motivated by the supplied literature:

- Normal
- Laplace / double exponential
- Cauchy / long-tailed
- contaminated normal

The experiment reports coefficient estimation error and both squared- and absolute-error criteria without adding external inferential machinery.

### 4. Computational comparison

Simple timing experiments compare OLS with the custom LP-based LAD solver over several sample sizes and predictor counts. Runtime results are implementation- and hardware-specific and are not used to claim reproduction of historical CPU-time results.

## Reproduce the study

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/generate_all_results.py
```

Run the tests with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Generated outputs

The rebuilt workflow creates:

```text
outputs/results/original_data_metrics.csv
outputs/results/original_data_coefficients.csv
outputs/results/contamination_metrics.csv
outputs/results/contamination_coefficient_changes.csv
outputs/results/distribution_experiment.csv
outputs/results/runtime_results.csv
outputs/results/lad_solver_validation.csv
```

and literature-aligned figures in `outputs/figures/`.

## Repository structure

```text
.
|-- data/
|   `-- processed/
|-- docs/
|-- outputs/
|   |-- figures/
|   `-- results/
|-- scripts/
|   `-- generate_all_results.py
|-- src/
|   `-- absolute_deviation/
|       |-- data.py
|       |-- experiments.py
|       |-- models.py
|       `-- plotting.py
|-- tests/
|-- pyproject.toml
|-- requirements.txt
`-- requirements-dev.txt
```

## Interpretation

The project does not assume LAD is universally superior. OLS directly optimizes squared error and remains attractive when the squared-error criterion, approximately Gaussian light-tailed errors, and computational simplicity are central. LAD directly optimizes absolute error and is particularly relevant when large vertical response errors or long-tailed distributions make squared-error fitting unstable.

## Limitations

- The project studies standard linear LAD only.
- 2SLAD and censored LAD are literature extensions, not implemented models.
- LAD is not claimed to be immune to all forms of unusual observations, especially extreme predictor configurations.
- The contamination experiment is illustrative and depends on its selected magnitude and proportions.
- Runtime depends on hardware, solver and data dimensions.
- The absolute-value objective is non-differentiable, which is one reason the literature emphasizes specialized computational approaches.

See `docs/literature_mapping.md`, `docs/theoretical_background.md`, `docs/experimental_design.md`, and `docs/literature_compliance_audit.md` for the full methodological mapping.
