# Experimental Design

The experiments are restricted to questions directly motivated by the six core papers retained for the project: Wagner (1959), Barrodale and Roberts (1973), Bassett and Koenker (1978), Bloomfield and Steiger (1980), Narula and Wellington (1982), and Pollard (1991).

## Research question

How does regression obtained by minimizing absolute deviations differ from regression obtained by minimizing squared deviations, particularly when observations contain large or extreme errors?

## Experiment 1: original empirical data

Only three empirical datasets are used: Boston Housing, Concrete Strength, and Hawkins-Bradu-Kass (HBK). The same observations and predictors are fitted with OLS and LAD. The comparison reports coefficients, SSE, SAE, MAE, RMSE, and runtime.

The previously included synthetic OLS/LAD CSV dataset is excluded from this experiment.

The purpose is descriptive: OLS should directly minimize SSE and LAD should directly minimize SAE on the fitted sample. The project does not infer universal out-of-sample superiority from these fits.

### HBK multivariate inlier/outlier visualization

The project-objective figure uses the case labels already stored in the HBK data rather than applying a separate outlier-detection method. Cases 1-10 are labeled as bad-leverage observations, cases 11-14 as good-leverage observations, and cases 15-75 as regular observations. The visualization treats the regular cases as inliers, identifies the bad-leverage cases as the outlier group, and preserves good-leverage cases as a separate non-regular category.

OLS and LAD are each fitted using all three HBK predictors (`X1`, `X2`, and `X3`) at the same time. The figure plots the absolute OLS residual against the absolute LAD residual for every observation. This allows the multivariate dataset to be visualized in two dimensions without reducing the regression to a single predictor. The case grouping is descriptive metadata supplied with the HBK data; it is not an MCD, Mahalanobis, PCA, IQR, or other external classification rule.

## Experiment 2: large vertical response errors

A controlled regression design is generated internally for this experiment rather than loaded as an empirical dataset. Selected response values are then given deliberately large errors and both OLS and LAD are refitted. The experiment records coefficient change, mean absolute change in fitted values, MAE and RMSE relative to the uncontaminated response, and fit runtime.

The contamination levels 0%, 5%, 10%, and 20% and the chosen error magnitude are project-selected illustration settings. They are not represented as numerical prescriptions from the papers. Their purpose is to operationalize the retained literature's concern with extreme errors and contaminated observations.

## Experiment 3: error distributions

The controlled simulation compares Normal, Laplace, Cauchy, and contaminated-normal errors. These distributions are included because the retained papers contrast the Gaussian setting with long-tailed/non-Gaussian cases and discuss Laplace, Cauchy, and contaminated-normal situations.

The same true regression coefficients are used for OLS and LAD. Repeated samples are summarized using coefficient estimation error, SSE, SAE, MAE, and RMSE. Only simple descriptive summaries are used.

## Experiment 4: computational comparison

OLS and LP-based LAD are timed over several sample sizes and predictor counts. The goal is to illustrate the computational trade-off emphasized by Wagner (1959), Barrodale and Roberts (1973), Bloomfield and Steiger (1980), and Narula and Wellington (1982).

The repository does not claim that its modern LP solver reproduces the historical Barrodale-Roberts algorithm or its CPU-time results.

## Methods deliberately excluded

The active experimental design does not use MCD, Mahalanobis-distance screening, PCA outlier detection, k-fold cross-validation, bootstrap intervals, permutation/randomization tests, Holm correction, or external robust-regression methods because those are outside the six-paper methodological basis for this project.
