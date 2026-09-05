"""Literature-aligned OLS and least-absolute-deviation regression."""

from .models import RegressionResult, fit_lad, fit_ols, regression_metrics

__all__ = ["RegressionResult", "fit_lad", "fit_ols", "regression_metrics"]
