"""Build paired OLS-versus-LAD effect comparisons."""

import numpy as np
import pandas as pd


SAMPLE_METRICS = (
    "SAE",
    "SSE",
    "MAE",
    "RMSE",
    "MedianAbsoluteError",
    "R2",
)
CONTAMINATION_METRICS = (
    "CleanMAE",
    "CleanRMSE",
    "PredictionRMSEFromCleanFit",
    "StandardizedSlopeL2Shift",
)


def pair_model_results(
    data: pd.DataFrame,
    key_columns: list[str],
    metric_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Join paired OLS/LAD rows and calculate signed model differences."""

    required = set(key_columns) | {"Model", *metric_columns}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"comparison data missing columns: {sorted(missing)}")
    if data.duplicated(key_columns + ["Model"]).any():
        raise ValueError("comparison data contain duplicate model keys")

    model_rows: dict[str, pd.DataFrame] = {}
    for model_name in ("OLS", "LAD"):
        model_data = data[data["Model"].eq(model_name)]
        if model_data.empty:
            raise ValueError(f"comparison data missing {model_name} rows")
        model_rows[model_name] = model_data.set_index(key_columns)

    if set(model_rows["OLS"].index) != set(model_rows["LAD"].index):
        raise ValueError("OLS and LAD comparison keys do not match")

    ols = model_rows["OLS"].sort_index()
    lad = model_rows["LAD"].reindex(ols.index)
    result = pd.DataFrame(index=ols.index)
    for metric in metric_columns:
        ols_values = ols[metric].astype(float)
        lad_values = lad[metric].astype(float)
        result[f"OLS_{metric}"] = ols_values
        result[f"LAD_{metric}"] = lad_values
        result[f"LADMinusOLS_{metric}"] = lad_values - ols_values
        if metric != "R2":
            denominator = ols_values.where(
                np.abs(ols_values) > np.finfo(float).eps
            )
            result[f"LADReductionPercent_{metric}"] = (
                100.0 * (ols_values - lad_values) / denominator
            )
            result[f"LADWins_{metric}"] = lad_values < ols_values
        else:
            result["LADWins_R2"] = lad_values > ols_values

    return result.reset_index()


def compare_sample_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair OLS and LAD for each observed fit/evaluation condition."""

    return pair_model_results(
        metrics,
        key_columns=[
            "Dataset",
            "FitCondition",
            "EvaluationSet",
            "TrainingRows",
            "EvaluationRows",
        ],
        metric_columns=SAMPLE_METRICS,
    )


def compare_contamination_repetitions(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair models receiving identical contamination in every repetition."""

    return pair_model_results(
        metrics,
        key_columns=[
            "Dataset",
            "ContaminationRate",
            "ContaminationPercent",
            "Repetition",
            "Rows",
            "ContaminatedRows",
            "NoiseSD",
        ],
        metric_columns=CONTAMINATION_METRICS,
    )


def summarize_contamination_comparison(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired effects and LAD win rates at every contamination rate."""

    rows: list[dict[str, object]] = []
    grouping = paired.groupby(
        ["Dataset", "ContaminationRate", "ContaminationPercent"],
        sort=False,
    )
    for keys, group in grouping:
        dataset, rate, percent = keys
        row: dict[str, object] = {
            "Dataset": dataset,
            "ContaminationRate": rate,
            "ContaminationPercent": percent,
            "Repetitions": group["Repetition"].nunique(),
            "ContaminatedRows": int(group["ContaminatedRows"].iloc[0]),
            "NoiseSD": float(group["NoiseSD"].iloc[0]),
        }
        for metric in CONTAMINATION_METRICS:
            ols_values = group[f"OLS_{metric}"].astype(float)
            lad_values = group[f"LAD_{metric}"].astype(float)
            differences = lad_values - ols_values
            row.update(
                {
                    f"OLS_{metric}_Mean": float(ols_values.mean()),
                    f"LAD_{metric}_Mean": float(lad_values.mean()),
                    f"LADMinusOLS_{metric}_Mean": float(differences.mean()),
                    f"LADMinusOLS_{metric}_SD": float(
                        differences.std(ddof=1)
                    ),
                    f"LADWinRate_{metric}": float((lad_values < ols_values).mean()),
                }
            )
            ols_mean = float(ols_values.mean())
            row[f"LADReductionPercent_{metric}"] = (
                100.0 * (ols_mean - float(lad_values.mean())) / ols_mean
                if abs(ols_mean) > np.finfo(float).eps
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_key_findings(
    comparison_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build one publication-friendly result row per dataset."""

    contamination = comparison_summary[
        comparison_summary["ContaminationPercent"].eq(20.0)
    ].copy()
    runtime = runtime_summary[
        runtime_summary["FitCondition"].eq("Inliers only")
        & runtime_summary["Model"].eq("LAD")
    ][["Dataset", "MedianRelativeToOLS"]]
    findings = contamination.merge(runtime, on="Dataset", validate="one_to_one")
    return findings[
        [
            "Dataset",
            "OLS_CleanMAE_Mean",
            "LAD_CleanMAE_Mean",
            "LADReductionPercent_CleanMAE",
            "LADWinRate_CleanMAE",
            "LADReductionPercent_PredictionRMSEFromCleanFit",
            "LADReductionPercent_StandardizedSlopeL2Shift",
            "MedianRelativeToOLS",
        ]
    ].rename(
        columns={
            "LADReductionPercent_CleanMAE": "MAEReductionPercent",
            "LADWinRate_CleanMAE": "MAEWinRate",
            "LADReductionPercent_PredictionRMSEFromCleanFit": (
                "PredictionDriftReductionPercent"
            ),
            "LADReductionPercent_StandardizedSlopeL2Shift": (
                "SlopeShiftReductionPercent"
            ),
            "MedianRelativeToOLS": "LADRuntimeMultiple",
        }
    )
