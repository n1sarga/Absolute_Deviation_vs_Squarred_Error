"""Run formal descriptive comparison of OLS and LAD results."""

import pandas as pd

from .benchmark import (
    benchmark_condition_models,
    runtime_environment,
    summarize_runtime,
)
from .comparison import (
    build_key_findings,
    compare_contamination_repetitions,
    compare_sample_conditions,
    summarize_contamination_comparison,
)
from .config import DATASETS, RESULT_DIR
from .detection import load_dataset
from .robustness_conditions import prepare_condition_fits


def _load_required_result(filename: str) -> pd.DataFrame:
    """Read a prior-stage result or explain how to create it."""

    path = RESULT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required result missing: {path}. Run robustness experiments first."
        )
    return pd.read_csv(path)


def run_formal_evaluation() -> pd.DataFrame:
    """Create paired effects, timing summaries, key findings, and figures."""

    from .evaluation_plotting import (
        save_model_comparison_dashboard,
        save_runtime_figure,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sample_metrics = _load_required_result("sample_condition_metrics.csv")
    contamination_metrics = _load_required_result(
        "contamination_experiment_metrics.csv"
    )

    sample_comparison = compare_sample_conditions(sample_metrics)
    contamination_comparison = compare_contamination_repetitions(
        contamination_metrics
    )
    contamination_summary = summarize_contamination_comparison(
        contamination_comparison
    )

    conditions = [
        prepare_condition_fits(spec, load_dataset(spec)) for spec in DATASETS
    ]
    runtime_benchmark = benchmark_condition_models(conditions)
    runtime_summary = summarize_runtime(runtime_benchmark)
    environment = runtime_environment()
    findings = build_key_findings(contamination_summary, runtime_summary)

    tables = {
        "sample_condition_model_comparison.csv": sample_comparison,
        "contamination_model_comparison.csv": contamination_comparison,
        "contamination_model_comparison_summary.csv": contamination_summary,
        "runtime_benchmark.csv": runtime_benchmark,
        "runtime_summary.csv": runtime_summary,
        "runtime_environment.csv": environment,
        "stage5_key_findings.csv": findings,
    }
    for filename, table in tables.items():
        table.to_csv(
            RESULT_DIR / filename,
            index=False,
            float_format="%.10g",
        )

    save_model_comparison_dashboard(findings)
    save_runtime_figure(runtime_summary)

    print("Step 5 key findings at 20% response contamination:")
    print(findings.to_string(index=False))
    print("\nRuntime environment:")
    print(environment.to_string(index=False))
    return findings


def main() -> None:
    """Console-script entry point."""

    run_formal_evaluation()
