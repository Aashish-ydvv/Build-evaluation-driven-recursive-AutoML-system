import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_FOLDER = PROJECT_ROOT / "outputs"

EXPERIMENTS_FOLDER = (
    OUTPUTS_FOLDER
    / "experiments"
)

FINAL_RESULTS_FOLDER = (
    OUTPUTS_FOLDER
    / "final_results"
)

REPORTS_FOLDER = (
    OUTPUTS_FOLDER
    / "reports"
)

AGGREGATED_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "aggregated_results.json"
)

FINAL_METRICS_PATH = (
    FINAL_RESULTS_FOLDER
    / "final_test_metrics.json"
)

PREDICTIONS_PATH = (
    FINAL_RESULTS_FOLDER
    / "final_test_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = (
    FINAL_RESULTS_FOLDER
    / "feature_importance.csv"
)

FINAL_REPORT_PATH = (
    REPORTS_FOLDER
    / "final_project_report.md"
)


def check_required_files():
    required_files = [
        AGGREGATED_RESULTS_PATH,
        FINAL_METRICS_PATH,
        PREDICTIONS_PATH,
        FEATURE_IMPORTANCE_PATH,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            str(file_path)
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "The following required files are missing:\n"
            f"{missing_text}"
        )


def load_json_file(file_path):
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def create_experiment_table(experiment_history):
    table_lines = [
        (
            "| Iteration | Agent | Candidate | "
            "RMSE | Decision |"
        ),
        "|---:|---|---|---:|---|",
    ]

    for experiment in experiment_history:
        table_lines.append(
            f"| {experiment['iteration']} "
            f"| {experiment['agent']} "
            f"| {experiment['candidate']} "
            f"| {experiment['rmse']:.4f} "
            f"| {experiment['decision']} |"
        )

    return table_lines


def create_importance_table(importance_data):
    table_lines = [
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]

    for index, row in importance_data.iterrows():
        table_lines.append(
            f"| {index + 1} "
            f"| {row['feature']} "
            f"| {row['importance']:.4f} |"
        )

    return table_lines


def main():
    print("\nFinal report generation started.")

    check_required_files()

    aggregated_results = load_json_file(
        AGGREGATED_RESULTS_PATH
    )

    final_results = load_json_file(
        FINAL_METRICS_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    importance_data = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    final_metrics = final_results["metrics"]

    mean_residual = predictions[
        "residual"
    ].mean()

    median_absolute_error = predictions[
        "absolute_error"
    ].median()

    maximum_absolute_error = predictions[
        "absolute_error"
    ].max()

    experiment_table = create_experiment_table(
        aggregated_results[
            "experiment_history"
        ]
    )

    importance_table = create_importance_table(
        importance_data
    )

    report_lines = [
        "# Evaluation-Driven Recursive AutoML",
        "",
        (
            "A controlled multi-agent machine-learning "
            "system that proposes, evaluates, accepts or "
            "rejects experiments while protecting an "
            "untouched final test set."
        ),
        "",
        "## Problem statement",
        "",
        (
            "The project predicts building heating load "
            "from eight architectural design variables."
        ),
        "",
        (
            "The objective is not only to train one model. "
            "The system maintains a champion model, tests "
            "controlled improvements and promotes a new "
            "candidate only when it passes a predefined "
            "cross-validation threshold."
        ),
        "",
        "## Dataset",
        "",
        "- Dataset: UCI Energy Efficiency dataset",
        "- Total rows: 768",
        "- Input features: 8",
        "- Target: heating_load",
        "- Task type: regression",
        "- Missing values: 0",
        "- Duplicate rows: 0",
        "",
        "## System components",
        "",
        (
            "1. **Model Benchmark Agent:** compares "
            "approved baseline model families."
        ),
        (
            "2. **Feature Engineering Agent:** evaluates "
            "domain-informed transformations."
        ),
        (
            "3. **Results Aggregation Agent:** validates "
            "and combines experiment results."
        ),
        (
            "4. **Reporting Agent:** records the history "
            "and proposes the next controlled experiment."
        ),
        "",
        (
            "Specialized execution components perform "
            "approved hyperparameter and alternative-model "
            "experiments."
        ),
        "",
        "## Evaluation methodology",
        "",
        "- Development data: 614 rows",
        "- Protected test data: 154 rows",
        "- Development/test split: 80/20",
        "- Cross-validation: shuffled five-fold CV",
        "- Primary selection metric: mean RMSE",
        "- Promotion threshold: minimum 0.5% improvement",
        "- Final test evaluations: 1",
        "",
        (
            "The test set was not supplied to the "
            "experiment agents. It was evaluated once "
            "after model selection stopped."
        ),
        "",
        "## Experiment history",
        "",
        *experiment_table,
        "",
        "## Final champion",
        "",
        (
            f"- Model: "
            f"{final_results['champion']}"
        ),
        (
            f"- Cross-validation RMSE: "
            f"{final_results['cross_validation_rmse']:.4f}"
        ),
        "- Parameters:",
        (
            f"  - Number of trees: "
            f"{final_results['champion_parameters']['n_estimators']}"
        ),
        (
            f"  - Maximum depth: "
            f"{final_results['champion_parameters']['max_depth']}"
        ),
        (
            f"  - Minimum samples per leaf: "
            f"{final_results['champion_parameters']['min_samples_leaf']}"
        ),
        (
            f"  - Maximum features: "
            f"{final_results['champion_parameters']['max_features']}"
        ),
        "",
        "## Final protected-test results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| MAE | {final_metrics['MAE']:.4f} |",
        f"| RMSE | {final_metrics['RMSE']:.4f} |",
        f"| R2 | {final_metrics['R2']:.4f} |",
        "",
        (
            "The cross-validation RMSE and test RMSE were "
            "nearly identical, indicating consistent "
            "performance on the protected holdout."
        ),
        "",
        "## Feature importance",
        "",
        *importance_table,
        "",
        (
            "Tree-based feature importance describes how "
            "this model used the variables. It does not "
            "establish that a feature causes heating-load "
            "changes."
        ),
        "",
        "## Error analysis",
        "",
        (
            f"- Mean residual: "
            f"{mean_residual:.4f}"
        ),
        (
            f"- Median absolute error: "
            f"{median_absolute_error:.4f}"
        ),
        (
            f"- Maximum absolute error: "
            f"{maximum_absolute_error:.4f}"
        ),
        "",
        (
            "A small negative mean residual indicates "
            "slight average overprediction, but the value "
            "is close to zero."
        ),
        "",
        "## Diagnostic plots",
        "",
        (
            "![Final model diagnostics]"
            "(../figures/final_model_diagnostics.png)"
        ),
        "",
        "## Governance safeguards",
        "",
        "- The final test set was isolated before experiments.",
        "- Experiment agents received only development data.",
        "- Failed experiments were preserved.",
        "- Only approved model families were executed.",
        "- Promotion required a predefined improvement.",
        "- Duplicate history updates were prevented.",
        "- Test evaluation required explicit human approval.",
        "- The test set was evaluated only once.",
        "",
        "## Limitations",
        "",
        (
            "- The dataset contains simulated building "
            "configurations rather than measurements from "
            "operating buildings."
        ),
        (
            "- The dataset contains only 768 rows and a "
            "limited collection of design variables."
        ),
        (
            "- Climate, occupancy, construction materials "
            "and equipment schedules are not represented."
        ),
        (
            "- Feature importance may be affected by "
            "strongly correlated input variables."
        ),
        (
            "- The result should not replace professional "
            "building-energy simulation."
        ),
        "",
        "## Saved artifacts",
        "",
        (
            "- Final model: "
            "`outputs/models/extra_trees_final.joblib`"
        ),
        (
            "- Final metrics: "
            "`outputs/final_results/final_test_metrics.json`"
        ),
        (
            "- Test predictions: "
            "`outputs/final_results/final_test_predictions.csv`"
        ),
        (
            "- Feature importance: "
            "`outputs/final_results/feature_importance.csv`"
        ),
        (
            "- Diagnostic figure: "
            "`outputs/figures/final_model_diagnostics.png`"
        ),
        "",
        "## Reproducibility statement",
        "",
        (
            "All dataset splitting, model training and "
            "cross-validation operations use fixed random "
            "states. The experiment history records both "
            "accepted and rejected candidates."
        ),
        "",
    ]

    REPORTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = "\n".join(report_lines)

    FINAL_REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("Final report created:")
    print(FINAL_REPORT_PATH)

    print("\nFinal report generation completed.")


if __name__ == "__main__":
    main()