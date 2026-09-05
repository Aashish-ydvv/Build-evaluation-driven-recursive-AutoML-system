import json
from pathlib import Path


class ReportingAgent:
    """
    Creates a readable experiment report and proposes
    the next controlled experiment.
    """

    def __init__(
        self,
        reports_folder,
        experiments_folder,
    ):
        self.reports_folder = Path(reports_folder)
        self.experiments_folder = Path(experiments_folder)

        self.reports_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.experiments_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_next_experiment(
        self,
        aggregated_results,
    ):
        current_champion = aggregated_results[
            "current_champion"
        ]

        experiment_history = aggregated_results[
            "experiment_history"
        ]

        random_forest_tuning_completed = any(
            experiment["agent"]
            == "HyperparameterTuningAgent"
            and experiment["candidate"]
            == "Random Forest - Tuned"
            for experiment in experiment_history
        )

        alternative_models_completed = any(
            experiment["agent"]
            == "AlternativeModelAgent"
            for experiment in experiment_history
        )

        extra_trees_tuning_completed = any(
            experiment["agent"]
            == "HyperparameterTuningAgent"
            and experiment["candidate"]
            == "Extra Trees - Tuned"
            for experiment in experiment_history
        )

        if (
            current_champion == "Random Forest"
            and not random_forest_tuning_completed
        ):
            recommendation = {
                "experiment_type": (
                    "hyperparameter_tuning"
                ),
                "model": "Random Forest",
                "reason": (
                    "Random Forest is the current champion, "
                    "and its approved tuning experiment has "
                    "not been completed."
                ),
                "parameter_grid": {
                    "n_estimators": [200],
                    "max_depth": [None, 12],
                    "min_samples_leaf": [1, 2],
                    "max_features": [1.0, "sqrt"],
                },
                "total_combinations": 8,
                "primary_metric": "RMSE_mean",
                "lower_is_better": True,
                "minimum_improvement_percent": 0.5,
                "protected_test_access": False,
            }

        elif (
            current_champion == "Random Forest"
            and random_forest_tuning_completed
            and not alternative_models_completed
        ):
            recommendation = {
                "experiment_type": (
                    "alternative_model_benchmark"
                ),
                "models": [
                    "Extra Trees",
                    "Gradient Boosting",
                ],
                "reason": (
                    "Feature engineering and Random Forest "
                    "tuning did not pass the promotion gate. "
                    "The next iteration should compare other "
                    "tree-based model families."
                ),
                "primary_metric": "RMSE_mean",
                "lower_is_better": True,
                "minimum_improvement_percent": 0.5,
                "protected_test_access": False,
            }

        elif (
            current_champion.startswith("Extra Trees")
            and not extra_trees_tuning_completed
        ):
            recommendation = {
                "experiment_type": (
                    "hyperparameter_tuning"
                ),
                "model": "Extra Trees",
                "reason": (
                    "Extra Trees improved RMSE and became "
                    "the new champion. The next experiment "
                    "will test a small approved parameter "
                    "grid around this model."
                ),
                "parameter_grid": {
                    "n_estimators": [200, 400],
                    "max_depth": [None, 12],
                    "min_samples_leaf": [1, 2],
                    "max_features": [1.0],
                },
                "total_combinations": 8,
                "primary_metric": "RMSE_mean",
                "lower_is_better": True,
                "minimum_improvement_percent": 0.5,
                "protected_test_access": False,
            }

        elif (
            current_champion.startswith("Extra Trees")
            and extra_trees_tuning_completed
        ):
            recommendation = {
                "experiment_type": (
                    "stop_and_review"
                ),
                "model": "Extra Trees - Tuned",
                "reason": (
                    "The approved model comparison and "
                    "Extra Trees tuning experiments are "
                    "complete. Human review is required "
                    "before evaluating the final test set."
                ),
                "protected_test_access": False,
            }

        else:
            recommendation = {
                "experiment_type": (
                    "manual_review_required"
                ),
                "model": current_champion,
                "reason": (
                    "No approved automatic experiment is "
                    "available for the current champion."
                ),
                "protected_test_access": False,
            }

        return recommendation

    def create_markdown_report(
        self,
        aggregated_results,
        next_experiment,
    ):
        model_or_models = next_experiment.get(
            "model",
            next_experiment.get("models"),
        )

        report_lines = [
            "# Recursive AutoML Experiment Report",
            "",
            "## Current champion",
            "",
            (
                f"- Model: "
                f"{aggregated_results['current_champion']}"
            ),
            (
                f"- Cross-validation RMSE: "
                f"{aggregated_results['current_champion_rmse']:.4f}"
            ),
            (
                f"- Champion source: "
                f"{aggregated_results['champion_source']}"
            ),
            (
                f"- Protected test used: "
                f"{aggregated_results['protected_test_used']}"
            ),
            "",
            "## Experiment history",
            "",
            (
                "| Iteration | Agent | Candidate | "
                "RMSE | Decision |"
            ),
            "|---:|---|---|---:|---|",
        ]

        for experiment in aggregated_results[
            "experiment_history"
        ]:
            report_lines.append(
                f"| {experiment['iteration']} "
                f"| {experiment['agent']} "
                f"| {experiment['candidate']} "
                f"| {experiment['rmse']:.4f} "
                f"| {experiment['decision']} |"
            )

        report_lines.extend(
            [
                "",
                "## Next controlled experiment",
                "",
                (
                    f"- Experiment type: "
                    f"{next_experiment['experiment_type']}"
                ),
                (
                    f"- Model or models: "
                    f"{model_or_models}"
                ),
                (
                    f"- Reason: "
                    f"{next_experiment['reason']}"
                ),
                (
                    f"- Protected test access: "
                    f"{next_experiment['protected_test_access']}"
                ),
                "",
                "## Governance rules",
                "",
                "- The final test set remains untouched.",
                "- Failed experiments remain in the history.",
                (
                    "- A candidate must improve RMSE by at "
                    "least 0.5% before promotion."
                ),
                (
                    "- Only parameters from the approved "
                    "configuration may be tested."
                ),
                "",
            ]
        )

        return "\n".join(report_lines)

    def run(self, aggregated_results):
        print("\nReporting Agent started.")

        next_experiment = self.create_next_experiment(
            aggregated_results
        )

        next_experiment_path = (
            self.experiments_folder
            / "next_experiment.json"
        )

        with next_experiment_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                next_experiment,
                file,
                indent=4,
            )

        markdown_report = self.create_markdown_report(
            aggregated_results,
            next_experiment,
        )

        report_path = (
            self.reports_folder
            / "iteration_report.md"
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(markdown_report)

        print(
            "Current champion:",
            aggregated_results["current_champion"],
        )

        print(
            "Next experiment:",
            next_experiment["experiment_type"],
        )

        print(
            "Model or models:",
            next_experiment.get(
                "model",
                next_experiment.get("models"),
            ),
        )

        if "total_combinations" in next_experiment:
            print(
                "Total parameter combinations:",
                next_experiment[
                    "total_combinations"
                ],
            )

        print(
            "Protected test access:",
            next_experiment[
                "protected_test_access"
            ],
        )

        print(
            "Experiment plan saved to:",
            next_experiment_path,
        )

        print("Report saved to:", report_path)
        print("Reporting Agent completed.")

        return {
            "next_experiment": next_experiment,
            "report_path": str(report_path),
        }