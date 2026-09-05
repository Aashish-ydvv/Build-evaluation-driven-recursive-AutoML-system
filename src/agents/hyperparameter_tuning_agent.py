import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

from cross_validate_models import (
    evaluate_with_cross_validation,
)


class HyperparameterTuningAgent:
    """
    Executes an approved parameter search for
    supported tree-based models.
    """

    SUPPORTED_MODELS = {
        "Random Forest",
        "Extra Trees",
    }

    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate_experiment_plan(
        self,
        experiment_plan,
    ):
        if (
            experiment_plan["experiment_type"]
            != "hyperparameter_tuning"
        ):
            raise ValueError(
                "The experiment plan is not a "
                "hyperparameter-tuning experiment."
            )

        model_name = experiment_plan["model"]

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        if experiment_plan["protected_test_access"]:
            raise ValueError(
                "The experiment requested access to the "
                "protected test set."
            )

        parameter_grid = experiment_plan[
            "parameter_grid"
        ]

        actual_combinations = len(
            list(ParameterGrid(parameter_grid))
        )

        expected_combinations = experiment_plan[
            "total_combinations"
        ]

        if actual_combinations != expected_combinations:
            raise ValueError(
                "The number of parameter combinations "
                "does not match the approved plan."
            )

    def create_model(
        self,
        model_name,
        parameters,
    ):
        common_settings = {
            "random_state": 42,
            "n_jobs": -1,
        }

        if model_name == "Random Forest":
            return RandomForestRegressor(
                **parameters,
                **common_settings,
            )

        if model_name == "Extra Trees":
            return ExtraTreesRegressor(
                **parameters,
                **common_settings,
            )

        raise ValueError(
            f"Cannot create unsupported model: "
            f"{model_name}"
        )

    def create_output_path(self, model_name):
        safe_model_name = (
            model_name.lower().replace(" ", "_")
        )

        filename = (
            f"{safe_model_name}_tuning_results.json"
        )

        return self.output_folder / filename

    def run(
        self,
        experiment_plan,
        X_development,
        y_development,
        current_champion_rmse,
    ):
        print(
            "\nHyperparameter Tuning Agent started."
        )

        self.validate_experiment_plan(
            experiment_plan
        )

        model_name = experiment_plan["model"]

        parameter_combinations = list(
            ParameterGrid(
                experiment_plan["parameter_grid"]
            )
        )

        results = []

        for experiment_number, parameters in enumerate(
            parameter_combinations,
            start=1,
        ):
            print(
                f"\nExperiment "
                f"{experiment_number}/"
                f"{len(parameter_combinations)}"
            )

            print("Model:", model_name)
            print("Parameters:", parameters)

            model = self.create_model(
                model_name=model_name,
                parameters=parameters,
            )

            summary = evaluate_with_cross_validation(
                model,
                X_development,
                y_development,
            )

            result = {
                "experiment_number": (
                    experiment_number
                ),
                "model": model_name,
                "parameters": parameters,
                "metrics": summary,
            }

            results.append(result)

            print(
                "    Mean RMSE:",
                f"{summary['RMSE_mean']:.4f}",
            )

        best_result = min(
            results,
            key=lambda result: (
                result["metrics"]["RMSE_mean"]
            ),
        )

        best_rmse = best_result[
            "metrics"
        ]["RMSE_mean"]

        improvement_percent = (
            (
                current_champion_rmse
                - best_rmse
            )
            / current_champion_rmse
            * 100
        )

        required_improvement = experiment_plan[
            "minimum_improvement_percent"
        ]

        promoted = (
            improvement_percent
            >= required_improvement
        )

        if promoted:
            decision = (
                "Promoted because the tuned model passed "
                "the minimum improvement threshold."
            )
        else:
            decision = (
                "Rejected because the tuned model did not "
                "pass the minimum improvement threshold."
            )

        candidate_name = f"{model_name} - Tuned"

        output = {
            "agent": "HyperparameterTuningAgent",
            "model": model_name,
            "candidate": candidate_name,
            "current_champion_rmse": (
                current_champion_rmse
            ),
            "best_parameters": best_result[
                "parameters"
            ],
            "best_rmse": best_rmse,
            "best_rmse_standard_deviation": (
                best_result["metrics"]["RMSE_std"]
            ),
            "improvement_percent": (
                improvement_percent
            ),
            "required_improvement_percent": (
                required_improvement
            ),
            "promoted": promoted,
            "decision": decision,
            "protected_test_used": False,
            "results": results,
        }

        output_path = self.create_output_path(
            model_name
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=4,
            )

        print(
            "\nHyperparameter Tuning Agent completed."
        )

        print("Model:", model_name)
        print(
            "Candidate:",
            candidate_name,
        )

        print(
            "Best parameters:",
            output["best_parameters"],
        )

        print(
            "Previous champion RMSE:",
            f"{current_champion_rmse:.4f}",
        )

        print(
            "Best tuned RMSE:",
            f"{best_rmse:.4f}",
        )

        print(
            "Improvement:",
            f"{improvement_percent:.2f}%",
        )

        print("Promoted:", promoted)
        print("Decision:", decision)

        print(
            "Protected test used:",
            output["protected_test_used"],
        )

        print("Results saved to:", output_path)

        return output