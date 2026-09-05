import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from cross_validate_models import (
    evaluate_with_cross_validation,
)


class AlternativeModelAgent:
    """
    Evaluates approved alternative model families
    against the current champion.
    """

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
            != "alternative_model_benchmark"
        ):
            raise ValueError(
                "The experiment is not an approved "
                "alternative-model benchmark."
            )

        if experiment_plan["protected_test_access"]:
            raise ValueError(
                "The experiment requested access to the "
                "protected test set."
            )

        approved_models = {
            "Extra Trees",
            "Gradient Boosting",
        }

        requested_models = set(
            experiment_plan["models"]
        )

        unsupported_models = (
            requested_models - approved_models
        )

        if unsupported_models:
            raise ValueError(
                f"Unsupported models requested: "
                f"{unsupported_models}"
            )

    def create_models(self, requested_models):
        available_models = {
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),

            "Gradient Boosting": (
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                    loss="squared_error",
                )
            ),
        }

        selected_models = {
            model_name: available_models[model_name]
            for model_name in requested_models
        }

        return selected_models

    def run(
        self,
        experiment_plan,
        X_development,
        y_development,
        current_champion_rmse,
    ):
        print("\nAlternative Model Agent started.")

        self.validate_experiment_plan(
            experiment_plan
        )

        models = self.create_models(
            experiment_plan["models"]
        )

        results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}")

            summary = evaluate_with_cross_validation(
                model,
                X_development,
                y_development,
            )

            improvement_percent = (
                (
                    current_champion_rmse
                    - summary["RMSE_mean"]
                )
                / current_champion_rmse
                * 100
            )

            summary["improvement_percent"] = (
                improvement_percent
            )

            results[model_name] = summary

            print(
                "    Mean RMSE:",
                f"{summary['RMSE_mean']:.4f}",
            )

            print(
                "    Improvement over champion:",
                f"{improvement_percent:.2f}%",
            )

        best_model = min(
            results,
            key=lambda name: (
                results[name]["RMSE_mean"]
            ),
        )

        best_rmse = results[
            best_model
        ]["RMSE_mean"]

        best_improvement = results[
            best_model
        ]["improvement_percent"]

        required_improvement = experiment_plan[
            "minimum_improvement_percent"
        ]

        promoted = (
            best_improvement
            >= required_improvement
        )

        if promoted:
            decision = (
                "Promoted because the alternative model "
                "passed the improvement threshold."
            )
        else:
            decision = (
                "Rejected because the alternative model "
                "did not pass the improvement threshold."
            )

        output = {
            "agent": "AlternativeModelAgent",
            "previous_champion_rmse": (
                current_champion_rmse
            ),
            "best_model": best_model,
            "best_rmse": best_rmse,
            "best_rmse_standard_deviation": (
                results[best_model]["RMSE_std"]
            ),
            "improvement_percent": (
                best_improvement
            ),
            "required_improvement_percent": (
                required_improvement
            ),
            "promoted": promoted,
            "decision": decision,
            "protected_test_used": False,
            "results": results,
        }

        output_path = (
            self.output_folder
            / "alternative_model_results.json"
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

        print("\nAlternative Model Agent completed.")
        print("Best model:", best_model)
        print(
            "Previous champion RMSE:",
            f"{current_champion_rmse:.4f}",
        )
        print(
            "Best alternative RMSE:",
            f"{best_rmse:.4f}",
        )
        print(
            "Improvement:",
            f"{best_improvement:.2f}%",
        )
        print("Promoted:", promoted)
        print("Decision:", decision)
        print(
            "Protected test used:",
            output["protected_test_used"],
        )
        print("Results saved to:", output_path)

        return output