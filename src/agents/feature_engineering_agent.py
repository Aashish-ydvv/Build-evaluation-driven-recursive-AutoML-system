import json
from pathlib import Path

from cross_validate_models import evaluate_with_cross_validation


class FeatureEngineeringAgent:
    """
    Evaluates feature configurations using the current
    champion model.
    """

    def __init__(
        self,
        output_folder,
        minimum_improvement_percent=0.5,
    ):
        self.output_folder = Path(output_folder)

        self.minimum_improvement_percent = (
            minimum_improvement_percent
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(
        self,
        candidate_pipelines,
        X_development,
        y_development,
        previous_champion_rmse,
    ):
        print("\nFeature Engineering Agent started.")

        results = {}

        for candidate_name, pipeline in candidate_pipelines.items():
            print(f"\nEvaluating {candidate_name}")

            summary = evaluate_with_cross_validation(
                pipeline,
                X_development,
                y_development,
            )

            improvement_percent = (
                (
                    previous_champion_rmse
                    - summary["RMSE_mean"]
                )
                / previous_champion_rmse
                * 100
            )

            summary["improvement_percent"] = (
                improvement_percent
            )

            results[candidate_name] = summary

            print(
                "    Mean RMSE:",
                f"{summary['RMSE_mean']:.4f}",
            )

            print(
                "    Improvement:",
                f"{improvement_percent:.2f}%",
            )

        best_candidate = min(
            results,
            key=lambda name: results[name]["RMSE_mean"],
        )

        best_rmse = results[best_candidate]["RMSE_mean"]
        best_improvement = results[
            best_candidate
        ]["improvement_percent"]

        promoted = (
            best_improvement
            >= self.minimum_improvement_percent
        )

        if promoted:
            decision = (
                "Promoted because the improvement passed "
                "the required threshold."
            )
        else:
            decision = (
                "Rejected because the improvement did not "
                "pass the required threshold."
            )

        output = {
            "agent": "FeatureEngineeringAgent",
            "previous_champion_rmse": (
                previous_champion_rmse
            ),
            "minimum_improvement_percent": (
                self.minimum_improvement_percent
            ),
            "best_candidate": best_candidate,
            "best_candidate_rmse": best_rmse,
            "improvement_percent": best_improvement,
            "promoted": promoted,
            "decision": decision,
            "results": results,
        }

        output_path = (
            self.output_folder
            / "feature_engineering_results.json"
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

        print("\nFeature Engineering Agent completed.")
        print("Best candidate:", best_candidate)
        print("Promoted:", promoted)
        print("Decision:", decision)
        print("Results saved to:", output_path)

        return output