import json
from pathlib import Path

from cross_validate_models import evaluate_with_cross_validation


class ModelBenchmarkAgent:
    """
    Evaluates approved models using cross-validation
    and selects the model with the lowest mean RMSE.
    """

    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self, models, X_development, y_development):
        print("\nModel Benchmark Agent started.")

        results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}")

            summary = evaluate_with_cross_validation(
                model,
                X_development,
                y_development,
            )

            results[model_name] = summary

            print(
                "    Mean RMSE:",
                f"{summary['RMSE_mean']:.4f}",
            )

            print(
                "    RMSE standard deviation:",
                f"{summary['RMSE_std']:.4f}",
            )

        champion_name = min(
            results,
            key=lambda name: results[name]["RMSE_mean"],
        )

        output = {
            "agent": "ModelBenchmarkAgent",
            "primary_metric": "RMSE_mean",
            "lower_is_better": True,
            "champion": champion_name,
            "results": results,
        }

        output_path = (
            self.output_folder
            / "benchmark_results.json"
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

        print("\nModel Benchmark Agent completed.")
        print("Champion:", champion_name)
        print("Results saved to:", output_path)

        return output