import json
from pathlib import Path


class ResultsAggregationAgent:
    """
    Combines experiment results and determines
    which configuration remains the champion.
    """

    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate_results(
        self,
        benchmark_results,
        feature_results,
    ):
        benchmark_champion = benchmark_results["champion"]

        benchmark_champion_rmse = benchmark_results[
            "results"
        ][benchmark_champion]["RMSE_mean"]

        feature_previous_rmse = feature_results[
            "previous_champion_rmse"
        ]

        difference = abs(
            benchmark_champion_rmse
            - feature_previous_rmse
        )

        if difference > 0.000001:
            raise ValueError(
                "The feature-engineering experiment does "
                "not match the benchmark champion."
            )

        return (
            benchmark_champion,
            benchmark_champion_rmse,
        )

    def run(
        self,
        benchmark_results,
        feature_results,
    ):
        print("\nResults Aggregation Agent started.")

        (
            benchmark_champion,
            benchmark_champion_rmse,
        ) = self.validate_results(
            benchmark_results,
            feature_results,
        )

        feature_candidate = feature_results[
            "best_candidate"
        ]

        feature_candidate_rmse = feature_results[
            "best_candidate_rmse"
        ]

        feature_promoted = feature_results["promoted"]

        if feature_promoted:
            current_champion = feature_candidate
            current_champion_rmse = (
                feature_candidate_rmse
            )

            champion_source = (
                "FeatureEngineeringAgent"
            )
        else:
            current_champion = benchmark_champion
            current_champion_rmse = (
                benchmark_champion_rmse
            )

            champion_source = (
                "ModelBenchmarkAgent"
            )

        experiment_history = [
            {
                "iteration": 0,
                "agent": "ModelBenchmarkAgent",
                "candidate": benchmark_champion,
                "rmse": benchmark_champion_rmse,
                "decision": "Promoted",
            },
            {
                "iteration": 1,
                "agent": "FeatureEngineeringAgent",
                "candidate": feature_candidate,
                "rmse": feature_candidate_rmse,
                "improvement_percent": (
                    feature_results[
                        "improvement_percent"
                    ]
                ),
                "decision": (
                    "Promoted"
                    if feature_promoted
                    else "Rejected"
                ),
                "reason": feature_results["decision"],
            },
        ]

        output = {
            "agent": "ResultsAggregationAgent",
            "current_champion": current_champion,
            "current_champion_rmse": (
                current_champion_rmse
            ),
            "champion_source": champion_source,
            "protected_test_used": False,
            "experiment_count": len(
                experiment_history
            ),
            "experiment_history": (
                experiment_history
            ),
        }

        output_path = (
            self.output_folder
            / "aggregated_results.json"
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

        print("Results successfully validated.")
        print(
            "Current champion:",
            current_champion,
        )
        print(
            "Current champion RMSE:",
            f"{current_champion_rmse:.4f}",
        )
        print(
            "Champion source:",
            champion_source,
        )
        print(
            "Protected test used:",
            output["protected_test_used"],
        )
        print(
            "Results saved to:",
            output_path,
        )
        print(
            "Results Aggregation Agent completed."
        )

        return output