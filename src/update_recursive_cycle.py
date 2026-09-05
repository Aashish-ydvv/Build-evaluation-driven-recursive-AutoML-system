import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)

AGGREGATED_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "aggregated_results.json"
)

TUNING_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "hyperparameter_tuning_results.json"
)


def load_json_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n"
            f"{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json_file(file_path, data):
    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def main():
    print("\nRecursive cycle update started.")

    aggregated_results = load_json_file(
        AGGREGATED_RESULTS_PATH
    )

    tuning_results = load_json_file(
        TUNING_RESULTS_PATH
    )

    current_champion_rmse = aggregated_results[
        "current_champion_rmse"
    ]

    tuning_starting_rmse = tuning_results[
        "current_champion_rmse"
    ]

    difference = abs(
        current_champion_rmse
        - tuning_starting_rmse
    )

    if difference > 0.000001:
        raise ValueError(
            "Tuning results do not match the "
            "current champion."
        )

    existing_agents = [
        experiment["agent"]
        for experiment in aggregated_results[
            "experiment_history"
        ]
    ]

    if "HyperparameterTuningAgent" in existing_agents:
        raise ValueError(
            "The tuning result has already been "
            "added to the experiment history."
        )

    tuning_experiment = {
        "iteration": len(
            aggregated_results[
                "experiment_history"
            ]
        ),
        "agent": "HyperparameterTuningAgent",
        "candidate": "Random Forest - Tuned",
        "rmse": tuning_results["best_rmse"],
        "improvement_percent": (
            tuning_results["improvement_percent"]
        ),
        "parameters": (
            tuning_results["best_parameters"]
        ),
        "decision": (
            "Promoted"
            if tuning_results["promoted"]
            else "Rejected"
        ),
        "reason": tuning_results["decision"],
    }

    aggregated_results[
        "experiment_history"
    ].append(tuning_experiment)

    if tuning_results["promoted"]:
        aggregated_results[
            "current_champion"
        ] = "Random Forest - Tuned"

        aggregated_results[
            "current_champion_rmse"
        ] = tuning_results["best_rmse"]

        aggregated_results[
            "champion_source"
        ] = "HyperparameterTuningAgent"

        aggregated_results[
            "current_champion_parameters"
        ] = tuning_results["best_parameters"]

    aggregated_results["experiment_count"] = len(
        aggregated_results["experiment_history"]
    )

    aggregated_results[
        "protected_test_used"
    ] = False

    save_json_file(
        AGGREGATED_RESULTS_PATH,
        aggregated_results,
    )

    print(
        "Tuning result added to experiment history."
    )
    print(
        "Tuning decision:",
        tuning_experiment["decision"],
    )
    print(
        "Current champion:",
        aggregated_results["current_champion"],
    )
    print(
        "Current champion RMSE:",
        f"{aggregated_results['current_champion_rmse']:.4f}",
    )
    print(
        "Experiment count:",
        aggregated_results["experiment_count"],
    )
    print(
        "Protected test used:",
        aggregated_results["protected_test_used"],
    )
    print(
        "Updated:",
        AGGREGATED_RESULTS_PATH,
    )
    print("Recursive cycle update completed.")


if __name__ == "__main__":
    main()