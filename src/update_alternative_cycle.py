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

ALTERNATIVE_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "alternative_model_results.json"
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
    print("\nAlternative-model cycle update started.")

    aggregated_results = load_json_file(
        AGGREGATED_RESULTS_PATH
    )

    alternative_results = load_json_file(
        ALTERNATIVE_RESULTS_PATH
    )

    current_champion_rmse = aggregated_results[
        "current_champion_rmse"
    ]

    experiment_starting_rmse = alternative_results[
        "previous_champion_rmse"
    ]

    difference = abs(
        current_champion_rmse
        - experiment_starting_rmse
    )

    if difference > 0.000001:
        raise ValueError(
            "The alternative-model experiment does not "
            "match the current champion."
        )

    existing_agents = [
        experiment["agent"]
        for experiment in aggregated_results[
            "experiment_history"
        ]
    ]

    if "AlternativeModelAgent" in existing_agents:
        raise ValueError(
            "The alternative-model result has already "
            "been added to the experiment history."
        )

    alternative_experiment = {
        "iteration": len(
            aggregated_results[
                "experiment_history"
            ]
        ),
        "agent": "AlternativeModelAgent",
        "candidate": alternative_results[
            "best_model"
        ],
        "rmse": alternative_results[
            "best_rmse"
        ],
        "rmse_standard_deviation": (
            alternative_results[
                "best_rmse_standard_deviation"
            ]
        ),
        "improvement_percent": (
            alternative_results[
                "improvement_percent"
            ]
        ),
        "decision": (
            "Promoted"
            if alternative_results["promoted"]
            else "Rejected"
        ),
        "reason": alternative_results["decision"],
    }

    aggregated_results[
        "experiment_history"
    ].append(alternative_experiment)

    if alternative_results["promoted"]:
        aggregated_results[
            "current_champion"
        ] = alternative_results["best_model"]

        aggregated_results[
            "current_champion_rmse"
        ] = alternative_results["best_rmse"]

        aggregated_results[
            "champion_source"
        ] = "AlternativeModelAgent"

        if (
            alternative_results["best_model"]
            == "Extra Trees"
        ):
            aggregated_results[
                "current_champion_parameters"
            ] = {
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1,
            }

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
        "Alternative-model result added to history."
    )

    print(
        "Experiment decision:",
        alternative_experiment["decision"],
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
        "Champion source:",
        aggregated_results["champion_source"],
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

    print(
        "Alternative-model cycle update completed."
    )


if __name__ == "__main__":
    main()