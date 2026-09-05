import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from agents.alternative_model_agent import (
    AlternativeModelAgent,
)
from load_data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)

NEXT_EXPERIMENT_PATH = (
    EXPERIMENTS_FOLDER
    / "next_experiment.json"
)

AGGREGATED_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "aggregated_results.json"
)

FEATURE_COLUMNS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_distribution",
]

TARGET_COLUMN = "heating_load"


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


def main():
    experiment_plan = load_json_file(
        NEXT_EXPERIMENT_PATH
    )

    aggregated_results = load_json_file(
        AGGREGATED_RESULTS_PATH
    )

    dataframe = load_data()

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    print("Development rows:", X_development.shape[0])
    print("Protected test rows:", X_test.shape[0])

    current_champion = aggregated_results[
        "current_champion"
    ]

    current_champion_rmse = aggregated_results[
        "current_champion_rmse"
    ]

    print("Current champion:", current_champion)
    print(
        "Current champion RMSE:",
        f"{current_champion_rmse:.4f}",
    )

    agent = AlternativeModelAgent(
        output_folder=EXPERIMENTS_FOLDER,
    )

    agent.run(
        experiment_plan=experiment_plan,
        X_development=X_development,
        y_development=y_development,
        current_champion_rmse=(
            current_champion_rmse
        ),
    )

    print("\nThe protected test set was not evaluated.")


if __name__ == "__main__":
    main()