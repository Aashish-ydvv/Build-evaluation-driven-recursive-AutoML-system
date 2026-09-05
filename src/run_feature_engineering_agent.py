import json
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from agents.feature_engineering_agent import (
    FeatureEngineeringAgent,
)
from feature_engineering import DomainFeatureEngineer
from load_data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)

BENCHMARK_RESULTS_PATH = (
    EXPERIMENT_FOLDER
    / "benchmark_results.json"
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


def load_previous_champion_rmse():
    if not BENCHMARK_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Benchmark results were not found. "
            "Run run_benchmark_agent.py first."
        )

    with BENCHMARK_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        benchmark_results = json.load(file)

    champion_name = benchmark_results["champion"]

    champion_rmse = benchmark_results[
        "results"
    ][champion_name]["RMSE_mean"]

    print("Previous champion:", champion_name)
    print(
        "Previous champion RMSE:",
        f"{champion_rmse:.4f}",
    )

    return champion_rmse


def create_candidate_pipelines():
    base_random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    engineered_random_forest = Pipeline(
        steps=[
            (
                "feature_engineering",
                DomainFeatureEngineer(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    candidates = {
        "Random Forest - Base Features": (
            base_random_forest
        ),
        "Random Forest - Domain Features": (
            engineered_random_forest
        ),
    }

    return candidates


def main():
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

    previous_champion_rmse = (
        load_previous_champion_rmse()
    )

    candidate_pipelines = (
        create_candidate_pipelines()
    )

    agent = FeatureEngineeringAgent(
        output_folder=EXPERIMENT_FOLDER,
        minimum_improvement_percent=0.5,
    )

    agent.run(
        candidate_pipelines=candidate_pipelines,
        X_development=X_development,
        y_development=y_development,
        previous_champion_rmse=(
            previous_champion_rmse
        ),
    )

    print("\nThe protected test set was not evaluated.")


if __name__ == "__main__":
    main()