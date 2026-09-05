from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from agents.model_benchmark_agent import ModelBenchmarkAgent
from load_data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
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


def create_models():
    models = {
        "Mean Baseline": DummyRegressor(
            strategy="mean",
        ),

        "Linear Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    return models


def main():
    dataframe = load_data()

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("Development rows:", X_development.shape[0])
    print("Protected test rows:", X_test.shape[0])

    models = create_models()

    agent = ModelBenchmarkAgent(
        output_folder=OUTPUT_FOLDER,
    )

    agent.run(
        models=models,
        X_development=X_development,
        y_development=y_development,
    )

    print("\nThe protected test set was not evaluated.")


if __name__ == "__main__":
    main()