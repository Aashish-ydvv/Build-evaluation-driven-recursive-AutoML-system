from math import sqrt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from load_data import load_data


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


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def print_metrics(model_name, metrics):
    print(f"\n{model_name}")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def main():
    dataframe = load_data()

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    # First split: reserve 20% as the untouched final test set.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Second split: divide the remaining 80% into:
    # 60% training and 20% validation.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=42,
    )

    print("Training rows:", X_train.shape[0])
    print("Validation rows:", X_validation.shape[0])
    print("Untouched test rows:", X_test.shape[0])

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),

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

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        validation_predictions = model.predict(X_validation)

        validation_metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        print_metrics(model_name, validation_metrics)

    print("\nModel comparison completed.")
    print("The final test set has not been evaluated.")


if __name__ == "__main__":
    main()