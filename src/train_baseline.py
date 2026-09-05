from math import sqrt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nTraining rows:", X_train.shape[0])
    print("Testing rows:", X_test.shape[0])

    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)

    baseline_predictions = baseline_model.predict(X_test)
    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    linear_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_test)
    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions,
    )

    print_metrics("Mean Baseline", baseline_metrics)
    print_metrics("Linear Regression", linear_metrics)


if __name__ == "__main__":
    main()