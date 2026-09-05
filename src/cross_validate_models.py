from math import sqrt
from statistics import mean
from statistics import stdev
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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


def evaluate_with_cross_validation(model, X, y):
    cross_validator = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_metrics = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cross_validator.split(X),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        fold_model = clone(model)

        fold_model.fit(X_train, y_train)

        predictions = fold_model.predict(X_validation)

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        fold_metrics.append(metrics)

        print(
            f"    Fold {fold_number}: "
            f"RMSE={metrics['RMSE']:.4f}"
        )

    summary = {
        "MAE_mean": mean(
            result["MAE"] for result in fold_metrics
        ),
        "RMSE_mean": mean(
            result["RMSE"] for result in fold_metrics
        ),
        "RMSE_std": stdev(
            result["RMSE"] for result in fold_metrics
        ),
        "R2_mean": mean(
            result["R2"] for result in fold_metrics
        ),
    }

    return summary


def main():
    dataframe = load_data()

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    # Reserve the final test set before cross-validation.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("Development rows:", X_development.shape[0])
    print("Untouched test rows:", X_test.shape[0])

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

    all_results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}")

        summary = evaluate_with_cross_validation(
            model,
            X_development,
            y_development,
        )

        all_results[model_name] = summary

        print(f"    Mean MAE: {summary['MAE_mean']:.4f}")
        print(f"    Mean RMSE: {summary['RMSE_mean']:.4f}")
        print(f"    RMSE standard deviation: {summary['RMSE_std']:.4f}")
        print(f"    Mean R2: {summary['R2_mean']:.4f}")

    best_model_name = min(
        all_results,
        key=lambda name: all_results[name]["RMSE_mean"],
    )

    print("\nCross-validation completed.")
    print("Best model:", best_model_name)
    print(
        "Best mean RMSE:",
        f"{all_results[best_model_name]['RMSE_mean']:.4f}",
    )
    print("The final test set remains untouched.")


if __name__ == "__main__":
    main()