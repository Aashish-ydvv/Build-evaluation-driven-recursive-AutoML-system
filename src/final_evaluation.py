import argparse
import json
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from load_data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)

FINAL_RESULTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "final_results"
)

MODELS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

AGGREGATED_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "aggregated_results.json"
)

NEXT_EXPERIMENT_PATH = (
    EXPERIMENTS_FOLDER
    / "next_experiment.json"
)

FINAL_METRICS_PATH = (
    FINAL_RESULTS_FOLDER
    / "final_test_metrics.json"
)

PREDICTIONS_PATH = (
    FINAL_RESULTS_FOLDER
    / "final_test_predictions.csv"
)

MODEL_PATH = (
    MODELS_FOLDER
    / "extra_trees_final.joblib"
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


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    mse = mean_squared_error(
        actual,
        predicted,
    )

    rmse = sqrt(mse)

    r2 = r2_score(
        actual,
        predicted,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Perform the one-time final test evaluation."
        )
    )

    parser.add_argument(
        "--approve-final-test",
        action="store_true",
        help=(
            "Confirm that model selection is complete "
            "and authorize final test evaluation."
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not arguments.approve_final_test:
        raise SystemExit(
            "Final test evaluation was not approved.\n"
            "Run again with --approve-final-test only "
            "after model selection is complete."
        )

    if FINAL_METRICS_PATH.exists():
        raise SystemExit(
            "Final test metrics already exist.\n"
            "The protected test set must not be "
            "evaluated repeatedly."
        )

    aggregated_results = load_json_file(
        AGGREGATED_RESULTS_PATH
    )

    next_experiment = load_json_file(
        NEXT_EXPERIMENT_PATH
    )

    if (
        next_experiment["experiment_type"]
        != "stop_and_review"
    ):
        raise ValueError(
            "The experiment system has not reached "
            "the stop-and-review stage."
        )

    current_champion = aggregated_results[
        "current_champion"
    ]

    if not current_champion.startswith(
        "Extra Trees"
    ):
        raise ValueError(
            "The final evaluator currently supports "
            "only the Extra Trees champion."
        )

    if aggregated_results["protected_test_used"]:
        raise ValueError(
            "The aggregated state says that the "
            "protected test set has already been used."
        )

    champion_parameters = aggregated_results[
        "current_champion_parameters"
    ].copy()

    dataframe = load_data()

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nFinal evaluation started.")
    print("Champion:", current_champion)
    print("Development rows:", len(X_development))
    print("Protected test rows:", len(X_test))
    print("Parameters:", champion_parameters)

    final_model = ExtraTreesRegressor(
        **champion_parameters,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_development,
        y_development,
    )

    final_predictions = final_model.predict(
        X_test
    )

    final_metrics = calculate_metrics(
        y_test,
        final_predictions,
    )

    FINAL_RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    predictions_table = X_test.copy()

    predictions_table["actual_heating_load"] = (
        y_test
    )

    predictions_table["predicted_heating_load"] = (
        final_predictions
    )

    predictions_table["residual"] = (
        predictions_table["actual_heating_load"]
        - predictions_table["predicted_heating_load"]
    )

    predictions_table["absolute_error"] = (
        predictions_table["residual"].abs()
    )

    predictions_table.to_csv(
        PREDICTIONS_PATH,
        index=True,
        index_label="original_row_index",
    )

    final_output = {
        "evaluation_type": (
            "one_time_protected_test_evaluation"
        ),
        "champion": current_champion,
        "champion_parameters": (
            champion_parameters
        ),
        "cross_validation_rmse": (
            aggregated_results[
                "current_champion_rmse"
            ]
        ),
        "development_rows": len(X_development),
        "test_rows": len(X_test),
        "test_evaluation_count": 1,
        "metrics": final_metrics,
        "model_path": str(MODEL_PATH),
        "predictions_path": str(
            PREDICTIONS_PATH
        ),
    }

    save_json_file(
        FINAL_METRICS_PATH,
        final_output,
    )

    aggregated_results[
        "protected_test_used"
    ] = True

    aggregated_results[
        "final_test_evaluations"
    ] = 1

    aggregated_results[
        "final_test_metrics"
    ] = final_metrics

    aggregated_results[
        "final_model_path"
    ] = str(MODEL_PATH)

    save_json_file(
        AGGREGATED_RESULTS_PATH,
        aggregated_results,
    )

    print("\nFinal test evaluation completed.")
    print(
        "MAE:",
        f"{final_metrics['MAE']:.4f}",
    )
    print(
        "RMSE:",
        f"{final_metrics['RMSE']:.4f}",
    )
    print(
        "R2:",
        f"{final_metrics['R2']:.4f}",
    )
    print("Model saved to:", MODEL_PATH)
    print(
        "Predictions saved to:",
        PREDICTIONS_PATH,
    )
    print(
        "Metrics saved to:",
        FINAL_METRICS_PATH,
    )
    print(
        "Protected test set is now marked as used."
    )


if __name__ == "__main__":
    main()