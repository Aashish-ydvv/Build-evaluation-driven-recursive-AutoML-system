from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "extra_trees_final.joblib"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_results"
    / "final_test_predictions.csv"
)

FINAL_RESULTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "final_results"
)

FIGURES_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
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


def check_required_files():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model was not found:\n{MODEL_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions were not found:\n"
            f"{PREDICTIONS_PATH}"
        )


def create_feature_importance_table(model):
    importance_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance_table = importance_table.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_table


def print_error_summary(predictions):
    residual_mean = predictions["residual"].mean()

    median_absolute_error = predictions[
        "absolute_error"
    ].median()

    maximum_absolute_error = predictions[
        "absolute_error"
    ].max()

    print("\nError analysis")
    print(
        "Mean residual:",
        f"{residual_mean:.4f}",
    )
    print(
        "Median absolute error:",
        f"{median_absolute_error:.4f}",
    )
    print(
        "Maximum absolute error:",
        f"{maximum_absolute_error:.4f}",
    )

    print("\nTen largest prediction errors:")

    worst_predictions = predictions.nlargest(
        10,
        "absolute_error",
    )

    columns_to_display = [
        "actual_heating_load",
        "predicted_heating_load",
        "residual",
        "absolute_error",
    ]

    print(
        worst_predictions[
            columns_to_display
        ].round(4)
    )


def create_diagnostic_figure(
    predictions,
    importance_table,
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    minimum_value = min(
        predictions["actual_heating_load"].min(),
        predictions["predicted_heating_load"].min(),
    )

    maximum_value = max(
        predictions["actual_heating_load"].max(),
        predictions["predicted_heating_load"].max(),
    )

    sns.scatterplot(
        data=predictions,
        x="actual_heating_load",
        y="predicted_heating_load",
        ax=axes[0],
    )

    axes[0].plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        color="red",
        linestyle="--",
    )

    axes[0].set_title(
        "Actual vs Predicted Heating Load"
    )

    sns.histplot(
        data=predictions,
        x="residual",
        bins=25,
        kde=True,
        ax=axes[1],
    )

    axes[1].axvline(
        0,
        color="red",
        linestyle="--",
    )

    axes[1].set_title(
        "Residual Distribution"
    )

    sns.barplot(
        data=importance_table,
        x="importance",
        y="feature",
        ax=axes[2],
    )

    axes[2].set_title(
        "Model Feature Importance"
    )

    figure.tight_layout()

    figure_path = (
        FIGURES_FOLDER
        / "final_model_diagnostics.png"
    )

    figure.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return figure_path


def main():
    check_required_files()

    FINAL_RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = joblib.load(MODEL_PATH)

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    importance_table = (
        create_feature_importance_table(model)
    )

    importance_path = (
        FINAL_RESULTS_FOLDER
        / "feature_importance.csv"
    )

    importance_table.to_csv(
        importance_path,
        index=False,
    )

    print("Feature importance")
    print(importance_table.round(4))

    print_error_summary(predictions)

    figure_path = create_diagnostic_figure(
        predictions,
        importance_table,
    )

    print(
        "\nFeature importance saved to:",
        importance_path,
    )

    print(
        "Diagnostic figure saved to:",
        figure_path,
    )

    print(
        "\nImportant: these test results may be "
        "reported, but must not be used for further "
        "model selection or tuning."
    )


if __name__ == "__main__":
    main()