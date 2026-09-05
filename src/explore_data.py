from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "figures"


def create_output_folder():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def print_basic_information(dataframe):
    print("Dataset shape:", dataframe.shape)

    print("\nData types:")
    print(dataframe.dtypes)

    print("\nStatistical summary:")
    print(dataframe.describe().round(2))

    print("\nUnique values:")
    print(dataframe.nunique())


def create_target_histograms(dataframe):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(
        data=dataframe,
        x="heating_load",
        bins=25,
        kde=True,
        ax=axes[0],
    )
    axes[0].set_title("Heating Load Distribution")

    sns.histplot(
        data=dataframe,
        x="cooling_load",
        bins=25,
        kde=True,
        ax=axes[1],
    )
    axes[1].set_title("Cooling Load Distribution")

    figure.tight_layout()

    output_path = OUTPUT_FOLDER / "target_distributions.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print("\nSaved:", output_path)


def create_correlation_heatmap(dataframe):
    correlations = dataframe.corr(numeric_only=True)

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved:", output_path)


def main():
    dataframe = load_data()

    create_output_folder()
    print_basic_information(dataframe)
    create_target_histograms(dataframe)
    create_correlation_heatmap(dataframe)

    print("\nExploratory data analysis completed.")


if __name__ == "__main__":
    main()