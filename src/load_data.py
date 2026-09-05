from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ENB2012_data.xlsx"
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found at:\n{DATA_PATH}"
        )

    dataframe = pd.read_excel(DATA_PATH)

    dataframe = dataframe.dropna(axis=1, how="all")
    dataframe = dataframe.dropna(axis=0, how="all")

    expected_columns = [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "orientation",
        "glazing_area",
        "glazing_distribution",
        "heating_load",
        "cooling_load",
    ]

    if dataframe.shape[1] != len(expected_columns):
        raise ValueError(
            f"Expected {len(expected_columns)} columns, "
            f"but found {dataframe.shape[1]}."
        )

    dataframe.columns = expected_columns

    return dataframe


def main():
    dataframe = load_data()

    print("Dataset loaded successfully.")
    print("Dataset location:", DATA_PATH)
    print("Number of rows:", dataframe.shape[0])
    print("Number of columns:", dataframe.shape[1])

    print("\nFirst five rows:")
    print(dataframe.head())

    print("\nMissing values:")
    print(dataframe.isna().sum())

    print("\nDuplicate rows:")
    print(dataframe.duplicated().sum())


if __name__ == "__main__":
    main()
