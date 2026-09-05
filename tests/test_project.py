import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FOLDER = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_FOLDER))

from load_data import load_data  # noqa: E402


def test_dataset_shape_and_columns():
    dataframe = load_data()

    assert dataframe.shape == (768, 10)
    assert dataframe.isna().sum().sum() == 0
    assert "heating_load" in dataframe.columns
    assert "cooling_load" in dataframe.columns


def test_dataset_has_eight_input_features():
    dataframe = load_data()
    target_columns = {"heating_load", "cooling_load"}
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in target_columns
    ]

    assert len(feature_columns) == 8
