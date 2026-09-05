from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "extra_trees_final.joblib"
)


st.set_page_config(
    page_title="Building Heating Load Predictor",
    page_icon="🏢",
    layout="wide",
)

st.title("Building Heating Load Predictor")

st.write(
    "Enter the architectural characteristics of a building "
    "to estimate its heating load."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. "
        "Run the final model training process first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
st.subheader("Building Design Inputs")

left_column, right_column = st.columns(2)

with left_column:
    relative_compactness = st.selectbox(
        "Relative compactness",
        [0.62, 0.64, 0.66, 0.69, 0.71, 0.74,
         0.76, 0.79, 0.82, 0.86, 0.90, 0.98],
        index=8,
    )

    surface_area = st.selectbox(
        "Surface area",
        [514.5, 563.5, 588.0, 612.5, 637.0, 661.5,
         686.0, 710.5, 735.0, 759.5, 784.0, 808.5],
        index=4,
    )

    wall_area = st.selectbox(
        "Wall area",
        [245.0, 269.5, 294.0, 318.5, 343.0, 367.5, 416.5],
        index=3,
    )

    roof_area = st.selectbox(
        "Roof area",
        [110.25, 122.5, 147.0, 220.5],
        index=2,
    )

with right_column:
    overall_height = st.selectbox(
        "Overall height",
        [3.5, 7.0],
        index=1,
    )

    orientation = st.selectbox(
        "Orientation category",
        [2, 3, 4, 5],
    )

    glazing_area = st.selectbox(
        "Glazing area",
        [0.0, 0.10, 0.25, 0.40],
        index=2,
    )

    glazing_distribution = st.selectbox(
        "Glazing distribution category",
        [0, 1, 2, 3, 4, 5],
        index=1,
    )

input_data = pd.DataFrame(
    [
        {
            "relative_compactness": relative_compactness,
            "surface_area": surface_area,
            "wall_area": wall_area,
            "roof_area": roof_area,
            "overall_height": overall_height,
            "orientation": orientation,
            "glazing_area": glazing_area,
            "glazing_distribution": glazing_distribution,
        }
    ]
)

st.subheader("Selected Input")

st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True,
)

if st.button(
    "Predict Heating Load",
    type="primary",
):
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted heating load: {prediction:.2f}"
    )

    st.caption(
        "This estimate is produced by the tuned Extra Trees "
        "model and should not replace professional "
        "building-energy simulation."
    )