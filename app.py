from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)


MODEL_PATH = Path("linear_regression_model.joblib")


@st.cache_resource
def load_model():
    """Load the trained model from disk with graceful error handling."""

    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            "Model file not found. Please ensure linear_regression_model.joblib is available in the project root."
        )
        st.stop()
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unable to load the model from {MODEL_PATH.name}: {exc}")
        st.stop()


def build_prediction_frame(
    square_feet: int,
    num_rooms: int,
    age: int,
    distance_to_city: int,
) -> pd.DataFrame:
    """Create the input DataFrame in the exact feature order used for training."""

    return pd.DataFrame(
        [[square_feet, num_rooms, age, distance_to_city]],
        columns=[
            "square_feet",
            "num_rooms",
            "age",
            "distance_to_city(km)",
        ],
    )


def predict_price(model, input_frame: pd.DataFrame) -> float:
    """Return the predicted house price as a float."""

    prediction = model.predict(input_frame)
    return float(prediction[0])


def render_model_information() -> None:
    """Show the model details and project background in expandable sections."""

    with st.expander("Model Information"):
        st.write("Algorithm: Linear Regression")
        st.write("R² Score: 0.9351")
        st.write("MAE: 16,543")
        st.write("RMSE: 24,701")
        st.write("Features Used:")
        st.markdown(
            "- Square Feet\n"
            "- Number of Rooms\n"
            "- House Age\n"
            "- Distance to City"
        )

    with st.expander("About Project"):
        st.write(
            "This project was developed as a simple machine learning web application "
            "for predicting house prices. It was built using Python, Streamlit, "
            "Scikit-Learn, and Pandas"
            "demonstrating model deployment with an intuitive user interface."
        )


def main() -> None:
    """Render the Streamlit application."""

    model = load_model()

    st.title("🏠 House Price Prediction")
    st.write(
        "Predict house prices using a trained Linear Regression model with numeric inputs."
    )

    st.sidebar.header("🏠 House Details")
    square_feet = st.sidebar.number_input(
        "Square Feet",
        min_value=500,
        max_value=5000,
        value=2000,
        step=1,
    )
    num_rooms = st.sidebar.number_input(
        "Number of Rooms",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
    )
    age = st.sidebar.number_input(
        "House Age (Years)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
    )
    distance_to_city = st.sidebar.number_input(
        "Distance to City (km)",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
    )

    if st.sidebar.button("Predict Price"):
        try:
            input_frame = build_prediction_frame(
                square_feet=int(square_feet),
                num_rooms=int(num_rooms),
                age=int(age),
                distance_to_city=int(distance_to_city),
            )
            predicted_price = predict_price(model, input_frame)

            st.metric("Estimated House Price", f"${predicted_price:,.2f}")
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"An unexpected error occurred while predicting the price: {exc}")

    render_model_information()

    st.divider()
    st.write("Developed using Streamlit, Scikit-Learn, and Python.")


if __name__ == "__main__":
    main()