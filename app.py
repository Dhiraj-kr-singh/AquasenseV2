import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and label encoder
with open("weather_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder_state.pkl", "rb") as state_file:
    label_encoder_state = pickle.load(state_file)

# Sample data to show available states and years for user selection (Replace with actual data)
states = ["Kolkata", "Meghalaya", "Goa", "Mizoram"]
years = [2023, 2024]  # You can dynamically fetch years from your dataset

# Thresholds for categorization (example values, replace with actual thresholds)
DROUGHT_THRESHOLD = 74  # mm (example)
FLOOD_THRESHOLD = 120  # mm (example)

# Add a little CSS for smoother styling
st.markdown(
    """
    <style>
    .title {
        font-size: 2.5em;
        color: #007bff;
        text-align: center;
        margin-top: 20px;
        font-family: Arial, sans-serif;
    }
    .select-box {
        margin: 10px 0 20px;
        font-size: 1.2em;
        color: #333;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 8px;
    }
    .button {
        background-color: #007bff;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        text-align: center;
    }
    .button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title">Flood & Drought Detection v1</div>', unsafe_allow_html=True)

# User input for selecting state
state = st.selectbox("Select State", states, key="state_select", help="Choose a state", format_func=lambda x: f"🌍 {x}")

# User input for selecting year
year = st.selectbox("Select Year", years, key="year_select", help="Pick a year to forecast")

# Predict rainfall
if st.button("Predict Rainfall", help="Click to predict rainfall"):
    try:
        # Encode the state input
        encoded_state = label_encoder_state.transform([state])[0]

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[encoded_state, int(year)]], columns=["State", "Year"])

        # Mock prediction for demo purposes
        prediction = (model.predict(input_data)[0]) * 10

        # Determine rainfall category
        if prediction < DROUGHT_THRESHOLD:
            status = "Drought"
        elif prediction > FLOOD_THRESHOLD:
            status = "Flood"
        else:
            status = "Normal"

        # Display the result
        st.success(f"The predicted average rainfall in {state} for {year} is {prediction:.2f} mm.")
        st.info(f"This is categorized as a '{status}' situation.")

    except Exception as e:
        st.error(f"Error: {e}")
