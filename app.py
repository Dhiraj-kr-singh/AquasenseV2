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

# Add CSS for animations and styling
st.markdown(
    """
    <style>
    /* General Styling */
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

    /* Animation Effects */
    .drought {
        background-color: #f4c2c2;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #900;
        animation: fadeIn 2s ease-in-out, shake 0.5s ease-in-out infinite alternate;
    }

    .flood {
        background-color: #c2e0f4;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #0056b3;
        animation: fadeIn 2s ease-in-out, pulsate 1.5s ease-in-out infinite;
    }

    .normal {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #155724;
        animation: fadeIn 2s ease-in-out, scaleUp 1.5s ease-in-out infinite;
    }

    /* Keyframes */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes shake {
        0% {
            transform: translateX(0);
        }
        50% {
            transform: translateX(-10px);
        }
        100% {
            transform: translateX(10px);
        }
    }

    @keyframes pulsate {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
        }
    }

    @keyframes scaleUp {
        0% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title">Flood & Drought Detection v1</div>', unsafe_allow_html=True)

# User input for selecting state
state = st.selectbox("Select State", states, key="state_select", help="Choose a state")

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
            css_class = "drought"
        elif prediction > FLOOD_THRESHOLD:
            status = "Flood"
            css_class = "flood"
        else:
            status = "Normal"
            css_class = "normal"

        # Display the result with animation
        st.markdown(
            f"""
            <div class="{css_class}">
                <h2>Prediction Result</h2>
                <p>The predicted average rainfall in <strong>{state}</strong> for <strong>{year}</strong> is <strong>{prediction:.2f} mm</strong>.</p>
                <p>This is categorized as a '<strong>{status}</strong>' situation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")
