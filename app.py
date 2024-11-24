import streamlit as st
import pickle
import pandas as pd

# Load the trained model and label encoder
with open("weather_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder_state.pkl", "rb") as state_file:
    label_encoder_state = pickle.load(state_file)

# Sample data to show available states and years for user selection (Replace with actual data)
states = ["Kolkata", "Meghalaya", "Goa", "Mizoram"]
years = [2023, 2024]  # You can dynamically fetch years from your dataset

# Thresholds for categorization
DROUGHT_THRESHOLD = 74  # Example threshold in mm
FLOOD_THRESHOLD = 120   # Example threshold in mm

# Add custom CSS for styling
def add_custom_css():
    css = """
    /* General Reset */
    body, h1, h2, h3, p {
      margin: 0;
      padding: 0;
    }

    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      color: #333;
      background-color: #f5f5f5;
    }

    /* Navbar */
    .navbar {
      background-color: #007bff;
      color: white;
      padding: 15px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .navbar .logo {
      font-size: 24px;
      font-weight: bold;
    }

    .navbar ul {
      list-style: none;
      display: flex;
    }

    .navbar ul li {
      margin: 0 15px;
    }

    .navbar ul li a {
      text-decoration: none;
      color: white;
      font-weight: bold;
      transition: color 0.3s;
    }

    .navbar ul li a:hover {
      color: #ccc;
    }

    /* Hero Section */
    .hero {
      text-align: center;
      background-color: #e3f2fd;
      padding: 50px 20px;
    }

    .hero h1 {
      font-size: 2.5em;
      color: #007bff;
    }

    .hero p {
      margin: 10px 0 20px;
    }

    .hero input, .hero button {
      padding: 10px 15px;
      margin: 5px;
      border: none;
      border-radius: 5px;
    }

    .hero button {
      background-color: #007bff;
      color: white;
      cursor: pointer;
      transition: background-color 0.3s;
    }

    .hero button:hover {
      background-color: #0056b3;
    }

    /* Dashboard */
    .dashboard {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      padding: 20px;
    }

    .dashboard h2 {
      text-align: center;
      color: #007bff;
    }

    .data-section img, .real-time img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
    }

    /* Community Section */
    .community {
      background-color: white;
      padding: 20px;
      margin: 20px;
      border-radius: 8px;
    }

    .community h2 {
      text-align: center;
      margin-bottom: 20px;
      color: #333;
    }

    .community .forums {
      text-align: center;
    }

    .community .forums button {
      padding: 10px 20px;
      background-color: #007bff;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
    }

    .community .forums button:hover {
      background-color: #0056b3;
    }

    /* Footer */
    footer {
      background-color: #333;
      color: white;
      padding: 15px;
      text-align: center;
    }

    footer .footer-links a {
      color: #007bff;
      text-decoration: none;
    }

    footer .footer-links a:hover {
      text-decoration: underline;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Add custom HTML structure
def add_custom_html():
    html = """
    <header class="navbar">
        <div class="logo">AquaSense</div>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#alerts">Alerts</a></li>
                <li><a href="#solutions">Solutions</a></li>
                <li><a href="#forums">Community Forums</a></li>
                <li><a href="#resources">Educational Resources</a></li>
                <li><a href="#pro">AquaSense Pro</a></li>
            </ul>
        </nav>
    </header>
    <section class="hero">
        <h1>Track Water Levels, Predict Disasters, Conserve Resources</h1>
        <p>Your ultimate platform for water management and climate resilience.</p>
    </section>
    """
    st.markdown(html, unsafe_allow_html=True)

# Prediction functionality
def prediction_section():
    st.title("Flood & Drought Detection v1")
    state = st.selectbox("Select State", states)
    year = st.selectbox("Select Year", years)
    if st.button("Predict Rainfall"):
        try:
            encoded_state = label_encoder_state.transform([state])[0]
            input_data = pd.DataFrame([[encoded_state, int(year)]], columns=["State", "Year"])
            prediction = (model.predict(input_data)[0]) * 10
            status = (
                "Drought" if prediction < DROUGHT_THRESHOLD else
                "Flood" if prediction > FLOOD_THRESHOLD else
                "Normal"
            )
            st.success(f"The predicted average rainfall in {state} for {year} is {prediction:.2f} mm.")
            st.info(f"This is categorized as a '{status}' situation.")
        except Exception as e:
            st.error(f"Error: {e}")

# Main function to render the app
def main():
    add_custom_css()
    add_custom_html()
    prediction_section()

if __name__ == "__main__":
    main()
