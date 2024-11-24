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

# Embed custom CSS for styling
def add_custom_css():
    with open("styles.css", "r") as css_file:
        custom_css = css_file.read()
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Custom header HTML
def render_custom_header():
    st.markdown(
        """
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
        """, unsafe_allow_html=True
    )

# Custom hero section
def render_hero_section():
    st.markdown(
        """
        <section class="hero">
            <h1>Track Water Levels, Predict Disasters, Conserve Resources</h1>
            <p>Your ultimate platform for water management and climate resilience.</p>
            <input type="text" id="search-input" placeholder="Enter ZIP code or city name">
            <button id="search-button">Check Water Level</button>
        </section>
        """, unsafe_allow_html=True
    )

# Custom dashboard section
def render_dashboard():
    st.markdown(
        """
        <section class="dashboard">
            <div class="real-time">
                <h2>Real-Time Water Level</h2>
                <div class="water-status">
                    <p id="water-level">Water Level: Normal</p>
                    <img id="map-img" src="assets/default-map.png" alt="Real-Time Water Level Map">
                </div>
            </div>
            <div class="data-section">
                <div id="historical-data">
                    <h2>Historical Data</h2>
                    <img id="historical-data-img" src="assets/default-historical.png" alt="Default Historical Data">
                </div>
                <div id="heat-maps">
                    <h2>Heat Maps</h2>
                    <img id="heat-maps-img" src="assets/default-heatmap.png" alt="Default Heat Map">
                </div>
            </div>
        </section>
        """, unsafe_allow_html=True
    )

# Custom community section
def render_community_section():
    st.markdown(
        """
        <section class="community">
            <h2>Community Engagement</h2>
            <div class="forums">
                <p>Join discussions about innovative water conservation methods, climate impact on resources, and more.</p>
                <button class="learn-more">Learn More</button>
            </div>
            <div class="actions">
                <h3>Get Involved</h3>
                <p>Join our efforts in promoting sustainable water practices by volunteering or sharing data with our partner organizations.</p>
            </div>
        </section>
        """, unsafe_allow_html=True
    )

# Custom footer
def render_footer():
    st.markdown(
        """
        <footer>
            <p>&copy; 2024 AquaSense Pro. All Rights Reserved.</p>
            <div class="footer-links">
                <a href="#privacy">Privacy Policy</a> | 
                <a href="#terms">Terms of Service</a> | 
                <a href="#contact">Contact Us</a>
            </div>
        </footer>
        """, unsafe_allow_html=True
    )

# Main function to render the app
def main():
    add_custom_css()
    render_custom_header()
    render_hero_section()
    render_dashboard()
    render_community_section()
    render_footer()

    # Prediction functionality
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

if __name__ == "__main__":
    main()
