import streamlit as st
import requests

# Page setup with agricultural theme
st.set_page_config(
    page_title="🌱 Agritech Predictor",
    layout="centered",
    page_icon="🌾"
)

# Custom CSS for agricultural colors and fonts
st.markdown("""
    <style>
    body {
        font-family: 'Trebuchet MS', sans-serif;
        background-color: #f4f9f4;
        color: #2e4a2e;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border: 1px solid #4CAF50;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title(" Agritech Moisture predictor")
st.markdown("""
This tool helps farmers and agronomists make **data-driven decisions**.  
Please provide the following sensor data: 
""")

# Input form
date = st.text_input("Date (YYYY-MM-DD)", value="2026-04-09")
type_value = st.text_input("Type (e.g., Maize, Beans)")
fertilizer = st.number_input("Fertilizer_Recommended (kg/ha)", min_value=0.0, format="%.2f")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f")
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, format="%.2f")
soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, format="%.2f")

# Feature list (numerical only for ONNX model)
features_list = [ fertilizer, humidity, temperature, soil_ph]

# Prediction button
if st.button(" Predict Recommendation"):
    try:
        response = requests.post("http://localhost:8000/predict", json={"features": features_list})
        st.success(f"Predicted Output: {response.json()['prediction']:.4f}")
    except Exception as e:
        st.error("Backend not reachable. Please ensure FastAPI is running.")
