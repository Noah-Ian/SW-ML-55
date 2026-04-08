import streamlit as st
import requests

st.title("Gradient Boosting Predictor")

features = st.text_input("Enter features (comma-separated):")

if st.button("Predict"):
    features_list = [float(x) for x in features.split(",")]
    response = requests.post("http://localhost:8000/predict", json={"features": features_list})
    st.write("Prediction:", response.json()["prediction"])
