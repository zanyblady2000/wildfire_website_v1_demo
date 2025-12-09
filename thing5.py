import streamlit as st
import joblib
import pandas as pd

# Load the Random Forest model (ensure the filename matches the one you downloaded)
try:
    model = joblib.load('rfc_model.joblib')
except FileNotFoundError:
    st.error("Model file 'rfc_model.joblib' not found. Please check your file names.")
    st.stop()

st.title("RFC Model Prediction Web App")
st.write("Enter the features below to get a prediction from the Random Forest Classifier.")

# --- Example of user input fields (customize these for your specific model features) ---
# Replace 'Feature 1', 'Feature 2', etc., with your actual feature names
feature1 = st.number_input("Input for Feature 1 ( temp):", value=0.0)
feature2 = st.number_input("Input for Feature 2 ( humidity):", value=0.0)
feature3 = st.number_input("Input For Feature 3 (windspeed):", value=0.0)

# Create a button to make a prediction
if st.button("Predict"):
    # Prepare input data into a format your model expects (usually a DataFrame or array)
    input_data = pd.DataFrame([[feature1, feature2, feature3]], columns=['Feature 1', 'Feature 2', 'Feature 3'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.success(f"The model predicted: {prediction[0]}")

