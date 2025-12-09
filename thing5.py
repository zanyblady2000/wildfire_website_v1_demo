import streamlit as st
import joblib
import pandas as pd

# Load the Random Forest model
try:
    # Ensure this file name matches what you uploaded to GitHub
    model = joblib.load('rfc_model.joblib') 
except FileNotFoundError:
    st.error("Model file 'rfc_model.joblib' not found.")
    st.stop()

st.title("Wildfire Predictor Web App (RFC Model)")
st.write("Enter weather conditions to predict the likelihood or size of a wildfire.")

# --- User input fields ---
feature1 = st.number_input("Temperature (°C):", value=0.0)
feature2 = st.number_input("Humidity (%):", value=0.0)
feature3 = st.number_input("Windspeed (km/h):", value=0.0)

# Create a button to make a prediction
if st.button("Predict Wildfire"):
    # *** THIS IS THE CRITICAL LINE TO ENSURE IT'S CORRECT ***
    # The 'columns' list must match your original training data names exactly
    input_data = pd.DataFrame(
        [[feature1, feature2, feature3]], 
        columns=['temp', 'humidity', 'windspeed'] 
    )
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"The model predicted: {prediction[0]}")



