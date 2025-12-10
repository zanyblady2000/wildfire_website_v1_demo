import streamlit as st
import joblib
import pandas as pd
# Import the specific class used for the scaler to help joblib load it correctly
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier # Import model class too

# Load the Random Forest model and the scaler object
try:
    # Ensure these filenames match exactly what is in your GitHub repo
    model = joblib.load('rfc_model.joblib')
    scaler = joblib.load('scaler.joblib') # Load the MinMax Scaler object
except FileNotFoundError:
    st.error("One or more required files (model or scaler) were not found in the repository.")
    st.stop()

st.title("Wildfire Predictor Web App (RFC Model)")
st.write("Enter weather conditions to predict the likelihood or size of a wildfire.")

# --- User input fields ---
feature1 = st.number_input("Temperature (°C):", value=0.0)
feature2 = st.number_input("Humidity (%):", value=0.0)
feature3 = st.number_input("Windspeed (km/h):", value=0.0)

if st.button("Predict Wildfire"):
    # 1. Put input data into a DataFrame (must match column names exactly!)
    input_data = pd.DataFrame(
        [[feature1, feature2, feature3]], 
        columns=['temp', 'humidity', 'windspeed'] 
    )
    
    # 2. CRITICAL EDIT: Transform the input data using the loaded scaler
    # The model expects data scaled between 0 and 1 because you used MinMaxScaler
    scaled_input_data = scaler.transform(input_data)

    # 3. Make prediction using the SCALED data
    prediction = model.predict(scaled_input_data)
    
    # Display the result
    st.success(f"The model predicted class: {prediction[0]}")










