import streamlit as st
import joblib
import pandas as pd
import plotly.express as px 
import numpy as np

# --- Load Assets ---
# NOTE: These files must exist in your environment for the app to run.
try:
    rfc = joblib.load('rfc_model.pkl')
    scaler = joblib.load('scaler (1).pkl') 
except FileNotFoundError:
    st.error("Error: Ensure 'rfc_model.pkl' and 'scaler (1).pkl' are available.")
    st.stop()

st.title("Weather Prediction App (RFC Model)")
st.sidebar.header("Input Weather Conditions")

# --- User Input Function ---
def user_input_features():
    """Collects user inputs via Streamlit sidebar sliders."""
    temp = st.sidebar.slider('Temperature (°C)', -10.0, 40.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider('Windspeed (km/h)', 0.0, 50.0, 15.0)
    lat = st.sidebar.slider('Latitude', 0.0, 59.0, 50.0)
    long = st.sidebar.slider('Longitude', -180.0, 180.0, -100.0) 

    data = {'temp': temp, 'humidity': humidity, 'windspeed': windspeed,
            'lat': lat, 'long': long}
            
    features_df = pd.DataFrame(data, index=[0]) 
    return features_df

# Define the numerical to descriptive mapping
PREDICTION_MAPPING = {0: 'Low', 1: 'High'}

# --- Main App Logic ---

# 0. Initialize Session State for persistence across reruns (e.g., slider changes)
# The prediction result is stored here.
if 'predicted_risk' not in st.session_state:
    st.session_state.predicted_risk = 'Not Predicted'


# 1. Get raw input and set up initial map data (Always runs)
raw_input_df = user_input_features() 
map_data = raw_input_df.copy()

# The 'risk_level' column is always populated from the persistent session state.
# This ensures the correct hover data is used even when the script reruns 
# due to slider manipulation.
map_data['risk_level'] = st.session_state.predicted_risk 

st.subheader('User Input Features (Raw)')
st.write(raw_input_df)


# 2. Prediction and update logic (Only runs when the button is clicked)
if st.button('Predict Outcome'):
    
    # Prepare data for model prediction
    prediction_data = raw_input_df[['temp', 'humidity', 'windspeed']]
    scaled_input_array = scaler.transform(prediction_data)
    prediction = rfc.predict(scaled_input_array)
    predicted_value = prediction[0] # Extract the scalar value

    # Calculate the descriptive risk level
    descriptive_risk = PREDICTION_MAPPING.get(predicted_value, 'Unknown')

    # Store the new prediction result in session state
    st.session_state.predicted_risk = descriptive_risk
    
    # Update the map_data for the *current* run
    map_data['risk_level'] = descriptive_risk

    st.subheader('Prediction Result')
    st.success(f"The model predicts: {descriptive_risk} Risk (Code: {predicted_value})")


# 3. Create and Display the Plotly map (Always runs)
# The map uses map_data, which is now consistently populated by st.session_state.
fig = px.scatter_mapbox(
    map_data, # Pass the prepared DataFrame here
    lat="lat", 
    lon="long", 
    color="risk_level",  
    color_discrete_map={'High': 'red', 'Low': 'green', 'Not Predicted': 'grey'},
    zoom=5,             
    height=400,
    mapbox_style="carto-positron", 
    hover_data=['temp', 'humidity', 'windspeed', 'risk_level'] 
)

st.plotly_chart(fig, use_container_width=True)














