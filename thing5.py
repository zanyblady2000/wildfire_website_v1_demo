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

# 1. Get raw input and set up initial map data (Always runs)
raw_input_df = user_input_features() 
map_data = raw_input_df.copy()

# Initialize the 'risk_level' column with a placeholder value 
# This ensures Plotly has a non-null value to work with before prediction.
map_data['risk_level'] = 'Not Predicted'

st.subheader('User Input Features (Raw)')
st.write(raw_input_df)


# 2. Prediction and update logic (Only runs when the button is clicked)
if st.button('Predict Outcome'):
    
    # Prepare data for model prediction
    prediction_data = raw_input_df[['temp', 'humidity', 'windspeed']]
    scaled_input_array = scaler.transform(prediction_data)
    prediction = rfc.predict(scaled_input_array)
    predicted_value = prediction[0] # Extract the scalar value

    # Update the 'risk_level' in the map data using the key mapping
    # This is where your specific map function is applied to the predicted value
    map_data['prediction_value'] = predicted_value
    map_data['risk_level'] = map_data['prediction_value'].map(PREDICTION_MAPPING)

    st.subheader('Prediction Result')
    st.success(f"The model predicts: {map_data['risk_level'].iloc[0]} Risk (Code: {predicted_value})")


# 3. Create and Display the Plotly map (Always runs, using the current state of map_data)
# We include 'Not Predicted' in the color map to handle the initial state
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














