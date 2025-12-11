# thing5.py

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# --- Load Assets ---
# Ensure these files are in the same folder and the names match EXACTLY
try:
    rfc = joblib.load('rfc_model.pkl')
    # Using the filename provided in previous conversation: 'scaler (1).pkl'
    scaler = joblib.load('scaler (1).pkl') 

except FileNotFoundError:
    st.error("Error: Ensure 'rfc_model.pkl' and 'scaler (1).pkl' are in the same folder.")
    st.stop()

st.title("Weather Prediction App (RFC Model)")
st.sidebar.header("Input Weather Conditions")

# --- User Input Function ---
def user_input_features():
    # Collect all inputs
    temp = st.sidebar.slider('Temperature (°C)', -10.0, 40.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider('Windspeed (km/h)', 0.0, 50.0, 15.0)
    
    # These are for display only, not model prediction factors
    lat = st.sidebar.slider('Latitude', 0.0, 59.0, 50.0)
    long = st.sidebar.slider('Longitude', -180.0, 180.0, -100.0) 

    # Create a DataFrame containing ALL data for display purposes
    data = {'temp': temp,
            'humidity': humidity,
            'windspeed': windspeed,
            'lat': lat,
            'long': long}
            
    features_df = pd.DataFrame(data, index=[0])
    return features_df

# --- Main App Logic ---

# Define the raw input DataFrame (this fixes NameError: raw_input_df not defined)
raw_input_df = user_input_features() 

st.subheader('User Input Features (Raw)')
st.write(raw_input_df)

predicted = {prediction[0]}

prediction = rfc.predict(scaled_input_array)

if st.button('Predict Outcome'):
    # *** PASTE THE NEW CODE HERE ***

    # Create a DataFrame for prediction with ONLY the 3 trained features:
    prediction_data = raw_input_df[['temp', 'humidity', 'windspeed']]
    
    # Scale ONLY the prediction data
    scaled_input_array = scaler.transform(prediction_data)

    # Make the prediction
    prediction = rfc.predict(scaled_input_array)
    
    # Extract the single value from the NumPy array:
    predicted_value = prediction
    
    st.subheader('Prediction Result')
    # Use the single value for the success message
    st.success(f"The model predicts: {predicted_value}")
    mapping_df = raw_input_df.copy()
    
    # 2. Add the single predicted value as a new column named 'predicted'
    mapping_df['predicted'] = predicted_value 
    
    # 3. Now select the desired columns.
    mapping_df = mapping_df[['lat', 'long', 'predicted']]

    fig = px.scatter_mapbox(mapping_df, lat='lat', lon='long',
                        color_discrete_map={'High': 'red', 'Low': 'green'},
                        zoom=3, height=500)
    fig.update_layout(mapbox_style='open-street-map')
    fig.show()
                    















