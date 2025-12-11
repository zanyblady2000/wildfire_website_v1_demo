# app.py

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# --- Load Assets ---
# Load the pre-trained model and the pre-fitted scaler from the local directory
# New code with correct filename
try:
    rfc = joblib.load('rfc_model.pkl')
    # Change the string literal below to match your actual filename:
    scaler = joblib.load('scaler (1).pkl') 
# ...
except FileNotFoundError:
    st.error("Error: Ensure 'rfc_model.pkl' and 'scaler.pkl' are in the same folder as app.py.")
    st.stop()

st.title("Weather Prediction App (RFC Model)")
st.sidebar.header("Input Weather Conditions")

# --- User Input Function ---
def user_input_features():
    # Define interactive input widgets for humidity, temp, and windspeed
    # These sliders appear in the sidebar for a clean interface
    temp = st.sidebar.slider('temp', 0.0, 100.0, 50.0)
    humidity = st.sidebar.slider('humidity', -10.0, 40.0, 20.0)
    windspeed = st.sidebar.slider('windspeed', 0.0, 50.0, 15.0)
    lat = st.sidebar.slider('lat', 0.0, 59.0, 50.0)
    long = st.sidebar.slider('long', 0.0, -113.0, -124.0)

    # Create a DataFrame from the inputs with correct feature names
    data = {'temp': temp,
            'humidity': humidity,
            'windspeed': windspeed,
            'lat': lat,
            'long': long}
    # index= is necessary for pandas to correctly handle single-row input
    features_df = pd.DataFrame(data, index=[0])

    return features_df

# --- Main App Logic ---
prediction_data = raw_input_data[['humidity', 'temp', 'windspeed']]

st.subheader('User Input Features (Raw)')
st.write(raw_input_df)

# Button to trigger the prediction
if st.button('Predict Outcome'):
    # Apply the SAME scaling used during training to the NEW user input
    # The scaler outputs a numpy array
    scaled_input_array = scaler.transform(prediction_data)

    # Make the prediction using the loaded 'rfc' model on the scaled data
    prediction = rfc.predict(scaled_input_array)
    
    # Optional: Get prediction probabilities if using a Classifier
    # prediction_proba = rfc.predict_proba(scaled_input_array)

    st.subheader('Prediction Result')
    # Display the result (assuming a simple classification outcome)
    predicted = {prediction[0]}
    st.success(f"The model predicts: {predicted}") 

    map_df = raw_input_df[['lat', 'long', 'predicted']]

    fig = px.scatter_mapbox(map_df, lat='lat', lon='long', color='fire_occurrence',
                        color_discrete_map={'High': 'red', 'Low': 'green'},
                        zoom=3, height=500)
    fig.update_layout(mapbox_style='open-street-map')
    fig.show



