import streamlit as st
import joblib
import pandas as pd
import plotly.express as px 

rfc = joblib.load('rfc_model.pkl')
scaler = joblib.load('scaler (1).pkl') 

st.title("Weather Prediction App (RFC Model)")
st.sidebar.header("Input Weather Conditions")

def user_input_features():
    temp = st.sidebar.slider('Temperature (°C)', -10.0, 40.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider('Windspeed (km/h)', 0.0, 50.0, 15.0)
    lat = st.sidebar.slider('Latitude', 0.0, 59.0, 50.0)
    long = st.sidebar.slider('Longitude', -180.0, 180.0, -100.0) 

    data = {'temp': temp, 'humidity': humidity, 'windspeed': windspeed, 'lat': lat, 'long': long}
            
    features_df = pd.DataFrame(data, index=[0])
    return features_df

raw_input_df = user_input_features() 

st.subheader('User Input Features')
st.write(raw_input_df)

if st.button('Predict Outcome'):
    
    prediction_data = raw_input_df[['temp', 'humidity', 'windspeed']]
    scaled_input_data = scaler.transform(prediction_data)
    prediction = rfc.predict(scaled_input_data)
    predicted_value = prediction
    
    st.subheader('Prediction Result')
    st.success(f"The model predicts: {predicted_value}")













