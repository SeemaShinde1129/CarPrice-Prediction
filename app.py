import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the trained model
model = pk.load(open('model.pkl', 'rb'))

# App title
st.header('Car Price Prediction Model')

# Load and preprocess car data
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit UI elements
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('Number of Kilometers', 1, 200000)
fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (km/l)', 10, 40)
engine = st.slider('Engine Capacity (cc)', 700, 5000)
max_power = st.slider('Maximum Power (bhp)', 0, 200)
seats = st.slider('Number of Seats', 5, 10)

# Encoding mappings
owner_map = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 
             'Fourth & Above Owner': 4, 'Test Drive Car': 5}
fuel_map = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
trans_map = {'Manual': 1, 'Automatic': 2}
brand_map = {brand: idx + 1 for idx, brand in enumerate(cars_data['name'].unique())}

# Prediction logic
if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Apply mappings to encode categorical values
    input_data_model['owner'] = input_data_model['owner'].map(owner_map)
    input_data_model['fuel'] = input_data_model['fuel'].map(fuel_map)
    input_data_model['seller_type'] = input_data_model['seller_type'].map(seller_map)
    input_data_model['transmission'] = input_data_model['transmission'].map(trans_map)
    input_data_model['name'] = input_data_model['name'].map(brand_map)
    
    # Handle unmapped values
    if input_data_model.isnull().any().any():
        st.error("Some values could not be encoded. Please check your inputs.")
    else:
        car_price = model.predict(input_data_model)
        st.markdown(f'### Predicted Car Price: ₹ {round(car_price[0], 2)}')
