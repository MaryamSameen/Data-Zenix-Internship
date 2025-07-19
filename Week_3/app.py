import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")

# Define title
st.title("🚗 Car Price Prediction App")

# Input Fields
car_name = st.selectbox("Car Name", ["ritz", "sx4", "ciaz", "wagon r", "swift"])  # Same as label-encoded values
year = st.slider("Year of Purchase", 2000, 2024, 2015)
present_price = st.number_input("Present Price (in lakhs)", value=5.0)
kms_driven = st.number_input("Kilometers Driven", value=30000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
selling_type = st.selectbox("Selling Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 2, 3])

# Encode categorical fields (manual encoding to match training)
fuel_type_encoded = {'Petrol': 2, 'Diesel': 0, 'CNG': 1}[fuel_type]
selling_type_encoded = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}[selling_type]
transmission_encoded = {'Manual': 1, 'Automatic': 0}[transmission]
car_name_encoded = {'ritz': 4, 'sx4': 6, 'ciaz': 0, 'wagon r': 8, 'swift': 7}[car_name]  # Example encoding

# Final input data
input_data = np.array([[car_name_encoded, year, present_price, kms_driven,
                        fuel_type_encoded, selling_type_encoded,
                        transmission_encoded, owner]])

# Predict
if st.button("Predict Selling Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
