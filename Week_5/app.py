import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained Random Forest model
model = joblib.load("rf_sales_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Sales Prediction App", layout="centered")
st.title("📊 Sales Prediction")
st.markdown("Enter the advertising budget below to predict Sales ⬇️")

# Input fields
tv = st.number_input("📺 TV Advertising Budget", min_value=0.0, format="%.2f")
radio = st.number_input("📻 Radio Advertising Budget", min_value=0.0, format="%.2f")
newspaper = st.number_input("🗞️ Newspaper Advertising Budget", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Sales"):
    input_data = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Sales: **{prediction:.2f} units**")

# Footer
st.markdown("---")
st.markdown("Made by Maryam Sameen")
