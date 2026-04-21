
import streamlit as st
import pickle
import numpy as np

# === Load Model and Scaler ===
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# === Mappings ===
brand_map = {
    'Ford': 0, 'Hyundai': 1, 'Lexus': 2, 'INFINITI': 3, 'Audi': 4, 'Acura': 5, 'BMW': 6, 'Tesla': 7, 
    'Land': 8, 'Aston': 9, 'Toyota': 10, 'Lincoln': 11, 'Jaguar': 12, 'Mercedes-Benz': 13, 'Dodge': 14, 'Nissan': 15, 
    'Genesis': 16, 'Chevrolet': 17, 'Kia': 18, 'Jeep': 19, 'Bentley': 20, 'Honda': 21, 'Lucid': 22, 'MINI': 23, 'Porsche': 24, 
    'Hummer': 25, 'Chrysler': 26, 'Volvo': 27, 'Cadillac': 28, 'Lamborghini': 29, 'Maserati': 30, 'Volkswagen': 31, 'Subaru': 32, 
    'Rivian': 33, 'GMC': 34, 'RAM': 35, 'Alfa': 36, 'Ferrari': 37, 'Scion': 38, 'Mitsubishi': 39, 'Mazda': 40, 'Saturn': 41, 
    'Bugatti': 42, 'Polestar': 43, 'Rolls-Royce': 44, 'McLaren': 45, 'Buick': 46, 'Lotus': 47, 'Pontiac': 48, 'FIAT': 49, 
    'Karma': 50, 'Saab': 51, 'Mercury': 52, 'Plymouth': 53, 'smart': 54, 'Maybach': 55, 'Suzuki': 56
}

fuel_map = {
    'Gasoline': 0,
    'Hybrid': 1,
    'E85 Flex Fuel': 2,
    'Diesel': 3,
    'Plug-In Hybrid': 4,
    'Other': 5
}

transmission_map = {
    'A/T': 0,
    '8-Speed A/T': 1,
    'Dual Shift Mode': 2,
    '6-Speed A/T': 3,
    '6-Speed M/T': 4,
    'Other': 5
}

# === UI ===
st.title("Used Car Price Category Predictor")
st.subheader("Enter car details below:")

brand = st.selectbox("Brand", list(brand_map.keys()))
model_year = st.number_input("Model Year", min_value=1970, max_value=2024, value=2020)
milage = st.number_input("Mileage (in miles)", min_value=0, step=1000)
fuel_type = st.selectbox("Fuel Type", list(fuel_map.keys()))
transmission = st.selectbox("Transmission", list(transmission_map.keys()))
clean_title = st.radio("Clean Title?", ['Yes', 'No'])

# === Predict ===
if st.button("Predict Price Class"):
    user_input = [[
        brand_map[brand], model_year, milage,
        fuel_map[fuel_type], transmission_map[transmission],
        1 if clean_title == 'Yes' else 0
    ]]
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]

    # Mapping prediction number to text label
    prediction_label = {0: 'High', 1: 'Medium', 2: 'Low'}[prediction]

    st.success(f"Predicted Price Category: **{prediction_label}**")
