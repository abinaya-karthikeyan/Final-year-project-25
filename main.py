import streamlit as st
import numpy as np
import joblib
import pandas as pd

MODEL_PATH = 'artifacts/model_xgb.joblib'
SCALER_PATH = 'artifacts/scaler_rest.joblib'
COLUMNS_PATH = 'artifacts/columns.pkl'

# Load model, scaler, and training columns
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(COLUMNS_PATH, 'rb') as f:
    X_columns = joblib.load(f)

# Feature input UI
st.set_page_config(page_title="House Price Estimator")
st.title("🏠House Price Predictor")

# Input options
locations = [col for col in X_columns if col not in ['BHK', 'total_sqft', 'bath', 'balcony', 'availability']]
selected_location = st.selectbox("📍 Select Location", sorted(locations))

total_sqft = st.number_input("Total Area (in sqft)", min_value=500, max_value=10000, step=50)
bhk = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5, 6])
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5, 6])
balcony = st.selectbox("Number of Balconies", [0, 1, 2])

# Predict button
if st.button("Predict Price"):
    # Create input vector
    # Create input vector
    x = np.zeros(len(X_columns))

    # Extract numeric inputs


    numeric_df = pd.DataFrame([[bhk, total_sqft, bath, balcony]],
                              columns=['BHK', 'total_sqft', 'bath', 'balcony'])
    scaled_numeric = scaler['scaler'].transform(numeric_df)

    # Assign scaled numeric values into correct positions in x
    x[X_columns.index('BHK')] = scaled_numeric[0][0]
    x[X_columns.index('total_sqft')] = scaled_numeric[0][1]
    x[X_columns.index('bath')] = scaled_numeric[0][2]
    x[X_columns.index('balcony')] = scaled_numeric[0][3]

    # Set location one-hot encoding
    if selected_location in X_columns:
        x[X_columns.index(selected_location)] = 1

    # Predict
    predicted_price = model.predict([x])[0]

    st.success(f"💰 Predicted Price: ₹{predicted_price:.2f} Lakhs")

