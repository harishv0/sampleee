# streamlit_taxi_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Load trained model ---
model_path = "taxi_rf_model_joblib.pkl"
model = joblib.load(model_path)

# --- App title ---
st.title("🚖 NYC Taxi Trip Price Predictor")
st.write("Enter the trip details below to predict the trip price:")

# --- Load CSV for structure (optional, used for dropdowns & ranges) ---
df = pd.read_csv(r"taxi_trip_pricingR.csv")

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include='number').columns.drop('Trip_Price')
cat_cols = df.select_dtypes(include='object').columns

# --- Sidebar inputs ---
st.sidebar.header("Trip Details")
user_input = {}

# Numeric inputs
for col in num_cols:
    min_val = int(df[col].min())
    max_val = int(df[col].max())
    mean_val = int(df[col].mean())
    user_input[col] = st.sidebar.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Categorical inputs
for col in cat_cols:
    options = df[col].unique().tolist()
    user_input[col] = st.sidebar.selectbox(f"{col}", options)

# --- Convert input to DataFrame ---
input_df = pd.DataFrame([user_input])

# One-hot encode categorical features (like training)
input_df = pd.get_dummies(input_df, drop_first=True)

# Add missing columns (columns in training model but not in input)
missing_cols = set(model.feature_names_in_) - set(input_df.columns)
for c in missing_cols:
    input_df[c] = 0

# Ensure columns order matches training
input_df = input_df[model.feature_names_in_]

# --- Prediction ---
if st.button("Predict Trip Price"):
    prediction = model.predict(input_df)[0]
    st.subheader("💰 Predicted Trip Price")
    st.write(f"${prediction:.2f}")

    # --- Feature importance plot ---
    importances = model.feature_importances_
    feat_importance = pd.Series(importances, index=model.feature_names_in_).sort_values(ascending=False)
    st.subheader("📊 Feature Importance")
    fig, ax = plt.subplots(figsize=(12,6))
    feat_importance.plot(kind='bar', ax=ax)
    st.pyplot(fig)
