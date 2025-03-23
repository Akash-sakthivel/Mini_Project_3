import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the cleaned dataset and trained model
df_cleaned = pd.read_csv("cleaned_crop_data.csv")
model = joblib.load("crop_production_model.pkl", mmap_mode=None)

# Load model features
try:
    model_features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("Feature mapping file not found. Please check your model setup.")
    model_features = []

# Streamlit UI
st.title("Crop Production Prediction")
st.sidebar.image("crop_logo.png", use_container_width=True)
st.sidebar.header("User Input")

# User Inputs
area = st.sidebar.selectbox("Select Region", df_cleaned["Area"].unique())
crop = st.sidebar.selectbox("Select Crop", df_cleaned["Item"].unique())
year = st.sidebar.slider("Select Year", int(df_cleaned["Year"].min()), int(df_cleaned["Year"].max()), 2025)

# Convert inputs into model-friendly format
input_data = pd.DataFrame([[area, crop, year]], columns=["Area", "Item", "Year"])

# One-hot encoding
input_encoded = pd.get_dummies(input_data, columns=["Area", "Item"])

# Align input with model features
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Ensure input features match model's feature names
input_encoded = input_encoded.astype(float)

# Prediction
if st.sidebar.button("Predict Production"):
    prediction = model.predict(input_encoded)
    st.session_state.prediction = np.round(prediction[0], 2)
    st.success(f"Predicted Crop Production: {st.session_state.prediction} tons")

# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis")

# Dataset Summary
st.subheader("Dataset Summary")
st.write(df_cleaned.describe())

# Crop Production Trends
st.subheader("Crop Production Trends Over the Years")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x='Year', y='Production', data=df_cleaned, hue='Item', legend=False, ax=ax)
ax.set_title("Yearly Trend in Crop Production")
st.pyplot(fig)

# Region-wise Production Distribution
st.subheader("Region-wise Production Distribution")
fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data=df_cleaned, x="Area", y="Production", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Download the Prediction Report
if st.button("Download Report") and "prediction" in st.session_state:
    result = pd.DataFrame({"Region": [area], "Crop": [crop], "Year": [year], "Predicted Production": [st.session_state.prediction]})
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download as CSV", data=csv, file_name="Crop_Prediction_Report.csv", mime="text/csv")
