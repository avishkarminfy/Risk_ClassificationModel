import streamlit as st
import pandas as pd
import pickle
import os

# Get folder path where this Python file is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths to load files
model = pickle.load(open(os.path.join(BASE_DIR, "xgb_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

st.set_page_config(page_title="Loan Defaulter Prediction", layout="wide")

st.title("💼 Loan Defaulter Predictor")
st.markdown("Upload a CSV file with customer details to predict loan defaulters.")

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(data.head())

    # Preprocessing
    data.columns = data.columns.str.strip()
    original_data = data.copy()

    data.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')
    if 'Experience' in data.columns:
        data = data[data['Experience'] >= 0]
    data = pd.get_dummies(data, drop_first=True)

    for col in columns:
        if col not in data.columns:
            data[col] = 0
    data = data[columns]

    # Scale and predict
    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)

    # Add predictions to original data
    original_data["Loan Status"] = ["Defaulter" if p == 1 else "Non-Defaulter" for p in preds]

    st.subheader("✅ Prediction Results")
    st.dataframe(original_data)

    # Option to download result
    csv_download = original_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_download,
        file_name='loan_predictions.csv',
        mime='text/csv'
    )
