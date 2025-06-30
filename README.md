# 🧪 Bank Loan Default Prediction System – Flask App

This Flask application allows users to upload a CSV file and get predictions for whether each customer is a **loan defaulter or not**, using a pre-trained XGBoost model.

---

## 🚀 Features

- 🔹 Upload `.csv` file via web interface
  ![image](https://github.com/user-attachments/assets/9f0ae0b0-7867-4163-b554-b95be29472bf)
- 🔹 Backend preprocessing: scaling, encoding, missing columns
- 🔹 Uses trained `xgb_model.pkl`, `scaler.pkl`, and `columns.pkl`
- 🔹 Returns predictions in an easy-to-read HTML table
  ![image](https://github.com/user-attachments/assets/d19fcc9c-e15c-4cda-bc28-d2f2a6935f14)

---
## 📂 Required Files
Place the following files in the same directory as app.py:

- 🔹xgb_model.pkl – Trained XGBoost model
- 🔹scaler.pkl – Scaler used during training
- 🔹columns.pkl – List of features used in training

  ```md
# 📊 Bank Loan Default Prediction System – Streamlit Dashboard

An interactive Streamlit dashboard that allows you to upload a `.csv` file and get **batch predictions** on loan default risk using a trained XGBoost model.

---

## 🌟 Features

- 📤 Upload customer CSV files directly from the UI
- ![image](https://github.com/user-attachments/assets/7711f066-7859-4710-a97c-6401cad7f809)
- 🔍 Displays full customer data + prediction result
- 🧪 Uses pre-trained model + preprocessing
- ⚡ Fast, simple, and user-friendly layout
- ![image](https://github.com/user-attachments/assets/148097d4-da6d-479f-b407-ad4c6cc6d780)

---

## 📁 Required Files
Ensure these files are in the same folder as streamlit_app.py:

- xgb_model.pkl
- scaler.pkl
- columns.pkl



