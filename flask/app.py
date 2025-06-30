from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize Flask
app = Flask(__name__)

# Load model, scaler, and columns
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)

    # Preprocessing
    data.columns = data.columns.str.strip()
    original_data = data.copy()  # keep original for display

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

    # Prepare for rendering
    table_data = original_data.values.tolist()
    column_names = original_data.columns.tolist()

    return render_template("result.html", table_data=table_data, column_names=column_names)

if __name__ == '__main__':
    app.run(debug=True)
