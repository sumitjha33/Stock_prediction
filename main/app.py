from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model and scaler
model_filename = 'main/stock_trend_prediction_model.pkl'
scaler_filename = 'main/scaler.pkl'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    data = request.form
    input_df = pd.DataFrame([data])

    # List of features expected by the model
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'Price_Range', 'Daily_Return']

    # Ensure all required features are present
    if not all(feature in input_df.columns for feature in features):
        return jsonify({'error': 'Missing required features'}), 400

    # Scale the input data
    scaled_data = scaler.transform(input_df[features])

    # Make prediction
    prediction = model.predict(scaled_data)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
