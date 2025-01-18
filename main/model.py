import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
dataset_path = 'D:/stock_prediction/World-Stock-Prices-Dataset.csv'
df = pd.read_csv(dataset_path)

# Step 1: Initial Inspection
print("Initial Dataset Information:")
print(df.info())
print(df.head())

# Step 2: Data Cleaning
df = df.drop(columns=['Capital Gains'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Ticker', 'Date'])

# Step 3: Feature Engineering
df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
df['SMA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=200).mean())
df['Price_Range'] = df['High'] - df['Low']
df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
df = df.dropna()

# Step 4: Preparing the Data for the Model
features = ['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'Price_Range', 'Daily_Return']
target = 'Close'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()  # Initialize the scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)        # Transform the test data

# Step 6: Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Step 8: Save the Model and Scaler
model_filename = 'stock_trend_prediction_model.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"\nModel saved as: {model_filename}")
print(f"Scaler saved as: {scaler_filename}")

# Step 9: Make a Prediction (Example)
example_data = X_test.iloc[0].values.reshape(1, -1)
example_data_scaled = scaler.transform(example_data)  # Scale the example input
example_prediction = model.predict(example_data_scaled)
print("\nExample Prediction:")
print(f"Input Features: {X_test.iloc[0].to_dict()}")
print(f"Predicted Close Price: {example_prediction[0]:.2f}")
