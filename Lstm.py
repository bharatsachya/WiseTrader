import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Stock Data
# -----------------------------
ticker = "AAPL"  # You can change this
data = yf.download(ticker, start="2010-01-01", end="2025-01-01")
close_prices = data["Close"].values.reshape(-1, 1)

# -----------------------------
# 2. Normalize Data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# -----------------------------
# 3. Create Sequence Data
# -----------------------------
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM input shape

# Train/Test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 4. Build LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# -----------------------------
# 5. Predict
# -----------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# 6. Plot Results
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.show()
