# Wise LSTM Model

The Wise LSTM (Long Short-Term Memory) model is designed for time-series forecasting, specifically for predicting stock closing prices. LSTM networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies, making them well-suited for sequential data like financial time series.

## Purpose

This model predicts the future closing price of a stock based on its historical 'Close' prices.

## Implementation Details

*   **Location**: `pages/wise_lstm.py` (Streamlit integration) and `Lstm.py` (core model script, though `wise_lstm.py` contains its own full implementation tailored for Streamlit).
*   **Data Source**: Fetches historical stock data using `yfinance`.
*   **Preprocessing**: Stock 'Close' prices are normalized using `MinMaxScaler`.
*   **Sequence Creation**: Data is transformed into sequences for LSTM input, where each sequence represents a look-back period.
*   **Model Architecture (`pages/wise_lstm.py` / `Lstm.py`):**
    *   Input Layer: `LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1))`
    *   Hidden Layer: `LSTM(units=50)`
    *   Output Layer: `Dense(units=1)` for predicting a single price value.
*   **Compilation**: Uses `adam` optimizer and `mean_squared_error` loss function.
*   **Training**: Trained on 80% of the historical data for 10 epochs (in `wise_lstm.py`).
*   **Prediction**: Generates predictions on the test set and inverse transforms them to original price scale.

## How to Use (via Streamlit)

1.  Run the WiseTrader application as described in the [Usage Guide](usage.md).
2.  Select 'Wise LSTM' from the sidebar navigation.
3.  **Enter Stock Ticker**: Input the ticker symbol of the stock you want to predict (e.g., `AAPL`).
4.  **Select Date Range**: Choose the start and end dates for the historical data to be used for training and prediction.
5.  The application will automatically fetch data, train the LSTM model, make predictions, and display a plot comparing actual vs. predicted prices.

## Code Snippet (Core Logic from `Lstm.py`)

```python
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load Stock Data
ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2025-01-01")
close_prices = data["Close"].values.reshape(-1, 1)

# 2. Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 3. Create Sequence Data
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # LSTM input shape

# Train/Test split example
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# 5. Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
```
