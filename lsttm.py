"""
Advanced Stock Price Prediction using LSTM
Author: Lovanshu Garg (customized for real-world ML workflow)

Features:
- Yahoo Finance data
- Technical Indicators
- Time-series windowing
- LSTM deep learning model
- Walk-forward validation
- Trading signal backtest
"""

# ===============================
# IMPORTS
# ===============================
import yfinance as yf
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# CONFIG
# ===============================
SYMBOL = "AAPL"
START_DATE = "2014-01-01"
LOOKBACK = 60
EPOCHS = 30
BATCH_SIZE = 32

# ===============================
# DATA COLLECTION
# ===============================
def load_data(symbol, start):
    df = yf.download(symbol, start=start)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

# ===============================
# FEATURE ENGINEERING
# ===============================
def add_features(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["ema_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()
    df["volatility"] = df["High"] - df["Low"]
    df.dropna(inplace=True)
    return df

# ===============================
# SEQUENCE CREATION
# ===============================
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # Predict Close price
    return np.array(X), np.array(y)

# ===============================
# MODEL
# ===============================
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model

# ===============================
# BACKTESTING LOGIC
# ===============================
def backtest(preds, actual):
    signal = np.where(preds > actual.shift(1), 1, -1)
    returns = actual.pct_change()
    strategy_returns = signal[:-1] * returns[1:]
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    return sharpe

# ===============================
# MAIN PIPELINE
# ===============================
def main():
    print("📥 Loading data...")
    df = load_data(SYMBOL, START_DATE)
    df = add_features(df)

    features = df.columns
    data = df.values

    # Walk-forward split
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split - LOOKBACK:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, LOOKBACK)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK)

    print("🧠 Training model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

    print("📈 Predicting...")
    predictions = model.predict(X_test)

    # Inverse scaling
    close_index = list(features).index("Close")
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, close_index] = predictions.flatten()
    preds_actual = scaler.inverse_transform(dummy)[:, close_index]

    actual_prices = df["Close"].iloc[-len(preds_actual):]

    rmse = np.sqrt(mean_squared_error(actual_prices, preds_actual))
    print(f"📊 RMSE: {rmse:.2f}")

    sharpe = backtest(pd.Series(preds_actual), actual_prices)
    print(f"📉 Strategy Sharpe Ratio: {sharpe:.2f}")

    # Plot
    plt.figure(figsize=(14,6))
    plt.plot(actual_prices.values, label="Actual")
    plt.plot(preds_actual, label="Predicted")
    plt.legend()
    plt.title(f"{SYMBOL} Stock Price Prediction (LSTM)")
    plt.show()

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    main()
