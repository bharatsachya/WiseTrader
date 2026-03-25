import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. DATA LOADING
# -----------------------------
def load_data(symbol="AAPL", start="2015-01-01"):
    df = yf.download(symbol, start=start)
    df.dropna(inplace=True)
    return df

# -----------------------------
# 2. FEATURE ENGINEERING
# -----------------------------
def create_features(df):
    df = df.copy()

    # Returns
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()

    # Momentum
    df['momentum'] = df['Close'] / df['Close'].shift(10) - 1

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Moving averages
    df['ma_fast'] = df['Close'].rolling(10).mean()
    df['ma_slow'] = df['Close'].rolling(50).mean()
    df['ma_ratio'] = df['ma_fast'] / df['ma_slow']

    df.dropna(inplace=True)
    return df

# -----------------------------
# 3. REGIME DETECTION (HMM)
# -----------------------------
def fit_hmm(df, n_states=3):
    features = df[['returns', 'volatility']].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(features_scaled)

    hidden_states = model.predict(features_scaled)
    df['regime'] = hidden_states

    return df, model, scaler

# -----------------------------
# 4. SEQUENCE DATA FOR LSTM
# -----------------------------
def create_sequences(df, feature_cols, target_col='returns', window=20):
    X, y = [], []

    data = df[feature_cols].values
    target = df[target_col].values

    for i in range(window, len(df)):
        X.append(data[i-window:i])
        y.append(target[i])

    return np.array(X), np.array(y)

# -----------------------------
# 5. LSTM MODEL
# -----------------------------
def build_lstm(input_shape):
    inp = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    x = LSTM(32)(x)
    x = Dropout(0.2)(x)

    out = Dense(1, activation='tanh')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')

    return model

# -----------------------------
# 6. POSITION SIZING LOGIC
# -----------------------------
def position_sizing(pred, regime, volatility):
    """
    Combine prediction + regime + volatility
    """

    base_signal = np.sign(pred)

    # Regime weighting
    if regime == 0:   # bull
        weight = 1.5
    elif regime == 1: # neutral
        weight = 1.0
    else:             # bear
        weight = 0.5

    # Volatility scaling (risk parity style)
    vol_adj = 1 / (volatility + 1e-6)

    position = base_signal * weight * vol_adj

    # Clip leverage
    return np.clip(position, -2, 2)

# -----------------------------
# 7. BACKTEST
# -----------------------------
def backtest(df, preds):
    df = df.copy()
    df = df.iloc[-len(preds):]

    positions = []
    for i in range(len(preds)):
        pos = position_sizing(
            preds[i],
            df['regime'].iloc[i],
            df['volatility'].iloc[i]
        )
        positions.append(pos)

    df['position'] = positions
    df['strategy_returns'] = df['position'].shift(1) * df['returns']

    df['cum_returns'] = np.exp(df['strategy_returns'].cumsum())

    return df

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    df = load_data("AAPL")
    df = create_features(df)

    df, hmm_model, scaler = fit_hmm(df)

    feature_cols = ['returns', 'volatility', 'momentum', 'rsi', 'ma_ratio', 'regime']

    X, y = create_sequences(df, feature_cols)

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    preds = model.predict(X_test).flatten()

    result = backtest(df.iloc[-len(preds):], preds)

    print(result[['cum_returns']].tail())
