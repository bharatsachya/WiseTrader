# Hybrid Wavelet-CNN-BiLSTM-Attention Stock Prediction Model
# Unique architecture for stock movement prediction

import numpy as np
import pandas as pd
import pywt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dense, Dropout, Attention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# -------------------------------
# 1. Download Data
# -------------------------------
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")

df["Return"] = df["Close"].pct_change()
df["Volatility"] = df["Return"].rolling(10).std()
df.dropna(inplace=True)

# -------------------------------
# 2. Wavelet Denoising Function
# -------------------------------
def wavelet_denoise(series, wavelet="db4", level=2):
    coeffs = pywt.wavedec(series, wavelet, mode="per")
    coeffs[1:] = [pywt.threshold(c, value=np.std(c)/2, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")

df["Denoised_Close"] = wavelet_denoise(df["Close"].values)

# -------------------------------
# 3. Feature Engineering
# -------------------------------
df["Momentum"] = df["Close"] - df["Close"].shift(5)
df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
df.dropna(inplace=True)

features = ["Denoised_Close", "Volatility", "Momentum"]
X = df[features].values
y = df["Target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# 4. Sequence Builder
# -------------------------------
def create_sequences(X, y, seq_len=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# -------------------------------
# 5. Unique Hybrid Model
# -------------------------------
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN Block
x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = LayerNormalization()(x)

# BiLSTM Block
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

# Attention Layer
attention = Attention()([x, x])
x = Concatenate()([x, attention])

# Global Pooling
x = GlobalAveragePooling1D()(x)

# Dense Head
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 6. Train
# -------------------------------
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -------------------------------
# 7. Evaluate
# -------------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# -------------------------------
# 8. Predict Next Movement
# -------------------------------
latest_sequence = X_seq[-1].reshape(1, X_seq.shape[1], X_seq.shape[2])
prediction = model.predict(latest_sequence)

print("Next Day Movement Probability (Up):", prediction[0][0])
