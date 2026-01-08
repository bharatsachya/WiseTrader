"""
Rare Stock Prediction Algorithm
--------------------------------
Topological Data Analysis (TDA)
Persistent Homology + SVM

Author: Lovanshu Garg (customized)
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from ripser import ripser
from persim import PersistenceEntropy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ===============================
# CONFIG
# ===============================

TICKER = "AAPL"
WINDOW_SIZE = 30
PRED_HORIZON = 1

# ===============================
# DATA LOADING
# ===============================

data = yf.download(TICKER, start="2019-01-01", end="2024-01-01")
prices = data["Close"].values

# ===============================
# TDA FEATURE EXTRACTION
# ===============================

def sliding_window(series, window):
    return np.array([series[i:i+window] for i in range(len(series) - window)])

def tda_features(window):
    window = window.reshape(-1, 1)
    diagrams = ripser(window, maxdim=1)["dgms"]

    pe = PersistenceEntropy()
    entropy_0 = pe.fit_transform([diagrams[0]])[0]
    entropy_1 = pe.fit_transform([diagrams[1]])[0] if len(diagrams) > 1 else 0.0

    return [entropy_0, entropy_1]

X = []
y = []

windows = sliding_window(prices, WINDOW_SIZE)

for i in range(len(windows) - PRED_HORIZON):
    feat = tda_features(windows[i])
    X.append(feat)

    future_return = prices[i + WINDOW_SIZE + PRED_HORIZON - 1] - prices[i + WINDOW_SIZE - 1]
    y.append(1 if future_return > 0 else 0)

X = np.array(X)
y = np.array(y)

# ===============================
# TRAIN / TEST SPLIT
# ===============================

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# MODEL
# ===============================

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"📊 Prediction Accuracy: {accuracy * 100:.2f}%")

# ===============================
# NEXT DAY PREDICTION
# ===============================

latest_window = prices[-WINDOW_SIZE:]
latest_feat = np.array(tda_features(latest_window)).reshape(1, -1)
latest_feat = scaler.transform(latest_feat)

prob_up = model.predict_proba(latest_feat)[0][1]
direction = "UP 📈" if prob_up > 0.5 else "DOWN 📉"

estimated_price = prices[-1] * (1 + (0.002 if prob_up > 0.5 else -0.002))

print(f"🔮 Next Day Direction: {direction}")
print(f"📈 Estimated Next Price: {estimated_price:.2f}")

# ===============================
# VISUALIZATION
# ===============================

plt.figure(figsize=(12, 5))
plt.plot(prices, label="Price")
plt.axvline(len(prices) - 1, linestyle="--", color="gray")
plt.title(f"{TICKER} Stock Price with TDA-Based Prediction")
plt.legend()
plt.show()
