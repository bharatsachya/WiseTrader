"""
Rare Stock Prediction Algorithm
--------------------------------
Unscented Kalman Filter (UKF)
with Stochastic Volatility State Space Model

Author: Lovanshu Garg (customized)
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ===============================
# UKF IMPLEMENTATION
# ===============================

class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, fx, hx, Q, R):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self.alpha**2 * (dim_x + self.kappa) - dim_x

        self.gamma = np.sqrt(dim_x + self.lambda_)
        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

    def sigma_points(self, x, P):
        U = np.linalg.cholesky(P)
        sigmas = [x]
        for i in range(len(x)):
            sigmas.append(x + self.gamma * U[:, i])
            sigmas.append(x - self.gamma * U[:, i])
        return np.array(sigmas)

    def predict(self):
        sigmas = self.sigma_points(self.x, self.P)
        sigmas_f = np.array([self.fx(s) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        self.P = self.Q.copy()

        for i in range(len(sigmas_f)):
            diff = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

        self.sigmas_f = sigmas_f

    def update(self, z):
        sigmas_h = np.array([self.hx(s) for s in self.sigmas_f])
        z_pred = np.sum(self.Wm * sigmas_h)

        S = self.R
        for i in range(len(sigmas_h)):
            diff = sigmas_h[i] - z_pred
            S += self.Wc[i] * diff * diff

        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(sigmas_h)):
            Pxz += self.Wc[i] * np.outer(
                self.sigmas_f[i] - self.x,
                sigmas_h[i] - z_pred
            )

        K = Pxz / S
        self.x += K.flatten() * (z - z_pred)
        self.P -= K @ K.T * S

# ===============================
# STATE TRANSITION & MEASUREMENT
# ===============================

def fx(state):
    price, log_vol = state
    vol = np.exp(log_vol)
    new_price = price + np.random.normal(0, vol)
    new_log_vol = log_vol + np.random.normal(0, 0.02)
    return np.array([new_price, new_log_vol])

def hx(state):
    return state[0]

# ===============================
# DATA LOADING
# ===============================

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
prices = data["Close"].values

# ===============================
# UKF SETUP
# ===============================

ukf = UnscentedKalmanFilter(
    dim_x=2,
    dim_z=1,
    fx=fx,
    hx=hx,
    Q=np.diag([0.1, 0.01]),
    R=1.0
)

ukf.x = np.array([prices[0], np.log(1)])
ukf.P = np.eye(2)

predictions = []

# ===============================
# FILTER LOOP
# ===============================

for price in prices:
    ukf.predict()
    ukf.update(price)
    predictions.append(ukf.x[0])

# ===============================
# NEXT DAY PREDICTION
# ===============================

ukf.predict()
next_day_price = ukf.x[0]
print(f"📈 Predicted Next Day Price: {next_day_price:.2f}")

# ===============================
# VISUALIZATION
# ===============================

plt.figure(figsize=(12, 6))
plt.plot(prices, label="Actual Price")
plt.plot(predictions, label="UKF Prediction", linestyle="--")
plt.title(f"{ticker} Stock Prediction using Unscented Kalman Filter")
plt.legend()
plt.show()
