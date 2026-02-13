"""
Improved Rare Stock Prediction Algorithm
----------------------------------------
Unscented Kalman Filter (UKF)
with Stochastic Volatility State Space Model

Improved Version
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ===============================
# UNSCENTED KALMAN FILTER CLASS
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

        # UKF parameters
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
        for i in range(self.dim_x):
            sigmas.append(x + self.gamma * U[:, i])
            sigmas.append(x - self.gamma * U[:, i])
        return np.array(sigmas)

    def predict(self):
        sigmas = self.sigma_points(self.x, self.P)
        sigmas_f = np.array([self.fx(s) for s in sigmas])

        # Predicted mean
        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        # Predicted covariance
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(len(sigmas_f)):
            diff = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q

        self.sigmas_f = sigmas_f

    def update(self, z):
        sigmas_h = np.array([self.hx(s) for s in self.sigmas_f])
        z_pred = np.sum(self.Wm * sigmas_h)

        # Innovation covariance
        S = 0
        for i in range(len(sigmas_h)):
            diff = sigmas_h[i] - z_pred
            S += self.Wc[i] * diff * diff
        S += self.R

        # Cross covariance
        Pxz = np.zeros((self.dim_x, 1))
        for i in range(len(sigmas_h)):
            Pxz += self.Wc[i] * np.outer(
                self.sigmas_f[i] - self.x,
                sigmas_h[i] - z_pred
            )

        # Kalman Gain
        K = Pxz / S

        # State update
        self.x += (K.flatten() * (z - z_pred))

        # Covariance update
        self.P -= K @ K.T * S


# ===============================
# STOCHASTIC VOLATILITY MODEL
# ===============================

def fx(state):
    """
    State = [return, log_volatility]
    """
    r, log_vol = state

    # Mean reversion in volatility
    phi = 0.98
    sigma_vol = 0.1

    new_r = r  # Randomness handled by Q
    new_log_vol = phi * log_vol

    return np.array([new_r, new_log_vol])


def hx(state):
    """
    Measurement = observed return
    """
    return state[0]


# ===============================
# DATA LOADING
# ===============================

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
prices = data["Close"].values

# Use log returns instead of price
returns = np.diff(np.log(prices))

# ===============================
# UKF INITIALIZATION
# ===============================

ukf = UnscentedKalmanFilter(
    dim_x=2,
    dim_z=1,
    fx=fx,
    hx=hx,
    Q=np.diag([0.0001, 0.01]),
    R=0.001
)

ukf.x = np.array([returns[0], np.log(np.std(returns))])
ukf.P = np.eye(2)

filtered_returns = []

# ===============================
# FILTER LOOP
# ===============================

for r in returns:
    ukf.predict()
    ukf.update(r)
    filtered_returns.append(ukf.x[0])

# ===============================
# NEXT DAY FORECAST
# ===============================

ukf.predict()
next_return = ukf.x[0]

last_price = prices[-1]
next_price = last_price * np.exp(next_return)

print(f"📈 Predicted Next Day Price: {next_price:.2f}")

# ===============================
# RECONSTRUCT PRICE SERIES
# ===============================

reconstructed_prices = [prices[0]]
for r in filtered_returns:
    reconstructed_prices.append(reconstructed_prices[-1] * np.exp(r))

reconstructed_prices = np.array(reconstructed_prices)

# ===============================
# VISUALIZATION
# ===============================

plt.figure(figsize=(12, 6))
plt.plot(prices, label="Actual Price")
plt.plot(reconstructed_prices, label="UKF Filtered Price", linestyle="--")
plt.title(f"{ticker} Stock Prediction using UKF with Stochastic Volatility")
plt.legend()
plt.show()
