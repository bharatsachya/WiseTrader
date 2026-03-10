import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# -----------------------------
# 1. Generate sample market data
# -----------------------------

np.random.seed(42)

hours = 200

data = pd.DataFrame({
    "demand": np.random.normal(1000, 100, hours),
    "renewable_supply": np.random.normal(300, 50, hours),
    "temperature": np.random.normal(25, 5, hours)
})

# Electricity price influenced by demand and supply
data["price"] = (
    0.05 * data["demand"]
    - 0.04 * data["renewable_supply"]
    + 0.3 * data["temperature"]
    + np.random.normal(0, 5, hours)
)

# -----------------------------
# 2. Train price prediction model
# -----------------------------

X = data[["demand", "renewable_supply", "temperature"]]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 3. Forecast next hour market
# -----------------------------

future_market = pd.DataFrame({
    "demand": [1100],
    "renewable_supply": [250],
    "temperature": [30]
})

predicted_price = model.predict(future_market)[0]

print("Predicted electricity price:", predicted_price)

# -----------------------------
# 4. Trading optimization
# Decide how much power to buy/sell
# -----------------------------

max_capacity = 500

def profit(power):
    
    # cost to generate power
    generation_cost = 30
    
    revenue = predicted_price * power
    cost = generation_cost * power
    
    return -(revenue - cost)   # negative for minimization


result = minimize(
    profit,
    x0=[100],
    bounds=[(0, max_capacity)]
)

optimal_power = result.x[0]

# -----------------------------
# 5. Final trading decision
# -----------------------------

if predicted_price > 30:
    decision = "SELL power"
else:
    decision = "BUY power"

print("Trading Decision:", decision)
print("Optimal Power (MW):", optimal_power)
print("Expected Profit:", (predicted_price - 30) * optimal_power)
