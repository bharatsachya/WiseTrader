# Stock Feature Engineering

The `StockFeatureEngineer` class provides a convenient way to generate various technical indicators and features from raw stock market data. These features can then be used as inputs for machine learning models to improve prediction accuracy.

## Purpose

To standardize and simplify the process of adding common technical indicators (e.g., Moving Averages, RSI, MACD) to a stock DataFrame.

## Implementation Details

*   **Location**: `stock.py`
*   **Class**: `StockFeatureEngineer`
*   **Initialization**: Takes a pandas DataFrame with expected columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`. It converts 'Date' to datetime and sets it as the index, and standardizes column names to lowercase.

### Available Methods (Chainable)

*   **`add_moving_averages(windows=[20, 50, 200])`**:
    *   Adds Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) for specified window sizes.
    *   New columns: `sma_X`, `ema_X` (where X is the window).
*   **`add_rsi(window=14)`**:
    *   Adds the Relative Strength Index (RSI).
    *   New column: `rsi`.
*   **`add_macd(fast=12, slow=26, signal=9)`**:
    *   Adds Moving Average Convergence Divergence (MACD), MACD Signal Line, and MACD Histogram.
    *   New columns: `macd`, `macd_signal`, `macd_hist`.
*   **`add_bollinger_bands(window=20, num_std_dev=2)`**:
    *   Adds Bollinger Bands (Middle Band, Upper Band, Lower Band).
    *   New columns: `bollinger_mid`, `bollinger_upper`, `bollinger_lower`.
*   **`add_stochastic_oscillator(k_window=14, d_window=3)`**:
    *   Adds Stochastic Oscillator (%K and %D).
    *   New columns: `stoch_k`, `stoch_d`.
*   **`add_will_r(window=14)`**:
    *   Adds Williams %R.
    *   New column: `will_r`.
*   **`add_obv()`**:
    *   Adds On-Balance Volume (OBV).
    *   New column: `obv`.
*   **`get_dataframe()`**:
    *   Returns the modified DataFrame with all added features.

## Example Usage (`stock.py`)

```python
import pandas as pd
import numpy as np
from stock import StockFeatureEngineer

# Sample DataFrame (replace with your actual data loading)
data = {
    'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=250)),
    'Open': np.random.rand(250) * 100 + 50,
    'High': np.random.rand(250) * 105 + 50,
    'Low': np.random.rand(250) * 95 + 50,
    'Close': np.random.rand(250) * 100 + 50,
    'Volume': np.random.randint(100000, 10000000, 250)
}
df = pd.DataFrame(data)

# Initialize the feature engineer and add features
feature_engineer = StockFeatureEngineer(df)
processed_df = feature_engineer \
    .add_moving_averages(windows=[10, 20]) \
    .add_rsi(window=14) \
    .add_macd() \
    .add_bollinger_bands() \
    .add_stochastic_oscillator() \
    .add_will_r() \
    .add_obv() \
    .get_dataframe()

print(processed_df.head())
print(processed_df.columns)
```
