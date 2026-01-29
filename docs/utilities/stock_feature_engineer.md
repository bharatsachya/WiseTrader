# Stock Feature Engineer Utility (`stock.py`)

The `stock.py` script contains the `StockFeatureEngineer` class, a utility designed to generate various technical indicators and derive features from raw stock market data. This class is essential for preparing data for machine learning models that benefit from these indicators.

## `StockFeatureEngineer` Class

### `__init__(self, df)`

*   **Purpose**: Initializes the feature engineer with a DataFrame of stock data.
*   **Parameters**:
    *   `df` (pd.DataFrame): The input stock data DataFrame. Expected columns include `Date`, `Open`, `High`, `Low`, `Close`, `Volume` (case-insensitive).
*   **Initialization**: Ensures the `Date` column is converted to datetime objects and set as the index, and standardizes column names to lowercase.

### `add_moving_averages(self, windows=[20, 50, 200])`

*   **Purpose**: Adds Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) to the DataFrame.
*   **Parameters**:
    *   `windows` (list): A list of integers representing the window sizes for the moving averages (default: `[20, 50, 200]`).
*   **Adds Columns**: `sma_{window}` and `ema_{window}` (e.g., `sma_20`, `ema_50`).
*   **Returns**: `self` (allows method chaining).

### `add_rsi(self, window=14)`

*   **Purpose**: Adds the Relative Strength Index (RSI) to the DataFrame.
*   **Parameters**:
    *   `window` (int): The period over which to calculate RSI (default: 14).
*   **Adds Column**: `rsi`.
*   **Returns**: `self` (allows method chaining).

### `add_macd(self, fast=12, slow=26, signal=9)`

*   **Purpose**: Adds Moving Average Convergence Divergence (MACD), MACD Signal Line, and MACD Histogram to the DataFrame.
*   **Parameters**:
    *   `fast` (int): Fast EMA window (default: 12).
    *   `slow` (int): Slow EMA window (default: 26).
    *   `signal` (int): Signal line EMA window (default: 9).
*   **Adds Columns**: `macd`, `macd_signal`, `macd_hist`.
*   **Returns**: `self` (allows method chaining).

### `add_bollinger_bands(self, window=20, num_std=2)`

*   **Purpose**: Adds Bollinger Bands (Middle Band, Upper Band, Lower Band) to the DataFrame.
*   **Parameters**:
    *   `window` (int): Rolling window for the middle band and standard deviation (default: 20).
    *   `num_std` (int): Number of standard deviations for upper and lower bands (default: 2).
*   **Adds Columns**: `bb_middle`, `bb_upper`, `bb_lower`.
*   **Returns**: `self` (allows method chaining).

### `add_vwap(self)`

*   **Purpose**: Adds Volume Weighted Average Price (VWAP) to the DataFrame.
*   **Assumes**: Daily data is input for accurate VWAP calculation for intraday is more complex.
*   **Adds Column**: `vwap`.
*   **Returns**: `self` (allows method chaining).

### `add_average_daily_range(self, window=14)`

*   **Purpose**: Adds Average Daily Range (ADR) to the DataFrame.
*   **Parameters**:
    *   `window` (int): Rolling window for ADR (default: 14).
*   **Adds Column**: `adr`.
*   **Returns**: `self` (allows method chaining).

### `add_lagged_features(self, lags=[1, 2, 3, 5])`

*   **Purpose**: Adds lagged versions of the 'close' price as features.
*   **Parameters**:
    *   `lags` (list): A list of integers representing the number of periods to lag (default: `[1, 2, 3, 5]`).
*   **Adds Columns**: `close_lag_{lag}` (e.g., `close_lag_1`).
*   **Returns**: `self` (allows method chaining).

### `add_daily_returns(self)`

*   **Purpose**: Calculates and adds daily percentage returns for the 'close' price.
*   **Adds Column**: `daily_return`.
*   **Returns**: `self` (allows method chaining).

### `add_all_features(self)`

*   **Purpose**: A convenience method to add a comprehensive set of common technical indicators and features.
*   **Returns**: `self` (allows method chaining).

### `get_df(self)`

*   **Purpose**: Returns the DataFrame with all the added features.
*   **Returns**:
    *   `pd.DataFrame`: The modified DataFrame containing original data plus engineered features.

## Example Usage

```python
import pandas as pd
# Assuming df is your initial DataFrame with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
# Example: df = yf.download('AAPL', start='2020-01-01', end='2023-01-01').reset_index()

# Create an instance of the feature engineer
fe = StockFeatureEngineer(df)

# Add all common features using method chaining
df_features = fe.add_all_features().get_df()

# Or add features individually
# df_features = fe.add_moving_averages([10, 30])\
#                 .add_rsi()\
#                 .add_macd()\
#                 .get_df()

print(df_features.head())
```