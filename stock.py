# `stock.py`: Stock Feature Engineering Utility

This file defines the `StockFeatureEngineer` class, a utility designed to generate a comprehensive set of technical indicators and features from raw stock market data. These features can then be used as input for various machine learning or neural network models for prediction and analysis.

## `StockFeatureEngineer` Class

### Purpose

The `StockFeatureEngineer` class takes a pandas DataFrame of stock data (expected to have 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' columns) and enriches it by adding several commonly used technical indicators.

### Initialization

```python
import pandas as pd
from stock import StockFeatureEngineer

# Example DataFrame (ensure 'Date' is present and convertible to datetime)
sample_df = pd.DataFrame({
    'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
    'Open': [100, 102, 101, 103, 105],
    'High': [103, 104, 103, 105, 106],
    'Low': [99, 101, 100, 102, 104],
    'Close': [102, 103, 102, 104, 105],
    'Volume': [1000, 1200, 1100, 1300, 1400]
})

feature_engineer = StockFeatureEngineer(sample_df)
```

### Methods

#### `add_moving_averages(windows=[20, 50, 200])`

Adds Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) for specified window periods.

-   `windows` (list of int): A list of window sizes for which to calculate the MAs. Defaults to `[20, 50, 200]`.

    *Returns:* `self` for method chaining.

#### `add_rsi(window=14)`

Adds the Relative Strength Index (RSI).

-   `window` (int): The period over which to calculate RSI. Defaults to `14`.

    *Returns:* `self`.

#### `add_macd(fast=12, slow=26, signal=9)`

Adds Moving Average Convergence Divergence (MACD), MACD Signal Line, and MACD Histogram.

-   `fast` (int): The period for the fast EMA. Defaults to `12`.
-   `slow` (int): The period for the slow EMA. Defaults to `26`.
-   `signal` (int): The period for the MACD Signal Line EMA. Defaults to `9`.

    *Returns:* `self`.

#### `add_bollinger_bands(window=20, num_std=2)`

Adds Bollinger Bands (Middle Band, Upper Band, Lower Band).

-   `window` (int): The period for the Moving Average. Defaults to `20`.
-   `num_std` (int): The number of standard deviations for the upper and lower bands. Defaults to `2`.

    *Returns:* `self`.

#### `add_stochastic_oscillator(k_window=14, d_window=3)`

Adds Stochastic Oscillator (%K and %D).

-   `k_window` (int): The period for %K. Defaults to `14`.
-   `d_window` (int): The period for %D (SMA of %K). Defaults to `3`.

    *Returns:* `self`.

#### `add_volatility(window=20)`

Adds historical volatility (standard deviation of log returns).

-   `window` (int): The period for calculating volatility. Defaults to `20`.

    *Returns:* `self`.

#### `add_volume_features()`

Adds volume-based features: Volume Moving Average and Volume Change.

    *Returns:* `self`.

#### `add_lagged_features(lags=[1, 2, 3])`

Adds lagged closing prices as features.

-   `lags` (list of int): A list of lag periods. Defaults to `[1, 2, 3]`.

    *Returns:* `self`.

### Example Usage

```python
import pandas as pd
from stock import StockFeatureEngineer

# Create a sample DataFrame (replace with actual stock data loading)
data = {
    'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08']),
    'Open': [100, 102, 101, 103, 105, 104, 106, 105],
    'High': [103, 104, 103, 105, 106, 105, 107, 106],
    'Low': [99, 101, 100, 102, 104, 103, 105, 104],
    'Close': [102, 103, 102, 104, 105, 104, 106, 105],
    'Volume': [1000, 1200, 1100, 1300, 1400, 1150, 1350, 1200]
}
df = pd.DataFrame(data)

# Initialize the feature engineer and chain methods to add features
fe = StockFeatureEngineer(df)
df_with_features = fe \
    .add_moving_averages(windows=[5, 10]) \
    .add_rsi(window=7) \
    .add_macd() \
    .add_bollinger_bands() \
    .add_stochastic_oscillator() \
    .add_volatility() \
    .add_volume_features() \
    .add_lagged_features() \
    .df

print(df_with_features.tail())
```