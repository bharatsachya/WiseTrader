# Stock Feature Engineer

**File:** `stock.py`

The `StockFeatureEngineer` class is a utility designed to generate various technical indicators and features from raw stock market data. It helps in preparing a richer dataset for machine learning models by adding common financial metrics.

## Class: `StockFeatureEngineer`

```python
class StockFeatureEngineer:
    def __init__(self, df):
        # ... (initialization code)

    def add_moving_averages(self, windows=[20, 50, 200]):
        # ... (implementation)

    def add_rsi(self, window=14):
        # ... (implementation)

    def add_macd(self, fast=12, slow=26, signal=9):
        # ... (implementation)

    def add_bollinger_bands(self, window=20, num_std_dev=2):
        # ... (implementation)

    def add_volume_indicators(self, window=14):
        # ... (implementation)

    def add_volatility(self, window=14):
        # ... (implementation)

    def dataframe(self):
        # ... (implementation)
```

### `__init__(self, df)`
-   **Purpose:** Initializes the `StockFeatureEngineer` with a pandas DataFrame containing stock data.
-   **Parameters:**
    -   `df` (pandas.DataFrame): The input DataFrame. Expected columns include `Date`, `Open`, `High`, `Low`, `Close`, `Volume` (case-insensitive).
-   **Details:** Ensures the 'Date' column is converted to datetime objects and set as the index, and standardizes column names to lowercase.

### `add_moving_averages(self, windows=[20, 50, 200])`
-   **Purpose:** Adds Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) to the DataFrame.
-   **Parameters:**
    -   `windows` (list of int, optional): A list of window sizes for calculating the moving averages. Defaults to `[20, 50, 200]`.
-   **Returns:** `self` (StockFeatureEngineer): The instance with added moving average columns.
-   **New Columns:** `sma_X`, `ema_X` (where X is a window size).

### `add_rsi(self, window=14)`
-   **Purpose:** Adds the Relative Strength Index (RSI) to the DataFrame.
-   **Parameters:**
    -   `window` (int, optional): The look-back period for RSI calculation. Defaults to `14`.
-   **Returns:** `self` (StockFeatureEngineer): The instance with the added RSI column.
-   **New Column:** `rsi`.

### `add_macd(self, fast=12, slow=26, signal=9)`
-   **Purpose:** Adds Moving Average Convergence Divergence (MACD), MACD Signal Line, and MACD Histogram to the DataFrame.
-   **Parameters:**
    -   `fast` (int, optional): The fast period for EMA. Defaults to `12`.
    -   `slow` (int, optional): The slow period for EMA. Defaults to `26`.
    -   `signal` (int, optional): The signal line period for EMA. Defaults to `9`.
-   **Returns:** `self` (StockFeatureEngineer): The instance with added MACD columns.
-   **New Columns:** `macd`, `macd_signal`, `macd_hist`.

### `add_bollinger_bands(self, window=20, num_std_dev=2)`
-   **Purpose:** Adds Bollinger Bands (Middle Band, Upper Band, Lower Band) to the DataFrame.
-   **Parameters:**
    -   `window` (int, optional): The look-back period for calculating the moving average and standard deviation. Defaults to `20`.
    -   `num_std_dev` (int, optional): The number of standard deviations for the upper and lower bands. Defaults to `2`.
-   **Returns:** `self` (StockFeatureEngineer): The instance with added Bollinger Band columns.
-   **New Columns:** `bb_middle`, `bb_upper`, `bb_lower`.

### `add_volume_indicators(self, window=14)`
-   **Purpose:** Adds Volume Moving Average (VMA) and On-Balance Volume (OBV) to the DataFrame.
-   **Parameters:**
    -   `window` (int, optional): The look-back period for Volume Moving Average. Defaults to `14`.
-   **Returns:** `self` (StockFeatureEngineer): The instance with added volume indicator columns.
-   **New Columns:** `vma`, `obv`.

### `add_volatility(self, window=14)`
-   **Purpose:** Adds True Range and Average True Range (ATR) to the DataFrame.
-   **Parameters:**
    -   `window` (int, optional): The look-back period for ATR. Defaults to `14`.
-   **Returns:** `self` (StockFeatureEngineer): The instance with added volatility columns.
-   **New Columns:** `tr`, `atr`.

### `dataframe(self)`
-   **Purpose:** Returns the DataFrame with all the calculated features.
-   **Returns:** `pandas.DataFrame`: The DataFrame with technical indicators.

## Usage Example

```python
import yfinance as yf
import pandas as pd
from stock import StockFeatureEngineer

# 1. Fetch historical stock data
df_raw = yf.download('GOOG', start='2022-01-01', end='2023-01-01')
df_raw.reset_index(inplace=True)

# 2. Instantiate the feature engineer
engineer = StockFeatureEngineer(df_raw)

# 3. Add desired technical indicators (chainable methods)
engineered_df = engineer.add_moving_averages([10, 30]) \
                        .add_rsi() \
                        .add_macd() \
                        .add_bollinger_bands() \
                        .dataframe()

# 4. Display the DataFrame with new features
print(engineered_df.head())
print(engineered_df.columns)
```
