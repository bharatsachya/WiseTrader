# `logistic_regression.py`: Stock Prediction using Logistic Regression

This file provides the `StockLogisticModel` class, which implements a traditional machine learning approach for predicting daily stock price movement (up or down) using Logistic Regression. It is designed to work with financial time-series data, particularly with features engineered using tools like `StockFeatureEngineer`.

## `StockLogisticModel` Class

### Purpose

The `StockLogisticModel` aims to predict whether a stock's closing price will increase (target = 1) or decrease (target = 0) on the *next day*, based on a set of input features (technical indicators).

### Initialization

```python
import pandas as pd
from logistic_regression import StockLogisticModel

# `df` should be a DataFrame with 'close' price and potentially other features
# (e.g., from StockFeatureEngineer)
sample_df = pd.DataFrame({
    'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
    'open': [100, 102, 101, 103, 105],
    'high': [103, 104, 103, 105, 106],
    'low': [99, 101, 100, 102, 104],
    'close': [102, 103, 102, 104, 105],
    'volume': [1000, 1200, 1100, 1300, 1400]
}).set_index('date')

# Example with engineered features (assuming `StockFeatureEngineer` was used)
# fe = StockFeatureEngineer(raw_df)
# df_with_features = fe.add_moving_averages().add_rsi().df
# Ensure 'close' column is present for target creation

model_instance = StockLogisticModel(sample_df)
```

### Methods

#### `prepare_data()`

Internal method to create the binary target variable (`Target`) based on whether the next day's close price is higher than today's. It also handles dropping the last row (which lacks a future target) and defines feature columns. It's called internally by `train()`.

#### `train(test_size=0.2)`

Splits the data sequentially into training and testing sets (critical for time-series to prevent data leakage from the future). It then scales the features using `StandardScaler` and trains the `LogisticRegression` model.

-   `test_size` (float): The proportion of the dataset to include in the test split. Defaults to `0.2`.

#### `evaluate()`

Evaluates the trained model on the test set and prints a classification report, confusion matrix, and accuracy score. This method should be called after `train()`.

#### `predict_future(last_n_days=60)`

*Truncated in code, but typically would use the last `n` available data points to predict the next single day's movement after scaling them correctly. Requires careful implementation to avoid using future data.* (Based on typical ML model usage, this function is expected but not fully provided in the diff, thus its description is generalized).

### Example Usage

```python
import pandas as pd
from logistic_regression import StockLogisticModel
from stock import StockFeatureEngineer # Assuming features are pre-engineered

# Create a sample DataFrame or load your stock data
data = {
    'Date': pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 16)]),
    'Open': [100+i for i in range(15)],
    'High': [102+i for i in range(15)],
    'Low': [99+i for i in range(15)],
    'Close': [101+i + (i%3-1) for i in range(15)], # Example fluctuating close
    'Volume': [1000+i*50 for i in range(15)]
}
raw_df = pd.DataFrame(data)

# 1. Feature Engineer the data
fe = StockFeatureEngineer(raw_df)
df_with_features = fe \
    .add_moving_averages(windows=[3, 5]) \
    .add_rsi() \
    .add_macd() \
    .df

# Ensure no NaNs from indicator calculations before training
df_final = df_with_features.dropna()

if not df_final.empty:
    # 2. Instantiate and train the Logistic Regression model
    model = StockLogisticModel(df_final)
    model.train(test_size=0.3) # Use 30% of data for testing

    # 3. Evaluate the model
    classification_report_output = model.evaluate()
    print("\n--- Classification Report ---")
    print(classification_report_output)

    # 4. (Optional) Predict future using `model.predict_future()` if fully implemented
    # prediction = model.predict_future(last_n_days=5)
    # print("\nNext day prediction (1=Up, 0=Down):", prediction)
else:
    print("DataFrame is empty after dropping NaNs. Cannot train model.")
```