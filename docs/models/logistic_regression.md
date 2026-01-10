# Logistic Regression Model

The `StockLogisticModel` provides a basic, yet robust, approach to predict stock price direction using logistic regression. This model is primarily used for binary classification, determining whether a stock's closing price will increase or decrease the next day based on various technical indicators.

## Purpose

To predict if the next day's closing price will be higher than the current day's closing price (Target = 1) or not (Target = 0).

## Implementation Details

*   **Location**: `logistic_regression.py`
*   **Class**: `StockLogisticModel`
*   **Initialization**: Takes a pandas DataFrame with stock data (expected to include 'close' and optionally other features).
*   **`prepare_data()` method**:
    *   Creates a `Target` column: 1 if `close.shift(-1) > close`, else 0.
    *   Drops the last row where the target cannot be determined.
    *   Identifies feature columns, excluding 'target', 'date', and 'Target'.
*   **`train()` method**:
    *   Splits the data into training and testing sets using `train_test_split` with `shuffle=False` to prevent data leakage (looking into the future).
    *   Scales features using `StandardScaler`.
    *   Trains a `LogisticRegression` model.
*   **`evaluate()` method**: Provides a classification report, confusion matrix, and accuracy score.
*   **`plot_confusion_matrix()` method**: Visualizes the confusion matrix using `seaborn`.

## Example Usage (`logistic_regression.py`)

```python
import pandas as pd
from logistic_regression import StockLogisticModel
from stock import StockFeatureEngineer # Assuming StockFeatureEngineer is available

# --- 1. Prepare some dummy data or load real data ---
# For a real scenario, you would load data and apply feature engineering
data = {
    'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100)),
    'open': np.random.rand(100) * 100 + 50,
    'high': np.random.rand(100) * 105 + 50,
    'low': np.random.rand(100) * 95 + 50,
    'close': np.random.rand(100) * 100 + 50,
    'volume': np.random.randint(100000, 1000000, 100)
}
df_raw = pd.DataFrame(data)

# --- 2. Apply Feature Engineering ---
# Ensure the 'stock' module is accessible
feature_engineer = StockFeatureEngineer(df_raw)
feature_df = feature_engineer.add_moving_averages().add_rsi().add_macd().get_dataframe()

# --- 3. Initialize and Train the Logistic Regression Model ---
# Ensure your DataFrame has a 'close' column in lowercase per StockLogisticModel's expectation
model = StockLogisticModel(feature_df.rename(columns={'Close': 'close'}))
model.train(test_size=0.2)

# --- 4. Evaluate the Model ---
model.evaluate()
# model.plot_confusion_matrix()
print("\nClassification Report:")
print(model.y_pred)
```
