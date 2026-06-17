# Stock Price Prediction with Logistic Regression

**File:** `logistic_regression.py`

The `StockLogisticModel` class provides an implementation of Logistic Regression for predicting binary stock price movements – specifically, whether tomorrow's closing price will be higher than today's. It leverages technical indicators as features.

## Class: `StockLogisticModel`

```python
class StockLogisticModel:
    def __init__(self, df):
        self.df = df.copy()
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def prepare_data(self):
        # ... (implementation)

    def train(self, test_size=0.2):
        # ... (implementation)

    def evaluate(self):
        # ... (implementation)

    def plot_confusion_matrix(self):
        # ... (implementation)

    def predict_future(self, last_n_days):
        # ... (implementation)
```

### `__init__(self, df)`
-   **Purpose:** Initializes the Logistic Regression model with historical stock data.
-   **Parameters:**
    -   `df` (pandas.DataFrame): The input DataFrame which *should already contain engineered features* (e.g., from `StockFeatureEngineer`). It must contain a 'close' column.

### `prepare_data(self)`
-   **Purpose:** Prepares the data for training by creating a binary target variable ('Target') and selecting feature columns.
-   **Details:** The 'Target' is 1 if `Tomorrow's Close > Today's Close`, otherwise 0. It expects the DataFrame to have a 'close' column and automatically identifies feature columns, excluding 'target', 'date', and 'Target'.
-   **Returns:**
    -   `X` (pandas.DataFrame): Feature matrix.
    -   `y` (pandas.Series): Target vector.

### `train(self, test_size=0.2)`
-   **Purpose:** Splits the data into training and testing sets *sequentially* (to prevent data leakage) and trains the Logistic Regression model.
-   **Parameters:**
    -   `test_size` (float, optional): The proportion of the dataset to include in the test split. Defaults to `0.2`.

### `evaluate(self)`
-   **Purpose:** Evaluates the trained model using a classification report, confusion matrix, and accuracy score.
-   **Returns:**
    -   `dict`: A dictionary containing 'classification_report', 'confusion_matrix', and 'accuracy_score'.

### `plot_confusion_matrix(self)`
-   **Purpose:** Generates and displays a heatmap of the confusion matrix for the model's predictions on the test set.

### `predict_future(self, last_n_days)`
-   **Purpose:** Predicts the next day's price movement (up/down) based on the most recent `last_n_days` data points.
-   **Parameters:**
    -   `last_n_days` (int): The number of recent days to use for making a prediction.
-   **Returns:** `int`: The predicted target (0 for down, 1 for up).

## Usage Example

```python
import yfinance as yf
import pandas as pd
from stock import StockFeatureEngineer
from logistic_regression import StockLogisticModel

# 1. Fetch raw stock data
df_raw = yf.download('MSFT', start='2020-01-01', end='2023-01-01')
df_raw.reset_index(inplace=True)

# 2. Engineer features using StockFeatureEngineer
engineer = StockFeatureEngineer(df_raw)
# Ensure 'Date' is handled if not already by yfinance default index
if 'Date' in df_raw.columns: del df_raw['Date']
engineered_df = engineer.add_moving_averages([10, 30]) \
                        .add_rsi() \
                        .add_macd() \
                        .add_bollinger_bands() \
                        .add_volume_indicators() \
                        .dataframe()

# 3. Instantiate and train the Logistic Regression model
model = StockLogisticModel(engineered_df)
model.train(test_size=0.2)

# 4. Evaluate the model
metrics = model.evaluate()
print("Classification Report:\n", metrics['classification_report'])
print("Confusion Matrix:\n", metrics['confusion_matrix'])
print("Accuracy:\n", metrics['accuracy_score'])

# 5. Plot Confusion Matrix
model.plot_confusion_matrix()

# 6. Predict the next day's movement
# Ensure enough data for prediction (e.g., last 30 days must have features)
if len(engineered_df) > 30:
    last_30_days = 30 # example window
    future_pred = model.predict_future(last_30_days)
    print(f"Predicted next day movement (0=down, 1=up): {future_pred}")
else:
    print("Not enough data to make a future prediction.")
```