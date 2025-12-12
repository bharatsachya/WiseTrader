# Logistic Regression Stock Prediction Model (`logistic_regression.py`)

The `logistic_regression.py` script provides a `StockLogisticModel` class for predicting whether a stock's closing price will go up or down the next day. Unlike deep learning models, this uses a classical machine learning approach, focusing on classification based on engineered features.

## `StockLogisticModel` Class

### `__init__(self, df)`

*   **Purpose**: Initializes the model with a DataFrame containing stock data. It creates a copy of the input DataFrame and initializes the `LogisticRegression` model and `StandardScaler`.
*   **Parameters**:
    *   `df` (pd.DataFrame): The input stock data DataFrame. Expected to contain a `close` column and potentially other technical indicators if `StockFeatureEngineer` is used beforehand.

### `prepare_data(self)`

*   **Purpose**: Prepares the data for logistic regression by creating a binary target variable and aligning features.
*   **Mechanism**:
    1.  **Target Creation**: Adds a `Target` column, where `1` indicates that 'Tomorrow's Close' > 'Today's Close', and `0` otherwise.
    2.  **Dropping NaNs**: Removes the last row which will have a NaN `Target` (as 'tomorrow's price' is unknown).
    3.  **Feature Selection**: Automatically selects columns that are not `target`, `date`, or `Target` as features.
*   **Returns**:
    *   `X` (pd.DataFrame): Feature matrix.
    *   `y` (pd.Series): Target vector.

### `train(self, test_size=0.2)`

*   **Purpose**: Splits the data, scales features, and trains the Logistic Regression model.
*   **Parameters**:
    *   `test_size` (float): The proportion of the dataset to include in the test split (default: 0.2).
*   **Important**: Uses `shuffle=False` for `train_test_split` to prevent data leakage, ensuring that training data always precedes test data chronologically.
*   **Scaling**: `StandardScaler` is applied to features to normalize their magnitudes, which is crucial for Logistic Regression.
*   **Output**: Prints the classification report and accuracy score of the trained model on the test set.

### `predict(self)`

*   **Purpose**: Makes predictions on the `X_test` data (from the training phase) and stores them for evaluation.
*   **Returns**:
    *   `y_pred` (np.array): Predicted target values (0 or 1).

### `evaluate(self)`

*   **Purpose**: Evaluates the model's performance on the test set.
*   **Output**: Prints a `classification_report`, an `accuracy_score`, and displays a `confusion_matrix` heatmap using `seaborn` and `matplotlib`.

## Example Usage Flow

While not included in the provided `logistic_regression.py` snippet, a typical usage would involve:

1.  Loading raw stock data into a DataFrame.
2.  Potentially using `StockFeatureEngineer` to add technical indicators.
3.  Instantiating `StockLogisticModel` with the prepared DataFrame.
4.  Calling `train()`.
5.  Calling `predict()`.
6.  Calling `evaluate()` to see results.

```python
# Assumed usage:
# from stock import StockFeatureEngineer
# from logistic_regression import StockLogisticModel

# 1. Load initial data (replace with your data loading method)
# df_raw = pd.read_csv('your_stock_data.csv')

# 2. Add features (optional but recommended)
# fe = StockFeatureEngineer(df_raw)
# df_features = fe.add_all_features().get_df()

# 3. Initialize and train the model
# model = StockLogisticModel(df_features)
# model.train(test_size=0.2)

# 4. Make predictions
# model.predict()

# 5. Evaluate
# model.evaluate()
```