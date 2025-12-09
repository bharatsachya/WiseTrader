# WiseTrader Stock Prediction Models

WiseTrader now incorporates advanced neural network models for stock price prediction, providing users with powerful tools to analyze market trends. This document details the available models and how to use them within the application.

## Accessing Prediction Models

From the main `WiseTrader` application (`home.py`), use the sidebar navigation to select your desired prediction model under the "**🚀 WiseTrader Pages**" section:

- **Wise CNN**: For Convolutional Neural Network-based predictions.
- **Wise LSTM**: For Long Short-Term Memory network-based predictions.

## 1. Wise CNN (Convolutional Neural Network)

The Wise CNN model leverages Convolutional Neural Networks to predict stock price direction (up or down). It takes historical stock data, preprocesses it, trains a CNN model, and evaluates its accuracy.

### Usage:
1.  Navigate to the "Wise CNN" page using the sidebar.
2.  **Enter Stock Ticker**: Input the stock symbol (e.g., `NV20.NS`) you wish to analyze in the provided text field.
3.  The application will fetch 1 year of hourly historical data via `yfinance`.
4.  The model will then be trained and evaluated.
5.  **Output**: You will see the model's accuracy, along with plots visualizing training and validation accuracy and loss over epochs. This helps in understanding the model's performance on the given stock.

### Model Details:
-   **Input Data**: Open, High, Low, Close, Volume for the specified stock.
-   **Data Scaling**: Uses `MinMaxScaler` to normalize data.
-   **Sequence Length**: Data is prepared into sequences of 10 timesteps for prediction.
-   **Target**: Predicts if the closing price will be higher (1) or lower (0) than the previous day's closing price.
-   **Architecture**: A `Sequential` Keras model with:
    -   `Conv1D` layer (64 filters, kernel size 3, ReLU activation).
    -   `MaxPooling1D` layer (pool size 2).
    -   `Flatten` layer.
    -   Two `Dense` layers (64 units with ReLU, and 1 unit with Sigmoid activation for binary classification).
-   **Training**: Trained using `adam` optimizer, `binary_crossentropy` loss, and `accuracy` metrics over 10 epochs with a batch size of 64.

### Example Output:
```
Fetching data for NV20.NS...
Training the model...
Epoch 1/10
...
Accuracy: 0.7542  (Example Accuracy)
(Plots for Train Accuracy, Validation Accuracy, Train Loss, Validation Loss)
```

## 2. Wise LSTM (Long Short-Term Memory)

The Wise LSTM model utilizes Long Short-Term Memory networks, a type of recurrent neural network particularly effective for sequential data like time series, to predict future stock prices.

### Usage:
1.  Navigate to the "Wise LSTM" page using the sidebar.
2.  **Enter Stock Ticker**: Input the stock symbol (e.g., `AAPL`).
3.  **Select Start Date**: Choose the beginning of your historical data range (e.g., `2010-01-01`).
4.  **Select End Date**: Choose the end of your historical data range (e.g., `2023-01-01`).
5.  The application will fetch the historical 'Close' prices for the specified date range.
6.  The data will be prepared for an LSTM model, including normalization and splitting into training and testing sets. Further details on model training and prediction will be displayed as the project evolves.

### Model Details:
-   **Input Data**: Primarily 'Close' prices for the specified stock and date range.
-   **Data Scaling**: Uses `MinMaxScaler` to normalize 'Close' prices.
-   **Data Split**: Splits data into 80% training and 20% testing sets.
-   **Sequence Generation**: Prepares data into sequences suitable for RNN models.

### Example Input:
-   **Stock Ticker**: `AAPL`
-   **Start Date**: `2010-01-01`
-   **End Date**: `2023-01-01`