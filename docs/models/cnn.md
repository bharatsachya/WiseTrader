# Wise CNN Model

The Wise CNN (Convolutional Neural Network) model is implemented for binary classification of stock price movements. Instead of predicting the exact price, it predicts whether the stock's closing price will go up or down relative to a previous point.

## Purpose

This model aims to classify if the stock price will increase (1) or decrease (0) after a given sequence length.

## Implementation Details

*   **Location**: `pages/wise_cnn.py`
*   **Data Source**: Fetches historical stock data using `yfinance` (1-year period, 1-hour interval).
*   **Preprocessing**: Uses `MinMaxScaler` to normalize 'Open', 'High', 'Low', 'Close', 'Volume' features.
*   **Sequence Creation**: Data is formatted into sequences of `sequence_length` (e.g., 10) for input, with the target `y` being 1 if the closing price after the sequence increases, else 0.
*   **Model Architecture:
    *   Conv1D Layer: `Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5), padding='same')`
    *   MaxPooling1D Layer: `MaxPooling1D(pool_size=2)`
    *   Flatten Layer
    *   Dense Hidden Layer: `Dense(64, activation='relu')`
    *   Output Layer: `Dense(1, activation='sigmoid')` for binary classification.
*   **Compilation**: Uses `adam` optimizer and `binary_crossentropy` loss function, with `accuracy` as a metric.
*   **Training**: Trained on 80% of the data for 10 epochs.
*   **Prediction**: Outputs probabilities which are then classified (e.g., >0.5 for 'up').

## How to Use (via Streamlit)

1.  Run the WiseTrader application as described in the [Usage Guide](usage.md).
2.  Select 'Wise CNN' from the sidebar navigation.
3.  **Enter Stock Ticker**: Input the ticker symbol (e.g., `NV20.NS`) for which you want to predict movement.
4.  The application will fetch data, train the CNN model, make predictions, and display training history and a classification report.

## Code Snippet (Key Model Definition from `pages/wise_cnn.py`)

```python
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def display_page():
    # ... (data fetching and preprocessing)

    sequence_length = 10
    X = np.array([scaled_data[i:i+sequence_length] for i in range(len(scaled_data) - sequence_length)])
    closing_prices = scaled_data[sequence_length:, 3]
    y = np.where(closing_prices > scaled_data[:-sequence_length, 3], 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5), padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0, validation_split=0.2)
    # ... (prediction and results display)
```
