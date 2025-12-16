# Standalone LSTM Stock Price Prediction Model (`Lstm.py`)

This script (`Lstm.py`) demonstrates a basic Long Short-Term Memory (LSTM) neural network for predicting stock closing prices. It is a standalone example, separate from the Streamlit application pages, showcasing the core LSTM implementation steps.

## Overview

The script performs the following key steps:

1.  **Data Loading**: Fetches historical stock data using `yfinance`.
2.  **Data Normalization**: Scales the closing prices to a 0-1 range using `MinMaxScaler`.
3.  **Sequence Creation**: Transforms the time-series data into sequences suitable for LSTM input.
4.  **Model Building**: Defines a sequential LSTM model with multiple LSTM layers and a Dense output layer.
5.  **Model Training**: Trains the model on the prepared historical data.
6.  **Prediction**: Generates price predictions on a held-out test set.
7.  **Result Plotting**: Visualizes the actual vs. predicted closing prices.

## Usage

To run this script, execute it directly:

```bash
python Lstm.py
```

You can modify the `ticker` variable within the script to analyze different stocks and adjust `start` and `end` dates.

## Model Details

### Data Preparation

*   **Ticker**: Configurable (default `AAPL`).
*   **Look Back Window**: `60` days (each input sequence to the LSTM contains 60 previous closing prices).
*   **Normalization**: `MinMaxScaler` applied to 'Close' prices.
*   **Train/Test Split**: 80% for training, 20% for testing (sequential split).

### LSTM Architecture

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1))) # First LSTM layer
model.add(LSTM(50)) # Second LSTM layer
model.add(Dense(1)) # Output layer for single price prediction
model.compile(loss="mean_squared_error", optimizer="adam")
```

*   Two LSTM layers, each with 50 units.
*   `return_sequences=True` for the first LSTM layer ensures that the output is a sequence, allowing chaining with another LSTM layer.
*   The final `Dense` layer outputs a single value, representing the predicted scaled closing price.
*   **Loss Function**: `mean_squared_error` (MSE), suitable for regression tasks.
*   **Optimizer**: `adam`.

### Training

*   **Epochs**: `20`
*   **Batch Size**: `32`
*   **Verbosity**: `1` (shows progress during training).

## Output

The script generates a plot titled "Stock Price Prediction" showing the actual closing prices from the test set against the prices predicted by the LSTM model.