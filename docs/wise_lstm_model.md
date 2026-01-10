# Wise LSTM Model for Stock Prediction

The 'Wise LSTM' page in the WiseTrader application utilizes a Long Short-Term Memory (LSTM) neural network for predicting future stock prices based on historical 'Close' prices. LSTMs are particularly well-suited for time-series data like stock prices due to their ability to capture long-term dependencies.

## How to Use

1.  **Navigate to 'Wise LSTM'**: Select 'Wise LSTM' from the sidebar navigation within the WiseTrader application.
2.  **Enter Stock Ticker**: Input the ticker symbol of the desired stock (e.g., `AAPL`, `GOOGL`).
3.  **Specify Date Range**: Select a `Start Date` and `End Date` for the historical data you wish to use for prediction. The default range is from 2010-01-01 to 2023-01-01.
4.  **Data Processing**: The application will fetch the historical stock data from Yahoo Finance for the specified symbol and date range. It then preprocesses the 'Close' prices by normalizing them.
5.  **Model (Partial Implementation)**: As of this version, the LSTM model setup and data preparation are implemented. Further steps for model training, prediction, and visualization are under development.

## Model Details (Current Implementation Focus)

### Data Fetching and Selection

*   **Source**: Stock data is fetched using `yfinance`.
*   **Features**: Only the 'Close' prices are selected for the LSTM model, as this is a common approach for univariate time series forecasting.

### Data Normalization

*   **Scaler**: `MinMaxScaler` is used to normalize the 'Close' prices. Normalization helps improve the performance and convergence of neural networks.

### Data Splitting

*   The data is split into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data.

## Current Status & Future Enhancements

This page currently sets up the foundational steps for an LSTM stock prediction model, including data input, fetching, normalization, and splitting. Future updates will incorporate the full LSTM model definition, training, and visualization of predictions.

## Example Usage

To begin, simply type your desired stock ticker (e.g., `MSFT`) into the 'Enter Stock Ticker' box and adjust the 'Start Date' and 'End Date' as needed. The current implementation will prepare the data accordingly.

```python
# Example of how to interact in Streamlit (Conceptual)
import streamlit as st
import pandas as pd
import yfinance as yf

# ... (Code snippet for text input and date input in Streamlit)
# (This code is handled internally by the Streamlit app)
```