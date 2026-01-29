# WiseTrader Streamlit Application

The `home.py` script serves as the main entry point for the WiseTrader Streamlit application, consolidating various stock analysis and prediction models into a user-friendly interface.

## How to Run

Ensure you have followed the installation steps in the `README.md`. Then, from the project root directory, execute:

```bash
streamlit run home.py
```

This command will launch the application in your default web browser.

## Application Overview

The WiseTrader app is divided into a 'Stock Watcher' section and several 'WiseTrader Pages' for different prediction models.

### Stock Watcher

This section allows you to fetch and view real-time (1-minute interval) stock data for any given ticker symbol.

1.  **Enter Stock Ticker**: Input a stock ticker symbol (e.g., `NV20.NS` for Nifty 20, or `AAPL` for Apple).
2.  The application will display the `Open`, `High`, `Low`, `Close`, and `Volume` data for the selected stock, fetched using `yfinance`.

### WiseTrader Pages

Use the sidebar navigation to select different prediction model pages. Currently, the following pages are available:

*   **Wise CNN**: For Convolutional Neural Network-based stock prediction.
*   **Wise RNN** (Placeholder for future development).
*   **Wise LSTM**: For Long Short-Term Memory-based stock prediction.

#### Wise CNN Page (`pages/wise_cnn.py`)

This page demonstrates stock trend prediction using a 1D Convolutional Neural Network.

1.  **Enter Stock Ticker**: Similar to the Stock Watcher, input a ticker symbol.
2.  The app fetches historical data, preprocesses it (scaling, sequence creation), and trains a CNN model.
3.  It then predicts future trends (up or down based on closing price movement) and displays the model's accuracy, loss, and a plot comparing actual vs. predicted trends.

#### Wise LSTM Page (`pages/wise_lstm.py`)

This page utilizes a Long Short-Term Memory network for stock price regression prediction.

1.  **Enter Stock Ticker**, **Start Date**, **End Date**: Define the stock and the historical period for analysis.
2.  The system fetches data, normalizes it, and trains an LSTM model to predict 'Close' prices.
3.  After training, it performs predictions on a test set and displays a plot comparing the actual closing prices with the model's predictions.

By navigating these pages, users can explore different algorithmic approaches to stock market analysis and prediction.