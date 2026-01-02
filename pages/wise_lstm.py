This file (`pages/wise_lstm.py`) contains the Streamlit page for stock price prediction using a Long Short-Term Memory (LSTM) neural network. When run via `home.py`, this page allows users to:

1.  **Enter Stock Ticker**: Input a stock symbol (e.g., 'AAPL').
2.  **Define Date Range**: Specify a start and end date for historical data retrieval.
3.  **Model Training & Prediction**: The page fetches historical 'Close' price data using `yfinance`, normalizes it, splits it into training/testing sets, and then trains and evaluates an LSTM model. The model predicts future stock prices based on the provided sequence length.
4.  **Visualize Results**: (Implied from truncated content, usually includes plotting actual vs. predicted prices).