This file (`pages/wise_cnn.py`) implements the Streamlit page for stock movement prediction (up/down) using a Convolutional Neural Network (CNN). When accessed through `home.py`, users can:

1.  **Enter Stock Ticker**: Provide a stock symbol (e.g., 'NV20.NS').
2.  **Data Processing**: The page fetches 1-year hourly stock data, preprocesses it, and creates sequences for CNN input. It defines a binary target: 1 if the closing price increased, 0 otherwise.
3.  **CNN Model**: A `Conv1D` model is built, compiled, and trained to predict this binary outcome (stock price movement).
4.  **Visualize Results**: (Implied from truncated content, typically showing model performance metrics or predictions).