# Usage Guide

This guide explains how to run the WiseTrader Streamlit application and interact with its features.

## Running the Application

1.  **Activate your virtual environment** (if you created one during installation):
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

2.  **Navigate to the project root directory** (where `home.py` is located).

3.  **Run the Streamlit application:**
    ```bash
    streamlit run home.py
    ```

    This command will open the WiseTrader application in your default web browser.

## Navigating the Application

The WiseTrader application has a main page and a sidebar for navigation.

### Home Page (Stock Watcher)

Upon launching, you'll see the 'Stock Watcher' page. Here you can:

*   **Enter Stock Ticker**: Type a stock symbol (e.g., `AAPL`, `NV20.NS`) into the text input field to fetch its live data (1-minute intervals for the current day).
*   **View Live Data**: The 'Open', 'High', 'Low', 'Close', and 'Volume' data for the entered ticker will be displayed.

### Sidebar Navigation

Use the sidebar on the left to select different prediction pages:

*   **Wise CNN**: Navigate to this page to use the Convolutional Neural Network model for stock movement prediction.
*   **Wise LSTM**: Navigate to this page to use the Long Short-Term Memory model for stock price forecasting.

### Common Interactions

*   **Stock Ticker Input**: Most pages will have an input for the stock ticker symbol. Ensure you use valid symbols (e.g., `AAPL` for Apple, `RELIANCE.NS` for Reliance on NSE).
*   **Date Inputs**: For models requiring historical data, you might find start and end date pickers to customize the data range.

## Contributing

If you're interested in contributing to WiseTrader, please refer to the [Contribution Guide](#) (if available).
