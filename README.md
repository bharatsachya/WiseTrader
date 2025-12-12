# WiseTrader

WiseTrader is an application designed to demonstrate and implement various machine learning and deep learning models for stock price prediction and analysis. It features a Streamlit-based user interface that allows users to interactively analyze stock data and run different prediction algorithms.

## Features

*   **Stock Watcher**: Real-time stock data fetching and display.
*   **Neural Network Models**: Implementations of CNN, LSTM, and potentially RNN for stock price trend prediction.
*   **Classical ML Models**: Includes Logistic Regression for predicting stock movement.
*   **Feature Engineering**: Utility to generate technical indicators for comprehensive analysis.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/bharatsachya/WiseTrader.git
    cd WiseTrader
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt # (assuming a requirements.txt will be created)
    # Alternatively, install manually:
    # pip install streamlit yfinance numpy pandas scikit-learn tensorflow keras matplotlib seaborn
    ```

### Running the Application

To start the WiseTrader Streamlit application, navigate to the project root and run:

```bash
streamlit run home.py
```

This will open the application in your web browser. You can then interact with the different stock prediction pages.

## Project Structure

*   `home.py`: The main Streamlit application entry point.
*   `pages/`: Contains individual Streamlit pages for different models (e.g., `wise_cnn.py`, `wise_lstm.py`).
*   `Lstm.py`: Standalone script demonstrating an LSTM model for stock prediction.
*   `logistic_regression.py`: Standalone script for Logistic Regression based stock prediction.
*   `stock.py`: Utility for stock feature engineering.

## Contributing

Contributions are welcome! Please refer to the contribution guidelines.