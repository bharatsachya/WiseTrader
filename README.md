# WiseTrader: Advanced Stock Price Prediction

WiseTrader is an interactive application designed for stock market analysis and prediction, leveraging various machine learning and deep learning models. This application provides a centralized Streamlit interface to explore different predictive algorithms, enabling users to gain insights into potential stock price movements.

## Features

- **Streamlit Web Application**: An intuitive and interactive web interface to access all predictive models.
- **Multiple Prediction Models**: Explore and utilize a range of algorithms for stock price forecasting:
    - **Convolutional Neural Network (CNN)**: For pattern recognition in time series data.
    - **Long Short-Term Memory (LSTM)**: Specialized recurrent neural network for sequence prediction.
    - **Recurrent Neural Network (RNN)**: General-purpose model for sequential data.
    - **Logistic Regression**: A classical machine learning approach for binary classification (e.g., price up/down).
    - **Random Forest**: (Implied by commit message) An ensemble learning method for robust predictions.
- **Real-time Data Fetching**: Integrates with `yfinance` to fetch live and historical stock data.
- **Stock Watcher**: A basic utility to view live stock data for a given ticker.

## Getting Started

### Prerequisites

To run WiseTrader, you will need Python 3.x and the following libraries:

```bash
pip install streamlit yfinance numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```

### Running the Application

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/bharatsachya/WiseTrader.git
    cd WiseTrader
    ```

2.  **Run the Streamlit application**:
    ```bash
    streamlit run home.py
    ```

    This will open the WiseTrader application in your web browser.

## How to Use

Upon launching the application, you will see the `WiseTrader` homepage. From the sidebar on the left, you can select different prediction models:

-   **Stock Watcher**: Enter a stock ticker (e.g., `NV20.NS`, `AAPL`) to view its current Open, High, Low, Close, and Volume data.
-   **Wise CNN**: Navigate to this page to use the Convolutional Neural Network model for stock prediction. You can input a stock ticker, and the model will provide predictions based on its training.
-   **Wise LSTM**: Select this option to utilize the Long Short-Term Memory model. Here, you can specify a stock symbol and date range for historical data analysis and prediction.
-   **Wise RNN**: (Future/Implied) This section will host a Recurrent Neural Network model for stock analysis.

Each model page will guide you through its specific inputs and display its predictions or analysis results.

## Contact

-   **GitHub**: [bharatsachya/WiseTrader](https://github.com/bharatsachya/WiseTrader)

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests.