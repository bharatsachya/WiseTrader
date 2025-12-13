# WiseTrader: All Neural Network Algorithms in One App 📈

WiseTrader is a Streamlit-powered application designed to provide various stock price prediction models, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Logistic Regression. This application consolidates multiple sophisticated algorithms into a single, easy-to-use interface, helping users to analyze and predict stock movements.

## Features

- **Multiple Prediction Models**: Choose from:
    - **Wise CNN**: Utilizes Convolutional Neural Networks for short-term stock trend prediction.
    - **Wise LSTM**: Employs Long Short-Term Memory networks, ideal for time-series forecasting of stock prices.
    - **Wise RNN** (Planned/Placeholder): Placeholder for future Recurrent Neural Network implementations.
    - **Random Forest** (Implemented per commit, details pending in app):
    - **Logistic Regression**: A classical machine learning model adapted for stock price movement prediction.
- **Interactive Streamlit Interface**: Easily switch between different models and input stock tickers directly within the web application.
- **Real-time Data Fetching**: Integrates with `yfinance` to fetch live stock data.
- **Customizable Inputs**: Adjust stock tickers and date ranges for predictions.

## Getting Started

To run the WiseTrader application locally, follow these steps:

### Prerequisites

Ensure you have Python 3.x installed. You will also need `pip` for package management.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bharatsachya/WiseTrader.git
   cd WiseTrader
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt # (Assuming a requirements.txt file will be created)
   ```
   *Note: If `requirements.txt` does not exist, you will need to install the following packages manually: `streamlit`, `yfinance`, `numpy`, `pandas`, `scikit-learn`, `tensorflow` (for LSTM/CNN), `matplotlib`, `seaborn`.*

### Running the Application

Once the dependencies are installed, you can run the Streamlit application from the root directory:

```bash
streamlit run home.py
```

This command will open the WiseTrader application in your web browser.

## How to Use

1.  **Navigate Pages**: Use the sidebar dropdown menu (labeled "Select Page") to switch between "Wise CNN", "Wise LSTM", and "Wise RNN" (and potentially other models like "Random Forest" or "Logistic Regression" if integrated into the Streamlit app).
2.  **Enter Stock Ticker**: On each model's page, enter a valid stock ticker symbol (e.g., `AAPL`, `NV20.NS`) in the input box.
3.  **View Predictions**: The application will fetch data, run the chosen model, and display the analysis or predictions based on the selected algorithm.

## Project Structure

- `home.py`: The main Streamlit application entry point.
- `Lstm.py`: Standalone LSTM stock prediction script (for `README`, it is better to say `pages/wise_lstm.py` is the one that's used).
- `logistic_regression.py`: Implements Logistic Regression for stock movement prediction.
- `stock.py`: Contains utility classes for stock feature engineering (e.g., calculating technical indicators like Moving Averages, RSI, MACD).
- `pages/wise_cnn.py`: Streamlit page implementation for CNN-based stock prediction.
- `pages/wise_lstm.py`: Streamlit page implementation for LSTM-based stock prediction.

## Contributing

We welcome contributions to WiseTrader! If you have suggestions or want to improve the models, please feel free to fork the repository and submit a pull request.

## Contact

- **GitHub Repository**: [bharatsachya/WiseTrader](https://github.com/bharatsachya/WiseTrader)
