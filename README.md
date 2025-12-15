# WiseTrader: All Neural Network Algorithms in One App 📈

WiseTrader is a Streamlit-based application designed to demonstrate and apply various machine learning models, including deep neural networks like LSTM and CNN, as well as classical models like Logistic Regression, for stock price prediction and analysis.

## Features

- **Live Stock Data**: Fetch real-time and historical stock data using `yfinance`.
- **Multiple Models**: Explore stock predictions using:
    - **Convolutional Neural Network (CNN)**
    - **Long Short-Term Memory (LSTM)**
    - **Logistic Regression** (for directional prediction)
- **Interactive Interface**: A user-friendly Streamlit interface for seamless interaction.
- **Technical Feature Engineering**: Utilities to generate various technical indicators for robust model training.

## Getting Started

Follow these instructions to set up and run the WiseTrader application locally.

### Prerequisites

Ensure you have Python 3.8+ installed. The following libraries are required:

```bash
pip install streamlit yfinance numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```

### Installation

1. Clone the WiseTrader repository:
   ```bash
git clone https://github.com/bharatsachya/WiseTrader.git
cd WiseTrader
   ```

2. Install the required Python packages:
   ```bash
pip install -r requirements.txt # (Assuming a requirements.txt will be added, otherwise use the command above)
   ```

### Running the Application

Navigate to the root directory of the cloned repository and run the Streamlit application:

```bash
streamlit run home.py
```

This command will open the WiseTrader application in your default web browser.

## How to Use

The `home.py` script serves as the main entry point for the Streamlit application. Upon launching, you will see a 'Stock Watcher' section showing current stock data for a given ticker.

### Navigating Models

Use the **sidebar** on the left to select different prediction models:

- **Wise CNN**: For Convolutional Neural Network based predictions.
- **Wise RNN**: (Currently under development, placeholder for future RNN models).
- **Wise LSTM**: For Long Short-Term Memory network based predictions.

Each page will allow you to input a stock ticker and visualize the model's predictions.

### Stock Watcher

- **Enter Stock Ticker**: Input any valid stock ticker symbol (e.g., `AAPL`, `NV20.NS`) to view its current open, high, low, close, and volume data.

### Model Specific Sections

#### Wise CNN

This page utilizes a 1D Convolutional Neural Network to predict stock price direction (up/down). You can:
- Enter a stock ticker.
- View the model's training accuracy and loss.
- See a plot comparing actual vs. predicted closing prices (or directional predictions).

#### Wise LSTM

This page employs an LSTM model for forecasting stock closing prices. You can:
- Enter a stock ticker.
- Specify start and end dates for historical data.
- Visualize the actual closing prices against the LSTM model's predictions.

## Underlying Models and Utilities

WiseTrader integrates several backend scripts:

- `home.py`: Main Streamlit application entry point and navigation.
- `pages/wise_cnn.py`: Implements the CNN model for stock prediction.
- `pages/wise_lstm.py`: Implements the LSTM model for stock price forecasting.
- `Lstm.py`: A standalone script demonstrating an LSTM model for stock prediction.
- `logistic_regression.py`: Implements a Logistic Regression model for classifying stock price movement (up/down) using technical indicators.
- `stock.py`: Contains the `StockFeatureEngineer` class for generating various technical indicators such as Moving Averages, RSI, MACD, Bollinger Bands, and Stochastic Oscillator.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new models, or bug fixes, please open an issue or submit a pull request.

