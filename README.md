# WiseTrader: All Neural Network Algorithms in One App

WiseTrader is a Streamlit-based application designed to provide stock price prediction using various neural network and machine learning algorithms. It includes implementations of Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Logistic Regression, along with a robust feature engineering utility.

## Table of Contents
1. [Features](#features)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the Application](#running-the-application)
3. [Models Overview](#models-overview)
   - [Wise CNN](#wise-cnn)
   - [Wise LSTM](#wise-lstm)
   - [Logistic Regression](#logistic-regression)
   - [Feature Engineering (`StockFeatureEngineer`)](#feature-engineering-stockfeatureengineer)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [Contact](#contact)

## Features
- **Multiple Prediction Models**: Integrates CNN, LSTM, and Logistic Regression for diverse prediction strategies.
- **Interactive UI**: User-friendly interface built with Streamlit for easy navigation and interaction.
- **Real-time Stock Data**: Fetches stock data using `yfinance`.
- **Comprehensive Feature Engineering**: Includes a utility to generate various technical indicators.

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bharatsachya/WiseTrader.git
   cd WiseTrader
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   (The `requirements.txt` file should be created based on the imports in the code, typical packages include `streamlit`, `yfinance`, `numpy`, `pandas`, `scikit-learn`, `tensorflow`/`keras`, `matplotlib`, `seaborn`.)

### Running the Application
To start the WiseTrader application, navigate to the project root directory and run:
```bash
streamlit run home.py
```
This will open the application in your default web browser.

## Models Overview

### Wise CNN
Located in `pages/wise_cnn.py`, this page utilizes a Convolutional Neural Network (CNN) for stock price prediction. The model takes sequences of historical OHLCV (Open, High, Low, Close, Volume) data to predict a binary outcome (price increase or decrease).

**Key components:**
- Data fetching via `yfinance`.
- Data normalization using `MinMaxScaler`.
- Sequence creation for CNN input.
- `Conv1D`, `MaxPooling1D`, `Flatten`, and `Dense` layers in the neural network architecture.
- Binary classification with `sigmoid` activation and `binary_crossentropy` loss.

### Wise LSTM
Located in `pages/wise_lstm.py`, this section implements a Long Short-Term Memory (LSTM) network to predict future stock closing prices. It processes sequential data to capture time-series dependencies.

**Key components:**
- Data fetching via `yfinance` based on user-defined symbol and dates.
- Data normalization of 'Close' prices.
- Sequential data preparation using `create_sequences`.
- `LSTM` layers followed by `Dense` layers for regression.
- Model compilation with `adam` optimizer and `mean_squared_error` loss.

### Logistic Regression
The `logistic_regression.py` file defines a `StockLogisticModel` class that uses Logistic Regression for binary classification of stock price movement (up or down the next day). This model leverages various technical indicators as features.

**Key features:**
- **Target Creation**: Generates a binary target: 1 if tomorrow's close > today's close, else 0.
- **Sequential Splitting**: Ensures data is split into train/test sequentially to prevent data leakage.
- **Feature Scaling**: Uses `StandardScaler` to normalize features.
- **Prediction & Evaluation**: Provides methods for training, prediction, and visualizing results with confusion matrices.

### Feature Engineering (`StockFeatureEngineer`)
The `stock.py` file contains the `StockFeatureEngineer` class, a utility designed to generate a wide array of technical indicators from raw stock data. It enhances the dataset, making it suitable for training machine learning models like Logistic Regression.

**Available Indicators:**
- **Moving Averages (SMA, EMA)**: Simple and Exponential Moving Averages for various windows (`add_moving_averages`).
- **Relative Strength Index (RSI)**: Measures the speed and change of price movements (`add_rsi`).
- **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator (`add_macd`).
- **Stochastic Oscillator**: Price change relative to high/low range over a period (`add_stochastic_oscillator`).
- **Bollinger Bands**: Volatility bands around a moving average (`add_bollinger_bands`).
- **Average True Range (ATR)**: Measures market volatility (`add_atr`).
- **On-Balance Volume (OBV)**: Relates volume to price change (`add_obv`).
- **Lagged Features**: Creates lagged versions of 'close' price for time-series context (`add_lagged_features`).

## Project Structure
```
WiseTrader/
├── home.py                 # Main Streamlit application entry point
├── Lstm.py                 # Standalone LSTM model implementation (legacy/example)
├── logistic_regression.py  # Logistic Regression model class
├── stock.py                # StockFeatureEngineer utility for technical indicators
├── pages/
│   ├── wise_cnn.py         # Streamlit page for CNN stock prediction
│   └── wise_lstm.py        # Streamlit page for LSTM stock prediction
└── ...                     # Other project files (e.g., requirements.txt, .gitignore)
```

## Contributing
We welcome contributions to WiseTrader! Please feel free to open issues or submit pull requests.

## Contact
- **GitHub**: [bharatsachya/WiseTrader](https://github.com/bharatsachya/WiseTrader)
