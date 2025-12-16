# LSTM Stock Price Prediction

This project implements an advanced stock price prediction system using a Long Short-Term Memory (LSTM) neural network. It incorporates various technical indicators, a robust time-series data windowing approach, and a walk-forward validation strategy to provide realistic performance evaluation.

## Features

- **Data Acquisition**: Fetches historical stock data directly from Yahoo Finance.
- **Technical Indicators**: Integrates popular indicators like RSI, MACD, and EMAs to enrich the dataset.
- **Data Preprocessing**: Scales data using `MinMaxScaler` for optimal model performance.
- **Time-Series Windowing**: Transforms sequential data into a format suitable for LSTM networks.
- **LSTM Model**: Utilizes a deep learning LSTM architecture for capturing complex temporal dependencies in stock prices.
- **Walk-Forward Validation**: Evaluates model performance iteratively on new data, mimicking a real trading environment.
- **Backtesting**: Provides a simple trading signal backtest to assess the practical utility of the predictions.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. The project dependencies are managed via `pip`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: A `requirements.txt` file needs to be created, listing `yfinance`, `numpy`, `pandas`, `ta`, `matplotlib`, `scikit-learn`, `tensorflow`)

### Usage

To run the stock prediction model, execute the `lsttm.py` script:

```bash
python lsttm.py
```

The script will download data for the configured `SYMBOL` (default: AAPL), train the LSTM model, perform walk-forward validation, and display results including MSE and a trading signal plot.

## Configuration

You can modify the following parameters within `lsttm.py`:

-   `SYMBOL`: Stock ticker symbol (e.g., "AAPL", "MSFT")
-   `START_DATE`: Historical data start date (e.g., "2014-01-01")
-   `LOOKBACK`: Number of past days to consider for prediction (sequence length)
-   `EPOCHS`: Number of training epochs
-   `BATCH_SIZE`: Training batch size

## Project Structure

```
. 
├── lsttm.py              # Main script for data, model, and backtesting
├── README.md             # Project overview and getting started guide
└── docs/
    └── guides/
        └── stock_prediction.md # Detailed guide on the LSTM model and methodology
```
