# WiseTrader: Stock Price Prediction with Neural Networks

WiseTrader is a Streamlit-based application that provides various neural network and machine learning models for stock price prediction and analysis. This project aims to bring different predictive algorithms into one interactive platform.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Available Models](#available-models)
5. [Contributing](#contributing)
6. [Contact](#contact)

## Features
- **Multiple Prediction Models:** Utilize CNN, LSTM, and other machine learning algorithms for stock forecasting.
- **Interactive Streamlit UI:** User-friendly interface to select stocks, view data, and run predictions.
- **Real-time Data Fetching:** Integrates with `yfinance` to fetch up-to-date stock data.
- **Technical Feature Engineering:** Includes utilities to automatically generate common technical indicators.

## Installation
Follow these steps to set up the WiseTrader project locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bharatsachya/WiseTrader.git
    cd WiseTrader
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt  # Assuming requirements.txt exists and contains all dependencies
    ```
    _Note: If `requirements.txt` is not provided, you might need to install `streamlit`, `yfinance`, `numpy`, `pandas`, `scikit-learn`, `tensorflow` (or `keras`), `matplotlib`, `seaborn` manually._

## Usage
To run the WiseTrader Streamlit application, execute the following command from the project root directory:

```bash
streamlit run home.py
```

Once the application starts, it will open in your web browser. You can then:
-   Enter a stock ticker symbol.
-   View stock data.
-   Select different prediction models (Wise CNN, Wise LSTM in the sidebar) to see their predictions.

## Available Models

### Wise CNN
Located in `pages/wise_cnn.py`, this model uses a Convolutional Neural Network (CNN) to predict stock price movements.

### Wise LSTM
Located in `pages/wise_lstm.py`, this model uses a Long Short-Term Memory (LSTM) network to predict stock prices.

### Logistic Regression
Implemented in `logistic_regression.py`, this script provides a classification model to predict whether a stock's price will go up or down the next day.

### Raw LSTM Script
The file `Lstm.py` contains a standalone implementation of an LSTM model for stock price prediction, useful for understanding the core logic outside the Streamlit app.

## Contributing
We welcome contributions! Please refer to the [GitHub repository](https://github.com/bharatsachya/WiseTrader) for guidelines on how to contribute.
