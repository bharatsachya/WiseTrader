# WiseTrader: A Stock Prediction Platform

WiseTrader is a comprehensive platform that leverages various neural network algorithms and machine learning models for stock price prediction and analysis. This application provides different pages to explore predictions using models like Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and traditional Logistic Regression.

## Features

- **Stock Watcher**: Real-time stock data fetching and display.
- **Wise CNN**: Predict stock movements using a Convolutional Neural Network model.
- **Wise LSTM**: Utilize Long Short-Term Memory networks for time-series stock prediction.
- **Wise RNN**: (Planned/Placeholder, functionality to be integrated via `home.py`)
- **Feature Engineering**: A utility to generate various technical indicators for stock data.
- **Logistic Regression Model**: A traditional machine learning model for predicting stock direction.

## Getting Started

### 1. Installation

To run WiseTrader locally, you'll need Python 3.x installed. First, clone the repository:

```bash
git clone https://github.com/bharatsachya/WiseTrader.git
cd WiseTrader
```

Then, install the required Python packages. It's highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file is assumed to be present with all necessary dependencies like `streamlit`, `yfinance`, `numpy`, `pandas`, `scikit-learn`, `tensorflow`/`keras`, `matplotlib`, `seaborn`.)*

### 2. Usage

WiseTrader is built with Streamlit, making it easy to run and interact with the application.

To start the application, navigate to the project root directory and run:

```bash
streamlit run home.py
```

This command will open the WiseTrader application in your web browser, typically at `http://localhost:8501`.

### 3. Navigating the Application

Once the application is running:

- **Stock Watcher (Home Page)**: Enter a stock ticker (e.g., `AAPL`, `NV20.NS`) to view its current 1-minute interval data.
- **Sidebar Navigation**: Use the sidebar on the left to select different prediction models:
    - **Wise CNN**: Access the Convolutional Neural Network prediction page.
    - **Wise LSTM**: Access the Long Short-Term Memory prediction page.
    - **Wise RNN**: (A placeholder; functionality will be similar to CNN/LSTM pages once implemented and linked).

### 4. Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

---

## Project Structure Overview

- `home.py`: The main Streamlit application entry point.
- `pages/wise_cnn.py`: Implements the Streamlit page for CNN-based stock prediction.
- `pages/wise_lstm.py`: Implements the Streamlit page for LSTM-based stock prediction.
- `Lstm.py`: A standalone script demonstrating a basic LSTM model without Streamlit integration.
- `logistic_regression.py`: A standalone class for stock prediction using Logistic Regression.
- `stock.py`: A utility class for generating technical indicators and features from stock data.