# WiseTrader 📈

WiseTrader is an advanced stock market prediction tool that leverages various machine learning and deep learning algorithms to analyze and forecast stock prices.

## ✨ Features

- **Multiple Prediction Models**: Explore stock price predictions using a suite of algorithms:
    - **Long Short-Term Memory (LSTM)**: A type of recurrent neural network (RNN) well-suited for time-series forecasting.
    - **Convolutional Neural Network (CNN)**: Utilizes 1D convolutional layers for pattern recognition in stock data.
    - **Logistic Regression**: A classical machine learning model adapted for binary classification (e.g., predicting if a stock price will go up or down).
- **Stock Feature Engineering**: Includes a robust `StockFeatureEngineer` class to generate a comprehensive set of technical indicators (Moving Averages, RSI, MACD, Bollinger Bands, etc.) for enriched data analysis.
- **Interactive Streamlit Application**: A user-friendly web interface to interact with the models, fetch real-time stock data, and visualize predictions directly in your browser.

## 🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow/Keras
- scikit-learn
- yfinance
- pandas, numpy, matplotlib, seaborn

## 🚀 Getting Started

To get started with WiseTrader locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/bharatsachya/WiseTrader.git
cd WiseTrader
```

### 2. Install Dependencies

Make sure you have Python (3.7+) installed. Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt # (Assuming a requirements.txt will be added with: streamlit, yfinance, tensorflow, scikit-learn, pandas, numpy, matplotlib, seaborn)
```

### 3. Run the Streamlit Application

Navigate to the root directory of the project and run the `home.py` application:

```bash
streamlit run home.py
```

This will open the WiseTrader application in your web browser, where you can select different prediction models and input stock tickers.

## 📚 Project Structure

- `home.py`: The entry point for the Streamlit web application.
- `Lstm.py`: Standalone script demonstrating LSTM model for stock prediction.
- `logistic_regression.py`: Implements Logistic Regression for stock movement prediction.
- `stock.py`: Contains the `StockFeatureEngineer` class for generating technical indicators.
- `pages/wise_cnn.py`: Streamlit page implementation for the CNN model.
- `pages/wise_lstm.py`: Streamlit page implementation for the LSTM model.

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.
