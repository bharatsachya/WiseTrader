# Neural Network Based Stock Prediction Models

This section details the various neural network models implemented in WiseTrader for stock price prediction, including Long Short-Term Memory (LSTM) networks and Convolutional Neural Networks (CNNs). Both standalone scripts and their integration into the Streamlit application are covered.

## 1. Long Short-Term Memory (LSTM) Model

LSTM networks are particularly well-suited for sequence prediction problems because they can learn long-term dependencies between time steps in sequence data.

### 1.1 Standalone LSTM Script (`Lstm.py`)

This file provides a comprehensive, runnable script demonstrating the implementation of an LSTM model for stock price prediction using `yfinance` data.

**Key Steps:**
1.  **Load Stock Data:** Fetches historical 'Close' prices for a specified `ticker` (e.g., 'AAPL').
2.  **Normalize Data:** Scales the closing prices to a range of 0-1 using `MinMaxScaler`.
3.  **Create Sequence Data:** Transforms the time-series data into sequences with a `look_back` window (default 60 days) for LSTM input.
4.  **Build LSTM Model:** Constructs a `Sequential` Keras model with LSTM and `Dense` layers.
5.  **Train Model:** Trains the model on the prepared training data.
6.  **Predict & Plot:** Makes predictions on the test set, inverse-transforms them, and plots actual vs. predicted prices.

**Usage:**
To run this script directly, execute:

```bash
python Lstm.py
```

You can modify the `ticker`, `start` and `end` dates within the script.

**Example Code Snippet (Model Definition):**
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
```

### 1.2 LSTM Integration in Streamlit (`pages/wise_lstm.py`)

The `pages/wise_lstm.py` file integrates an LSTM model into the WiseTrader Streamlit application. Users can select this page from the sidebar to interact with the LSTM predictor.

**`display_page()` function overview:**
-   Takes `stock_symbol`, `start_date`, and `end_date` as user inputs.
-   Fetches data using `yf.download`.
-   Normalizes 'Close' prices.
-   Creates sequences (`create_sequences` function with `seq_length=10`).
-   Defines and compiles a `Sequential` LSTM model (similar to `Lstm.py` but with `seq_length` as input shape).
-   Trains the model and makes predictions.
-   Inverse transforms predictions and plots results for visualization.

**Access in Streamlit:**
Select "Wise LSTM" from the sidebar dropdown in the main application.

**Example Code Snippet (Streamlit Integration):**
```python
def display_page():
    st.title('Stock Prediction Using LSTM')
    stock_symbol = st.text_input('Enter Stock Ticker', 'AAPL')
    start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))
    # ... (data fetching, preprocessing, model building, training, prediction, plotting)
```

## 2. Convolutional Neural Network (CNN) Model

CNNs are typically used for image processing but can also be adapted for time-series data by treating sequences as 1D arrays, extracting local patterns.

### 2.1 CNN Integration in Streamlit (`pages/wise_cnn.py`)

The `pages/wise_cnn.py` file implements a CNN model specifically for the WiseTrader Streamlit application. This model aims to classify stock price movement (up/down) rather than direct price prediction.

**`display_page()` function overview:**
-   Takes `stock_ticker` as user input.
-   Fetches 1 year of hourly stock data (`yf.download`).
-   Preprocesses 'Open', 'High', 'Low', 'Close', 'Volume' data.
-   Creates sequences with `sequence_length=10`.
-   Generates a binary target: 1 if `closing_prices > previous_closing_price`, else 0.
-   Builds a `Sequential` Keras model with `Conv1D`, `MaxPooling1D`, `Flatten`, and `Dense` layers, outputting a sigmoid activation for binary classification.
-   Compiles with `binary_crossentropy` loss and `accuracy` metric.
-   Trains the model and displays performance metrics like accuracy and plots loss/accuracy history.

**Access in Streamlit:**
Select "Wise CNN" from the sidebar dropdown in the main application.

**Example Code Snippet (Model Definition):**
```python
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5), padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

