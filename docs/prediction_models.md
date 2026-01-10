# Stock Prediction Models in WiseTrader

WiseTrader now incorporates advanced neural network models to predict stock prices. You can access these models through the sidebar navigation on the main page.

## Accessing Prediction Models

On the `Home` page of the WiseTrader application, use the 'Select Page' dropdown in the sidebar to choose between:

*   **Wise CNN**: For Convolutional Neural Network-based predictions.
*   **Wise LSTM**: For Long Short-Term Memory network-based predictions.

## Wise CNN Model

The 'Wise CNN' page leverages a Convolutional Neural Network to perform stock price prediction. 

### Functionality:
1.  **Stock Ticker Input**: Enter the desired stock ticker symbol (e.g., `NV20.NS`).
2.  **Data Fetching**: Fetches 1 year of historical stock data at 1-hour intervals using `yfinance`.
3.  **Data Preprocessing**: Scales the `Open`, `High`, `Low`, `Close`, and `Volume` data using `MinMaxScaler` and creates sequences for the CNN model.
4.  **Model Training**: Trains a `Sequential` CNN model with `Conv1D`, `MaxPooling1D`, `Flatten`, and `Dense` layers using `binary_crossentropy` loss and `adam` optimizer.
5.  **Evaluation**: Displays the model's accuracy after training.

### Usage Example:
1.  Navigate to the 'Wise CNN' page via the sidebar.
2.  Enter a stock ticker in the input field.
3.  The application will fetch data, train the model, and display the accuracy metrics.

**Code Snippet (pages/wise_cnn.py relevant parts):**
```python
st.set_option('deprecation.showPyplotGlobalUse', False)
page_name = "wise_cnn"
stock_ticker_key = f'{page_name}_stock_ticker'
stock_ticker = st.text_input('Enter Stock Ticker', value='NV20.NS', key=stock_ticker_key)

stock_data = yf.download(stock_ticker, period="1y", interval="1h")

df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ... model building and training ...

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5), padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Accuracy: {accuracy:.4f}")
```

## Wise LSTM Model

The 'Wise LSTM' page utilizes a Long Short-Term Memory network for stock price prediction, focusing on closing prices.

### Functionality:
1.  **Stock Ticker Input**: Enter the desired stock ticker symbol (e.g., `AAPL`).
2.  **Date Range Selection**: Select a start and end date for historical data fetching.
3.  **Data Fetching**: Downloads historical data for the specified stock and date range using `yfinance`.
4.  **Data Preprocessing**: Selects 'Close' prices and normalizes them using `MinMaxScaler`.
5.  **Data Splitting**: Divides the data into training (80%) and testing (20%) sets.
6.  **Sequence Conversion**: Prepares the data into sequences suitable for an LSTM model.

### Usage Example:
1.  Navigate to the 'Wise LSTM' page via the sidebar.
2.  Enter a stock ticker symbol.
3.  Select the desired start and end dates.
4.  The application will process the data, which will then be used for LSTM model training and prediction (further implementation details for the LSTM model's training and prediction display would be added as they become available).

**Code Snippet (pages/wise_lstm.py relevant parts):**
```python
st.title('Stock Prediction Using LSTM')

stock_symbol = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

data = yf.download(stock_symbol, start=start_date, end=end_date)

df = data[['Close']].copy()

scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

train_data = df.iloc[:int(0.8*len(df))]
test_data = df.iloc[int(0.8*len(df)):]
```
