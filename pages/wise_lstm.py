import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
def display_page():
    st.title('Stock Prediction Using LSTM')

    stock_symbol = st.text_input('Enter Stock Ticker', 'AAPL')  # You can change the default symbol
    start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

    # Fetch data from Yahoo Finance
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Selecting 'Close' prices for prediction
    df = data[['Close']].copy()

    # Normalize the data
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Split the data into training and testing sets
    train_data = df.iloc[:int(0.8*len(df))]
    test_data = df.iloc[int(0.8*len(df)):]

    # Convert the data to sequences for the RNN model
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)

    seq_length = 10  # Define sequence length
    X_train = create_sequences(train_data.values, seq_length)
    y_train = train_data.iloc[seq_length:].values

    X_test = create_sequences(test_data.values, seq_length)
    y_test = test_data.iloc[seq_length:].values

    # Define and compile the RNN model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Predict on test data
    predicted_values = model.predict(X_test)

    # Inverse transform the predictions
    predicted_values = scaler.inverse_transform(predicted_values)
    y_test = scaler.inverse_transform(y_test)

    # Visualize predictions vs actual values
    st.pyplot(plt.figure(figsize=(12, 6)))
    plt.plot(y_test, label='Actual')
    plt.plot(predicted_values, label='Predicted')
    plt.legend()
    st.pyplot(plt)
   