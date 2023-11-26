import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
import streamlit as st


def display_page():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    page_name = "wise_cnn"  # Change this based on the current page name
    stock_ticker_key = f'{page_name}_stock_ticker'
    stock_ticker = st.text_input('Enter Stock Ticker', value='NV20.NS', key=stock_ticker_key)

    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    sequence_length = 10
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])

    closing_prices = data[sequence_length:, 3]
    y = np.where(closing_prices > data[:-sequence_length, 3], 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64)

    _, accuracy = model.evaluate(X_test, y_test)
    st.write(f"Accuracy: {accuracy:.4f}")
    
    st.pyplot(plt.figure(figsize=(10, 6)))
    plt.plot(y_test, label='Actual Test Prices', color='green')
    plt.plot(model.predict(X_test), label='Predicted Test Prices', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Price Movement')
    plt.title('Intraday Stock Prediction - Test vs. Predicted')
    plt.legend()
    
    st.pyplot()
    

