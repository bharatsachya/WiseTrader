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

    # Fetch stock data
    st.write(f"Fetching data for {stock_ticker}...")
    stock_data = yf.download(stock_ticker, period="1y", interval="1h")  # Modify the interval if needed

    if stock_data.empty:
        st.error("Failed to fetch stock data. Please check the ticker symbol.")
        return

    # Preprocess data
    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    sequence_length = 10
    X = np.array([scaled_data[i:i+sequence_length] for i in range(len(scaled_data) - sequence_length)])
    closing_prices = scaled_data[sequence_length:, 3]
    y = np.where(closing_prices > scaled_data[:-sequence_length, 3], 1, 0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 5), padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    st.write("Training the model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Accuracy: {accuracy:.4f}")

    # Plot accuracy and loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    st.pyplot()

    # Predict and plot
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Test Classes', color='green', alpha=0.6)
    plt.plot(y_pred, label='Predicted Classes', color='blue', alpha=0.6)
    plt.xlabel('Time Steps')
    plt.ylabel('Class (0/1)')
    plt.title('Actual vs Predicted Classes')
    plt.legend()
    st.pyplot()


if __name__ == "__main__":
    display_page()
