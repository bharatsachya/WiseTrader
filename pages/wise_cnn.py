import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def display_page():
    st.title('🧠 Stock Price Direction Prediction Using CNN')
    st.markdown("""
    This module uses a **1D Convolutional Neural Network (CNN)** to perform binary classification. 
    It predicts whether the closing price of the next period will be higher (1) or lower/equal (0) compared to the current period, based on a sequence of historical features (Open, High, Low, Close, Volume).
    """)

    st.markdown("---")

    # Hyperparameter Sidebar settings
    st.subheader("⚙️ Model Configuration")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        stock_ticker = st.text_input('Stock Ticker Symbol', value='NV20.NS', key='cnn_ticker').upper().strip()
        lookback_period = st.selectbox('Lookback Data Period', ['3mo', '6mo', '1y', '2y'], index=2, key='cnn_lookback')
    with h_col2:
        seq_length = st.slider('Sequence Length (Timesteps)', min_value=5, max_value=30, value=10, step=1, key='cnn_seq')
        epochs = st.slider('Training Epochs', min_value=5, max_value=50, value=15, step=5, key='cnn_epochs')
    with h_col3:
        batch_size = st.select_slider('Batch Size', options=[16, 32, 64, 128], value=64, key='cnn_batch')
        test_size = st.slider('Test Split Size (%)', min_value=10, max_value=40, value=20, step=5, key='cnn_split') / 100.0

    st.markdown("---")

    # Fetch stock data
    st.write(f"Fetching data for **{stock_ticker}** (Period: {lookback_period})...")
    # Hourly data is optimal for CNN sequence length
    df_raw = yf.download(stock_ticker, period=lookback_period, interval="1h")

    if df_raw.empty:
        st.error("Failed to fetch stock data. Please check the ticker symbol or connection.")
        return

    # Check for multi-index and flatten if needed
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [col[0] for col in df_raw.columns]

    df = df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    if len(df) <= seq_length + 10:
        st.warning("Insufficient data. Try a longer lookback period or check the stock symbol.")
        return

    # Preprocess data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X = np.array([scaled_data[i:i+seq_length] for i in range(len(scaled_data) - seq_length)])
    # Closing prices of the next timestep after each sequence
    closing_prices = scaled_data[seq_length:, 3]
    # Predict if closing price goes up compared to the last close of the sequence
    last_sequence_closes = scaled_data[:-seq_length, 3]
    y = np.where(closing_prices > last_sequence_closes, 1, 0)
    
    # Adjust shapes
    X = X[:len(y)]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    st.info(f"Data ready. Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

    # Build the CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 5), padding='same'),
        MaxPooling1D(pool_size=2 if seq_length >= 4 else 1),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    with st.spinner("Training 1D CNN model..."):
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=0, 
            validation_split=0.2
        )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    st.subheader("📊 Model Performance & Training Logs")
    
    # Plot accuracy and loss with Plotly
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Accuracy', line=dict(color='#00FFCC')))
    fig_hist.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', line=dict(color='#0077FF')))
    fig_hist.update_layout(
        title='Model Accuracy over Epochs',
        xaxis_title='Epochs',
        yaxis_title='Accuracy',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Predict
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 10px;">
            <div class="metric-label">Out-of-Sample Accuracy</div>
            <div class="metric-value">{acc*100:.2f}%</div>
            <div class="metric-delta delta-up">Loss: {loss:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Confusion Matrix as a nice text block
        st.markdown("### Confusion Matrix")
        st.text(f"True Down: {cm[0][0]} | False Up: {cm[0][1]}\nFalse Down: {cm[1][0]} | True Up: {cm[1][1]}")
        
    with col2:
        # Plot predictions vs actual as a timeline
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=y_test[-80:], name='Actual Direction (Up=1/Down=0)', mode='markers+lines', line=dict(color='#A0AEC0', width=1)))
        fig_pred.add_trace(go.Scatter(y=y_pred[-80:], name='Predicted Direction', mode='markers', marker=dict(color='#00FFCC', size=8)))
        fig_pred.update_layout(
            title='Actual vs Predicted Class (Last 80 timesteps)',
            xaxis_title='Sample Index',
            yaxis_title='Class (0 or 1)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickvals=[0, 1])
        )
        st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    display_page()
