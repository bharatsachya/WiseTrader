import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, Dropout

def display_page():
    st.title('🧠 Stock Price Forecasting Using RNN & GRU')
    st.markdown("""
    This module uses standard **Recurrent Neural Networks (RNN)** or **Gated Recurrent Units (GRU)** to predict stock prices.
    Comparing these models with the LSTM demonstrates how different recurrent topologies learn temporal dependencies.
    """)

    st.markdown("---")

    # Hyperparameter settings
    st.subheader("⚙️ Model Configuration")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        stock_symbol = st.text_input('Stock Ticker Symbol', value='AAPL', key='rnn_ticker').upper().strip()
        start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'), key='rnn_start')
        end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'), key='rnn_end')
    with h_col2:
        model_type = st.selectbox('Recurrent Layer Type', ['GRU', 'SimpleRNN'], index=0, key='rnn_layer_type')
        lookback = st.slider('Lookback Window (Sequence length)', min_value=10, max_value=100, value=60, step=5, key='rnn_lookback')
    with h_col3:
        epochs = st.slider('Training Epochs', min_value=5, max_value=50, value=15, step=5, key='rnn_epochs')
        batch_size = st.select_slider('Batch Size', options=[16, 32, 64, 128], value=32, key='rnn_batch')
        test_split = st.slider('Testing Split Size (%)', min_value=10, max_value=30, value=20, step=5, key='rnn_split') / 100.0

    st.markdown("---")

    # Fetch data
    st.write(f"Fetching data for **{stock_symbol}** from **{start_date}** to **{end_date}**...")
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("Failed to fetch stock data. Please check ticker symbol or date parameters.")
        return

    # Check for multi-index and flatten if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    df = data[['Close']].copy()
    
    if len(df) <= lookback + 10:
        st.warning("Insufficient data points for training with the selected Lookback Window. Expand dates or reduce lookback.")
        return

    # Normalize closing prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Create dataset sequences
    def create_sequences(dataset, look_back_window):
        X, y = [], []
        for i in range(len(dataset) - look_back_window):
            X.append(dataset[i:i + look_back_window, 0])
            y.append(dataset[i + look_back_window, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split sequentially
    split_index = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    st.info(f"Data ready. Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

    # Build Model
    model = Sequential()
    if model_type == 'GRU':
        model.add(GRU(units=50, return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=False))
    else:
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=50, return_sequences=False))
        
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    with st.spinner(f"Training {model_type} network..."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict on test data
    predicted_scaled = model.predict(X_test, verbose=0)

    # Inverse scaling
    predictions = scaler.inverse_transform(predicted_scaled)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    mae = mean_absolute_error(y_test_unscaled, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))

    # Forecast next day
    latest_sequence = scaled_data[-lookback:]
    latest_sequence = np.reshape(latest_sequence, (1, lookback, 1))
    next_day_scaled = model.predict(latest_sequence, verbose=0)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
    
    last_price = float(df['Close'].iloc[-1])
    est_change = next_day_price - last_price
    est_pct = (est_change / last_price) * 100

    st.subheader(f"📊 {model_type} Forecast Results")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        # Prediction Card
        delta_class = "delta-up" if est_change >= 0 else "delta-down"
        delta_sign = "+" if est_change >= 0 else ""
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Estimated Next Close ({model_type})</div>
            <div class="metric-value">${next_day_price:.2f}</div>
            <div class="metric-delta {delta_class}">{delta_sign}${est_change:.2f} ({delta_sign}{est_pct:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Evaluation Metrics</div>
            <div class="metric-value" style="font-size: 1.4rem; color:#A0AEC0;">RMSE: ${rmse:.2f}</div>
            <div class="metric-delta" style="color: #718096;">Mean Absolute Error (MAE): ${mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Plot predictions vs actual with Plotly
        test_dates = df.index[lookback + split_index:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=y_test_unscaled.flatten(), name='Actual Close Price', line=dict(color='#A0AEC0', width=1.5)))
        fig.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), name=f'{model_type} Forecast', line=dict(color='#0077FF', width=1.5)))
        fig.update_layout(
            title=f'{stock_symbol} Out-of-Sample Forecasting vs Actual',
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    display_page()
