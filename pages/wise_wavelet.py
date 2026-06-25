import numpy as np
import pandas as pd
import pywt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dense, Dropout, Attention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam

def wavelet_denoise(series, wavelet="db4", level=2):
    # PyWavelets expects float64 array
    series_arr = np.squeeze(np.asarray(series, dtype=np.float64))
    coeffs = pywt.wavedec(series_arr, wavelet, mode="per")
    # Soft thresholding
    coeffs[1:] = [pywt.threshold(c, value=np.std(c)/2, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")

def display_page():
    st.title('⚡ Hybrid Wavelet-CNN-BiLSTM-Attention Model')
    st.markdown("""
    This advanced hybrid pipeline applies **Discrete Wavelet Transform (DWT)** to denoise the stock price series, 
    extracts local temporal patterns using a **1D CNN**, models global bidirectional dependencies using a **BiLSTM**, 
    and weights key timesteps using an **Attention Mechanism**.
    """)

    st.markdown("---")

    # Configuration panel
    st.subheader("⚙️ Model Configuration")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        ticker = st.text_input('Stock Ticker Symbol', value='AAPL', key='wavelet_ticker').upper().strip()
        start_date = st.date_input('Start Date', pd.to_datetime('2018-01-01'), key='wavelet_start')
        end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'), key='wavelet_end')
    with h_col2:
        wavelet_family = st.selectbox('Wavelet Family', ['db4', 'haar', 'sym5', 'coif2'], index=0, key='wavelet_family')
        decomp_level = st.slider('Decomposition Level', min_value=1, max_value=4, value=2, step=1, key='wavelet_level')
    with h_col3:
        seq_length = st.slider('Sequence Window Length', min_value=15, max_value=50, value=30, step=5, key='wavelet_seq')
        epochs = st.slider('Training Epochs', min_value=5, max_value=30, value=15, step=5, key='wavelet_epochs')

    st.markdown("---")

    # Fetch data
    with st.spinner("Fetching market data..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        
    if df.empty:
        st.error("Failed to fetch stock data. Please check ticker symbol or date parameters.")
        return

    # Check for multi-index and flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.dropna(inplace=True)
    
    if len(df) <= seq_length + 20:
        st.warning("Insufficient data points for training. Please extend the date range.")
        return

    # 1. Wavelet Denoising
    with st.spinner("Applying Wavelet Denoising..."):
        close_prices = df['Close'].values.flatten()
        denoised_close = wavelet_denoise(close_prices, wavelet=wavelet_family, level=decomp_level)
        # Handle slice length mismatch if any
        if len(denoised_close) > len(df):
            denoised_close = denoised_close[:len(df)]
        elif len(denoised_close) < len(df):
            # Pad with last item
            denoised_close = np.pad(denoised_close, (0, len(df) - len(denoised_close)), 'edge')
            
        df['Denoised_Close'] = denoised_close

    # 2. Features calculation
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    # Display Denoising comparison chart
    st.subheader("🌀 Wavelet Signal Denoising Analysis")
    fig_denoise = go.Figure()
    fig_denoise.add_trace(go.Scatter(x=df.index, y=df['Close'].values.flatten(), name='Original Close Price', line=dict(color='rgba(255,255,255,0.25)', width=1.5)))
    fig_denoise.add_trace(go.Scatter(x=df.index, y=df['Denoised_Close'].values.flatten(), name=f'Denoised Close ({wavelet_family})', line=dict(color='#00FFCC', width=1.5)))
    fig_denoise.update_layout(
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_denoise, use_container_width=True)

    # 3. Sequence Building
    features = ["Denoised_Close", "Volatility", "Momentum"]
    X_vals = df[features].values
    y_vals = df["Target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals)

    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X_scaled, y_vals, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    st.info(f"Data scaled and segmented. Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # 4. Neural Network Architecture
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

    # CNN Block
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(input_layer)
    x = MaxPooling1D(pool_size=2 if seq_length >= 4 else 1)(x)
    x = LayerNormalization()(x)

    # BiLSTM Block
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Attention Layer
    attention = Attention()([x, x])
    x = Concatenate()([x, attention])

    # Global Pooling
    x = GlobalAveragePooling1D()(x)

    # Dense Head
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    with st.spinner("Training Hybrid Neural Network..."):
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Predict next day direction
    latest_sequence = X_seq[-1].reshape(1, X_seq.shape[1], X_seq.shape[2])
    prediction_prob = float(model.predict(latest_sequence, verbose=0)[0][0])
    
    direction = "UP 📈" if prediction_prob > 0.5 else "DOWN 📉"
    confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)

    st.subheader("📊 Forecast and Diagnostics")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Prediction Card
        card_color = "#00FFCC" if prediction_prob > 0.5 else "#FF3366"
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px; border-left: 5px solid {card_color};">
            <div class="metric-label">Next Day Direction Forecast</div>
            <div class="metric-value" style="color: {card_color};">{direction}</div>
            <div class="metric-delta" style="color: #A0AEC0;">Probability: {prediction_prob*100:.2f}% | Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Testing Metrics</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #A0AEC0;">Accuracy: {accuracy*100:.2f}%</div>
            <div class="metric-delta" style="color: #718096;">Out-of-sample Cross-Entropy Loss: {loss:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Plot predicted probability on test set
        test_preds = model.predict(X_test, verbose=0).flatten()
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(y=test_preds[-100:], name='Estimated Up-Probability', line=dict(color='#00FFCC', width=1.5)))
        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", annotation_text="Threshold (50%)")
        fig_prob.update_layout(
            title='Next Day Up-Probability Sequence (Last 100 periods)',
            xaxis_title='Index',
            yaxis_title='Probability',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_prob, use_container_width=True)

if __name__ == "__main__":
    display_page()
