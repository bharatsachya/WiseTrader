import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_and_preprocess(ticker, start_date):
    df = yf.download(ticker, start=start_date)
    if df.empty:
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df.dropna(inplace=True)
    df = df.copy()

    # Log Returns
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # Volatility (rolling 20 periods)
    df['volatility'] = df['returns'].rolling(20).std()
    # Momentum (10 day close ratio)
    df['momentum'] = df['Close'] / df['Close'].shift(10) - 1
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Fast / Slow MA ratios
    df['ma_fast'] = df['Close'].rolling(10).mean()
    df['ma_slow'] = df['Close'].rolling(50).mean()
    df['ma_ratio'] = df['ma_fast'] / df['ma_slow']

    df.dropna(inplace=True)
    return df

def fit_hmm(df, n_states=3):
    features = df[['returns', 'volatility']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train Gaussian HMM
    hmm_model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    hmm_model.fit(features_scaled)

    hidden_states = hmm_model.predict(features_scaled)
    df['regime'] = hidden_states
    
    # Analyze regimes to find the one with lowest volatility (Bull) and highest volatility (Bear)
    regime_means = []
    for r in range(n_states):
        sub = df[df['regime'] == r]
        regime_means.append((r, sub['volatility'].mean(), sub['returns'].mean()))
    
    # Sort by volatility
    regime_means.sort(key=lambda x: x[1])
    
    # Map regimes to: 0 (lowest vol -> bull/calm), n_states-1 (highest vol -> bear/panic)
    mapping = {val[0]: idx for idx, val in enumerate(regime_means)}
    df['regime_sorted'] = df['regime'].map(mapping)

    return df, hmm_model, scaler

def create_sequences(df, feature_cols, target_col='returns', window=20):
    X, y = [], []
    data = df[feature_cols].values
    target = df[target_col].values

    for i in range(window, len(df)):
        X.append(data[i-window:i])
        y.append(target[i])

    return np.array(X), np.array(y)

def build_lstm_model(input_shape, lr=0.001):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='tanh')(x)
    
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def calculate_position(pred, regime, volatility, n_states=3):
    base_signal = np.sign(pred)
    
    # Volatility scaling
    vol_adj = 1.0 / (volatility + 1e-6)
    
    # Regime scaling
    # regime is sorted: 0 is lowest volatility (calm/bullish), n_states-1 is highest volatility (panic/bearish)
    if regime == 0:
        weight = 1.5   # Increase size in low-volatility regimes
    elif regime == n_states - 1:
        weight = 0.5   # Decrease size in high-volatility regimes
    else:
        weight = 1.0   # Neutral size
        
    position = base_signal * weight * vol_adj
    # Normalize position sizes to fit between -2.0 and +2.0 leverage
    return np.clip(position, -2.0, 2.0)

def display_page():
    st.title('📈 Unified HMM & LSTM Backtesting Strategy')
    st.markdown("""
    This backtesting suite combines a **Hidden Markov Model (HMM)** and a **Long Short-Term Memory (LSTM)** neural network.
    The pipeline works as follows:
    1. **HMM** detects the current market regime (Bullish/Bearish/Volatile) based on daily returns and volatility.
    2. **LSTM** uses technical features and HMM states to forecast next-day log returns.
    3. **Dynamic Position Sizing** scales trading leverage based on return predictions and regime volatility.
    """)

    st.markdown("---")

    # Hyperparameter selectors
    st.subheader("⚙️ Strategy Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input('Stock Ticker Symbol', value='AAPL', key='bt_ticker').upper().strip()
        start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'), key='bt_start')
    with col2:
        n_regimes = st.selectbox('HMM Market Regimes', [2, 3, 4], index=1, key='bt_hmm')
        seq_length = st.slider('Sequence Window (LSTM)', min_value=10, max_value=30, value=20, step=5, key='bt_seq')
    with col3:
        epochs = st.slider('Training Epochs', min_value=5, max_value=40, value=15, step=5, key='bt_epochs')
        learning_rate = st.select_slider('Learning Rate', options=[0.01, 0.005, 0.001, 0.0005], value=0.001, key='bt_lr')

    st.markdown("---")

    # Load data
    with st.spinner("Downloading and processing features..."):
        df = load_and_preprocess(ticker, start_date)

    if df.empty:
        st.error("Failed to load stock data. Verify settings.")
        return

    if len(df) <= seq_length + 50:
        st.warning("Insufficient historical data points. Expand the date range.")
        return

    # 1. Fit Hidden Markov Model
    with st.spinner("Fitting Hidden Markov Model for regime classification..."):
        df, hmm_model, hmm_scaler = fit_hmm(df, n_states=n_regimes)

    st.subheader("🌀 HMM Market Regime Clusters")
    
    # Plot returns vs volatility colored by HMM state
    colors = ['#00FFCC', '#0077FF', '#FF3366', '#FFCC00']
    fig_cluster = go.Figure()
    for r_id in range(n_regimes):
        mask = df['regime_sorted'] == r_id
        fig_cluster.add_trace(go.Scatter(
            x=df['volatility'][mask], 
            y=df['returns'][mask],
            mode='markers',
            marker=dict(color=colors[r_id % len(colors)], size=6, opacity=0.7),
            name=f'Regime {r_id} (Mean Vol: {df["volatility"][mask].mean()*100:.2f}%)'
        ))
    fig_cluster.update_layout(
        xaxis_title='Rolling Volatility',
        yaxis_title='Daily Log Return',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.02, x=0)
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # 2. Build and Train LSTM
    feature_cols = ['returns', 'volatility', 'momentum', 'rsi', 'ma_ratio', 'regime_sorted']
    X, y = create_sequences(df, feature_cols, target_col='returns', window=seq_length)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    st.info(f"HMM Complete. Training LSTM on {X_train.shape[0]} sequences...")

    model = build_lstm_model((X.shape[1], X.shape[2]), lr=learning_rate)
    
    with st.spinner("Training return prediction LSTM network..."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    # Predict
    preds = model.predict(X_test, verbose=0).flatten()

    # 3. Backtesting Loop
    test_df = df.iloc[-len(preds):].copy()
    
    positions = []
    for i in range(len(preds)):
        pos = calculate_position(
            preds[i],
            test_df['regime_sorted'].iloc[i],
            test_df['volatility'].iloc[i],
            n_states=n_regimes
        )
        positions.append(pos)

    test_df['position'] = positions
    # Shift position by 1 day because signal at day t takes effect on returns at day t+1
    test_df['strategy_returns'] = test_df['position'].shift(1) * test_df['returns']
    
    # Cumulative returns
    test_df['cum_strategy'] = np.exp(test_df['strategy_returns'].fillna(0.0).cumsum())
    test_df['cum_hold'] = np.exp(test_df['returns'].cumsum())

    # Strategy Metrics
    final_strat = (test_df['cum_strategy'].iloc[-1] - 1) * 100
    final_hold = (test_df['cum_hold'].iloc[-1] - 1) * 100
    outperformance = final_strat - final_hold

    st.subheader("📊 Strategy Backtest Performance vs. Buy-and-Hold")

    col_a, colb = st.columns([1, 2])
    with col_a:
        card_color = "#00FFCC" if outperformance >= 0 else "#FF3366"
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px; border-left: 5px solid {card_color};">
            <div class="metric-label">Strategy Return (Test Period)</div>
            <div class="metric-value" style="color: {card_color};">{final_strat:.2f}%</div>
            <div class="metric-delta">Outperformance: {outperformance:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Buy & Hold Return</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #A0AEC0;">{final_hold:.2f}%</div>
            <div class="metric-delta" style="color: #718096;">Total Test Days: {len(test_df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with colb:
        # Plot cumulative performance
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=test_df.index, y=test_df['cum_hold'], name='Buy & Hold (AAPL)', line=dict(color='rgba(255,255,255,0.3)', width=1.5)))
        fig_perf.add_trace(go.Scatter(x=test_df.index, y=test_df['cum_strategy'], name='HMM+LSTM Strategy', line=dict(color='#00FFCC', width=2)))
        fig_perf.update_layout(
            title='Cumulative Returns Growth Comparison',
            xaxis_title='Date',
            yaxis_title='Investment Growth ($1 -> x)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    # Plot position sizes (leverage) over time
    st.subheader("🛡️ Leverage Allocation Profile")
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        x=test_df.index, 
        y=test_df['position'],
        name='Leverage Position',
        line=dict(color='#0077FF', width=1),
        fill='tozeroy',
        fillcolor='rgba(0, 119, 255, 0.05)'
    ))
    fig_pos.update_layout(
        title='Dynamic Leverage over Time (Bounded -2.0x to +2.0x)',
        xaxis_title='Date',
        yaxis_title='Leverage Scale',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pos, use_container_width=True)

if __name__ == "__main__":
    display_page()
