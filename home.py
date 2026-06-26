import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import pages
from pages import wise_cnn
from pages import wise_lstm
from pages import wise_rnn
from pages import wise_wavelet
from pages import wise_rams
from pages import wise_ukf
from pages import wise_tda
from pages import wise_backtest
from pages import trading_optimizer

# Set page config
st.set_page_config(
    layout="wide",
    page_icon="📈",
    page_title="WiseTrader | Advanced Neural Network Trading Suite",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00FFCC 0%, #0077FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    
    .sub-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 400;
    }
    
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 255, 204, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #718096;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 2px;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .delta-up {
        color: #00FFCC;
    }
    
    .delta-down {
        color: #FF3366;
    }
    
    .sidebar-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00FFCC;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown('<div class="sidebar-header">🚀 WiseTrader Terminal</div>', unsafe_allow_html=True)
pages = {
    "📊 Market Dashboard": "dashboard",
    "🧠 Wise CNN Predictor": "cnn",
    "🧠 Wise LSTM Predictor": "lstm",
    "🧠 Wise RNN & GRU Predictor": "rnn",
    "⚡ Wavelet CNN-BiLSTM-Attention": "wavelet",
    "🤖 Regime-Aware MoE (RAMS)": "rams",
    "🌀 Unscented Kalman Filter": "ukf",
    "🔮 Topological Data Analysis (TDA)": "tda",
    "📈 HMM & LSTM Backtester": "backtest",
    "⚙️ Trading Strategy Optimizer": "optimizer"
}
selected_page_label = st.sidebar.selectbox("Navigate Module", list(pages.keys()))
selected_page = pages[selected_page_label]

st.sidebar.markdown("---")
# Contact
with st.sidebar.expander("📬 Contact / Project Details"):
    st.write("**GitHub Repo:**", "[bharatsachya/WiseTrader](https://github.com/bharatsachya/WiseTrader)")
    st.write("**Frameworks:** Streamlit, PyTorch, Keras, TensorFlow, Scikit-Learn, SciPy, Plotly")

# --- MODULE EXECUTION ---
if selected_page == "dashboard":
    st.markdown('<div class="main-title">WISE TRADER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced Quantitative Trading & Neural Network Terminal</div>', unsafe_allow_html=True)
    
    # Selection inputs
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input('Stock Ticker Symbol', value='AAPL', key='dashboard_ticker').upper().strip()
    with col2:
        period_options = ['3mo', '6mo', '1y', '2y', '5y']
        selected_period = st.selectbox('Select Lookback Period', period_options, index=2)
        
    # Download data
    with st.spinner(f"Retrieving market data for {ticker}..."):
        df = yf.download(ticker, period=selected_period)
        
    if df.empty:
        st.error(f"❌ Could not retrieve market data for symbol '{ticker}'. Please verify the symbol.")
    else:
        # Check standard columns are 1D arrays
        df_display = df.copy()
        if isinstance(df_display.columns, pd.MultiIndex):
            # Flatten multi-index columns if present
            df_display.columns = [col[0] for col in df_display.columns]
            
        close_prices = df_display['Close']
        open_prices = df_display['Open']
        high_prices = df_display['High']
        low_prices = df_display['Low']
        volumes = df_display['Volume']
        
        last_close = float(close_prices.iloc[-1])
        prev_close = float(close_prices.iloc[-2])
        price_change = last_close - prev_close
        pct_change = (price_change / prev_close) * 100
        
        last_open = float(open_prices.iloc[-1])
        last_high = float(high_prices.iloc[-1])
        last_low = float(low_prices.iloc[-1])
        last_volume = int(volumes.iloc[-1])
        
        # Calculate summary metrics cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        # Metric 1: Last Close
        delta_class = "delta-up" if price_change >= 0 else "delta-down"
        delta_sign = "+" if price_change >= 0 else ""
        m_col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Close ({ticker})</div>
            <div class="metric-value">${last_close:.2f}</div>
            <div class="metric-delta {delta_class}">{delta_sign}{price_change:.2f} ({delta_sign}{pct_change:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric 2: Open
        m_col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Today's Open</div>
            <div class="metric-value">${last_open:.2f}</div>
            <div class="metric-delta" style="color: #718096;">Prev. Close: ${prev_close:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric 3: Range
        m_col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day's Range</div>
            <div class="metric-value">${last_low:.2f} - ${last_high:.2f}</div>
            <div class="metric-delta" style="color: #718096;">Spread: ${(last_high - last_low):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric 4: Volume
        m_col4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{last_volume:,}</div>
            <div class="metric-delta" style="color: #718096;">10D Avg: {int(volumes.tail(10).mean()):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plot Candlestick chart
        st.subheader("📊 Interactive Candlestick Chart & Moving Averages")
        
        # Precompute moving averages
        df_display['EMA20'] = close_prices.ewm(span=20, adjust=False).mean()
        df_display['EMA50'] = close_prices.ewm(span=50, adjust=False).mean()
        
        # Bollinger Bands
        sma20 = close_prices.rolling(window=20).mean()
        std20 = close_prices.rolling(window=20).std()
        df_display['BB_Upper'] = sma20 + (2 * std20)
        df_display['BB_Lower'] = sma20 - (2 * std20)
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_display.index,
            open=df_display['Open'],
            high=df_display['High'],
            low=df_display['Low'],
            close=df_display['Close'],
            name='Candlestick'
        ))
        
        # EMA 20
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['EMA20'],
            mode='lines',
            line=dict(color='#00FFCC', width=1.5),
            name='EMA 20'
        ))
        
        # EMA 50
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['EMA50'],
            mode='lines',
            line=dict(color='#0077FF', width=1.5),
            name='EMA 50'
        ))
        
        # Bollinger bands Upper
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['BB_Upper'],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dash'),
            name='Bollinger Upper'
        ))
        
        # Bollinger bands Lower
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['BB_Lower'],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.02)',
            name='Bollinger Lower'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=30, r=30, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(10, 15, 30, 0.5)',
            paper_bgcolor='rgba(10, 15, 30, 0.5)',
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', title='Price ($)'),
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', title='Date')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data view
        with st.expander("📝 View Raw Historical Data Table"):
            st.dataframe(df_display.tail(100), use_container_width=True)

elif selected_page == "cnn":
    wise_cnn.display_page()

elif selected_page == "lstm":
    wise_lstm.display_page()

elif selected_page == "rnn":
    wise_rnn.display_page()

elif selected_page == "wavelet":
    wise_wavelet.display_page()

elif selected_page == "rams":
    wise_rams.display_page()

elif selected_page == "ukf":
    wise_ukf.display_page()

elif selected_page == "tda":
    wise_tda.display_page()

elif selected_page == "backtest":
    wise_backtest.display_page()

elif selected_page == "optimizer":
    trading_optimizer.display_page()
