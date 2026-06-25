import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import streamlit as st

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    # Annualized return
    portfolio_return = np.sum(mean_returns * weights) * 252
    # Annualized volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    # Sharpe ratio (assuming risk-free rate of 2%)
    sharpe_ratio = (portfolio_return - 0.02) / (portfolio_volatility + 1e-9)
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix):
    return -calculate_portfolio_metrics(weights, mean_returns, cov_matrix)[2]

def optimize_portfolio(mean_returns, cov_matrix, num_assets):
    # Initial guess: equal distribution
    init_weights = np.ones(num_assets) / num_assets
    # Bounds: weights between 0 and 1 (long-only)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    # Constraints: sum of weights equals 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    result = minimize(
        negative_sharpe,
        init_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def display_page():
    st.title('⚙️ Portfolio Allocation & Strategy Optimizer')
    st.markdown("""
    This optimization module implements **Modern Portfolio Theory (Markowitz Mean-Variance Optimization)** using SciPy's numerical solvers.
    It takes a custom list of tickers, computes their covariance and mean historical returns, and utilizes the SLSQP optimizer to solve for weights that maximize the portfolio **Sharpe Ratio** (risk-adjusted returns).
    """)

    st.markdown("---")

    # Parameters configuration
    st.subheader("⚙️ Optimizer Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        default_tickers = "AAPL, MSFT, GOOG, AMZN"
        tickers_input = st.text_input("Asset Ticker List (comma separated)", value=default_tickers, key='opt_tickers')
        lookback = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=2, key='opt_lookback')
    with col2:
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=8.0, value=2.0, step=0.5, key='opt_rf') / 100.0
    with col3:
        sizing_method = st.selectbox("Position Sizing Policy", ["Maximum Sharpe Ratio", "Equal Risk Contribution", "Equal Weights"], index=0, key='opt_policy')

    st.markdown("---")

    # Clean ticker string
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("Please provide at least 2 valid ticker symbols for portfolio optimization.")
        return

    # Load data
    with st.spinner("Downloading historical price data..."):
        try:
            # Download close prices
            data = yf.download(tickers, period=lookback)['Close']
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return

    if data.empty:
        st.error("Failed to load data for the specified tickers. Please verify the symbols.")
        return

    # Handle multi-index formats or single index checks
    if isinstance(data, pd.Series):
        st.error("Only one asset data downloaded. Ensure you provide multiple ticker symbols.")
        return

    data = data.dropna()
    if len(data) < 30:
        st.warning("Insufficient data overlap. Clean individual stock records or reduce lookback.")
        return

    # Daily Log Returns
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    num_assets = len(tickers)

    # Perform Optimization
    with st.spinner("Solving optimization bounds..."):
        if sizing_method == "Maximum Sharpe Ratio":
            opt_weights = optimize_portfolio(mean_returns, cov_matrix, num_assets)
        elif sizing_method == "Equal Risk Contribution":
            # Equal risk contribution approximation for simplicity
            # (risk parity weights are proportional to inverse volatility)
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1.0 / vols
            opt_weights = inv_vols / np.sum(inv_vols)
        else:
            opt_weights = np.ones(num_assets) / num_assets

    # Compute metrics
    p_return, p_vol, p_sharpe = calculate_portfolio_metrics(opt_weights, mean_returns, cov_matrix)

    st.subheader("📊 Optimal Asset Allocation Results")

    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Optimized Annual Return</div>
            <div class="metric-value">{p_return*100:.2f}%</div>
            <div class="metric-delta delta-up">Risk-Free Rate: {risk_free_rate*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Optimized Annual Risk</div>
            <div class="metric-value" style="color: #FF3366;">{p_vol*100:.2f}%</div>
            <div class="metric-delta" style="color: #718096;">Portfolio Sharpe Ratio: {p_sharpe:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        # Plot weights pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=tickers, 
            values=opt_weights,
            hole=.3,
            marker=dict(colors=['#00FFCC', '#0077FF', '#FF3366', '#FFCC00', '#8A2BE2'])
        )])
        fig_pie.update_layout(
            title=f'Optimal Portfolio Weight Distribution ({sizing_method})',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Dynamic Position Sizer Simulation (Confidence Thresholding from drop.py)
    st.markdown("---")
    st.subheader("🛡️ Dynamic Position Sizing Simulator")
    st.markdown("""
    Based on the confidence of prediction signals, we can dynamically adjust size to filter noise.
    Adjust the threshold below to simulate position sizes (capital: $10,000).
    """)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        threshold = st.slider("Signal Decision Threshold", min_value=0.0, max_value=0.01, value=0.002, step=0.0005, format="%.4f")
    with col_s2:
        multiplier = st.slider("Confidence Size Multiplier", min_value=1.0, max_value=20.0, value=10.0, step=1.0)

    # Simulate hypothetical strategy on the first ticker
    sim_ticker = tickers[0]
    sim_returns = returns.iloc[:, 0].values
    
    # Generate mock model prediction (using returns with slight lag/noise as proxy)
    np.random.seed(42)
    predictions = sim_returns + np.random.normal(0, 0.005, len(sim_returns))

    signals = []
    capital = 10000.0
    capital_curve = [capital]
    
    for pred, actual in zip(predictions, sim_returns):
        confidence = abs(pred)
        # Position scaling: capital * min(confidence * multiplier, 1.0)
        pos_scale = min(confidence * multiplier, 1.0)
        
        if pred > threshold:
            # Long
            capital = capital * (1 + pos_scale * actual)
            signals.append(1)
        elif pred < -threshold:
            # Short
            capital = capital * (1 - pos_scale * actual)
            signals.append(-1)
        else:
            # Out of market
            capital = capital
            signals.append(0)
        capital_curve.append(capital)

    # Plot simulated capital curve
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=returns.index, y=capital_curve[:-1], name='Dynamic Sizer Equity Curve', line=dict(color='#00FFCC', width=2)))
    # Buy and hold base capital curve
    bh_curve = 10000.0 * np.exp(np.cumsum(sim_returns))
    fig_sim.add_trace(go.Scatter(x=returns.index, y=bh_curve, name=f'Buy & Hold ({sim_ticker})', line=dict(color='rgba(255,255,255,0.3)', width=1.5)))
    
    fig_sim.update_layout(
        title=f'Dynamic Confidence Sizer Performance Simulator (Asset: {sim_ticker})',
        xaxis_title='Date',
        yaxis_title='Account Value ($)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_sim, use_container_width=True)

if __name__ == "__main__":
    display_page()
