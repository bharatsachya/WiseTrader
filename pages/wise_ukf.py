import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, fx, hx, Q, R):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self.alpha**2 * (dim_x + self.kappa) - dim_x
        self.gamma = np.sqrt(dim_x + self.lambda_)

        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

    def sigma_points(self, x, P):
        # Adding small value for positive-definiteness check
        try:
            U = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Fallback if covariance matrix becomes non-positive-definite
            P_stabilized = P + 1e-8 * np.eye(self.dim_x)
            U = np.linalg.cholesky(P_stabilized)
            
        sigmas = [x]
        for i in range(self.dim_x):
            sigmas.append(x + self.gamma * U[:, i])
            sigmas.append(x - self.gamma * U[:, i])
        return np.array(sigmas)

    def predict(self):
        sigmas = self.sigma_points(self.x, self.P)
        sigmas_f = np.array([self.fx(s) for s in sigmas])

        # Predicted mean
        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        # Predicted covariance
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(len(sigmas_f)):
            diff = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q

        self.sigmas_f = sigmas_f

    def update(self, z):
        sigmas_h = np.array([self.hx(s) for s in self.sigmas_f])
        z_pred = np.sum(self.Wm * sigmas_h)

        # Innovation covariance
        S = 0
        for i in range(len(sigmas_h)):
            diff = sigmas_h[i] - z_pred
            S += self.Wc[i] * diff * diff
        S += self.R

        # Cross covariance
        Pxz = np.zeros((self.dim_x, 1))
        for i in range(len(sigmas_h)):
            Pxz += self.Wc[i] * np.outer(
                self.sigmas_f[i] - self.x,
                sigmas_h[i] - z_pred
            )

        # Kalman Gain
        K = Pxz / (S + 1e-9)

        # State update
        self.x += (K.flatten() * (z - z_pred))

        # Covariance update
        self.P -= K @ K.T * S


# State transition
def state_fx(state):
    """
    State = [return, log_volatility]
    """
    r, log_vol = state
    # Mean reversion coefficient in volatility
    phi = 0.98
    new_r = r  # Prediction step assumes return stays constant
    new_log_vol = phi * log_vol
    return np.array([new_r, new_log_vol])

# Observation model
def observation_hx(state):
    """
    Measurement = observed return
    """
    return state[0]


def display_page():
    st.title('🌀 Unscented Kalman Filter State Estimation')
    st.markdown("""
    This module implements an **Unscented Kalman Filter (UKF)** mapped to a **Stochastic Volatility State-Space Model**. 
    The filter estimates the true log return and tracks log volatility as a latent state from noisy historical prices.
    """)

    st.markdown("---")

    # Interactive configuration
    st.subheader("⚙️ Filter Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input('Stock Ticker Symbol', value='AAPL', key='ukf_ticker').upper().strip()
        start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'), key='ukf_start')
        end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'), key='ukf_end')
    with col2:
        process_noise_r = st.select_slider('Process Noise (Return)', options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key='ukf_q_ret')
        process_noise_v = st.select_slider('Process Noise (Volatility)', options=[1e-4, 1e-3, 1e-2, 1e-1], value=1e-2, key='ukf_q_vol')
    with col3:
        measurement_noise = st.select_slider('Measurement Noise (R)', options=[1e-4, 1e-3, 1e-2, 1e-1], value=1e-3, key='ukf_r')

    st.markdown("---")

    # Load data
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("Failed to load stock data. Check configurations.")
        return

    # Check for multi-index and flatten if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    prices = data["Close"].values.flatten()
    # Log returns
    returns = np.diff(np.log(prices))

    if len(returns) < 30:
        st.warning("Insufficient data points for Kalman filtering. Select a wider range of dates.")
        return

    # Initialize UKF
    Q_matrix = np.diag([process_noise_r, process_noise_v])
    
    ukf = UnscentedKalmanFilter(
        dim_x=2,
        dim_z=1,
        fx=state_fx,
        hx=observation_hx,
        Q=Q_matrix,
        R=measurement_noise
    )

    # Initial state guess
    ukf.x = np.array([returns[0], np.log(np.std(returns) + 1e-6)])
    ukf.P = np.eye(2)

    filtered_returns = []
    estimated_volatilities = []

    # Run filter
    with st.spinner("Running Unscented Kalman Filter recursion..."):
        for r in returns:
            ukf.predict()
            ukf.update(r)
            filtered_returns.append(ukf.x[0])
            estimated_volatilities.append(np.exp(ukf.x[1])) # exp(log_vol)

    # Forecast next day
    ukf.predict()
    next_return = ukf.x[0]
    next_vol = np.exp(ukf.x[1])
    
    last_price = prices[-1]
    next_price = last_price * np.exp(next_return)
    price_change = next_price - last_price
    pct_change = (price_change / last_price) * 100

    # Reconstruct prices
    reconstructed_prices = [prices[0]]
    for r in filtered_returns:
        reconstructed_prices.append(reconstructed_prices[-1] * np.exp(r))
    
    reconstructed_prices = np.array(reconstructed_prices)

    st.subheader("📊 State Estimation & Latent Volatility Charts")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        # Volatility card
        delta_class = "delta-up" if price_change >= 0 else "delta-down"
        delta_sign = "+" if price_change >= 0 else ""
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">UKF Next Day Price Forecast</div>
            <div class="metric-value">${next_price:.2f}</div>
            <div class="metric-delta {delta_class}">{delta_sign}${price_change:.2f} ({delta_sign}{pct_change:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Latent Parameter Estimation</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #A0AEC0;">Est. Volatility: {next_vol*100:.3f}%</div>
            <div class="metric-delta" style="color: #718096;">Filter Return: {next_return:.6f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        # Plot filtered vs actual price
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=prices, name='Actual Close Price', line=dict(color='rgba(255,255,255,0.2)', width=1.5)))
        fig_price.add_trace(go.Scatter(x=data.index, y=reconstructed_prices, name='UKF Filtered Price', line=dict(color='#00FFCC', width=1.5, dash='dash')))
        fig_price.update_layout(
            title=f'UKF Filtered Price Series vs. Actual ({ticker})',
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # Plot latent estimated volatility over time
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data.index[1:], y=estimated_volatilities, name='Latent Volatility', line=dict(color='#0077FF', width=1.5)))
    fig_vol.update_layout(
        title='Latent Daily Return Volatility (Stochastic Volatility State)',
        xaxis_title='Date',
        yaxis_title='Volatility (std dev)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_vol, use_container_width=True)

if __name__ == "__main__":
    display_page()
