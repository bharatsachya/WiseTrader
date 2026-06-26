import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import models from root RAMS script
# To prevent import path issues, we can redefine or import the classes.
# Since we are limited to this folder, we can define the classes directly to ensure robust standalone execution.

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # remove future leakage
        out = self.bn(out)
        return self.relu(out)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.tcn1 = TemporalBlock(input_dim, hidden_dim, dilation=1)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # to (batch, features, seq_len)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:, :, -1]  # last timestep
        return self.fc(x)

class RegimeDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_regimes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_regimes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        regime_logits = self.fc(h[-1])
        regime_probs = F.softmax(regime_logits, dim=-1)
        return regime_probs

class GatingNetwork(nn.Module):
    def __init__(self, num_regimes, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(num_regimes, 16)
        self.fc2 = nn.Linear(16, num_experts)

    def forward(self, regime_probs):
        x = F.relu(self.fc1(regime_probs))
        return F.softmax(self.fc2(x), dim=-1)

class RAMSModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        
        self.regime_detector = RegimeDetector(input_dim, hidden_dim, num_experts)
        self.gating = GatingNetwork(num_experts, num_experts)
        
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        regime_probs = self.regime_detector(x)
        expert_weights = self.gating(regime_probs)

        expert_outputs = torch.cat(
            [expert(x) for expert in self.experts],
            dim=1
        )

        weighted_output = torch.sum(
            expert_weights * expert_outputs,
            dim=1,
            keepdim=True
        )

        return weighted_output, expert_outputs, expert_weights

def display_page():
    st.title('🤖 Regime-Aware Mixture of Experts (RAMS)')
    st.markdown("""
    This page hosts the **RAMS (Regime-Aware Mixture of Experts)** model implemented in PyTorch.
    It uses a recurrent **Regime Detector** to classify market states, a **Gating Network** to assign expert weights, and 
    several **Temporal Convolutional Network (TCN) Experts** that are dynamically routed to predict next-day log returns.
    """)

    st.markdown("---")

    # Layout config
    st.subheader("⚙️ Model configuration")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        ticker = st.text_input('Stock Ticker Symbol', value='AAPL', key='rams_ticker').upper().strip()
        start_date = st.date_input('Start Date', pd.to_datetime('2018-01-01'), key='rams_start')
        end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'), key='rams_end')
    with h_col2:
        num_experts = st.selectbox('Number of Experts (Regimes)', [2, 3, 4], index=1, key='rams_experts')
        seq_length = st.slider('Lookback Window (Sequence)', min_value=10, max_value=45, value=20, step=5, key='rams_seq')
    with h_col3:
        epochs = st.slider('Training Epochs', min_value=10, max_value=100, value=40, step=10, key='rams_epochs')
        learning_rate = st.select_slider('Learning Rate', options=[0.01, 0.005, 0.001, 0.0005], value=0.001, key='rams_lr')

    st.markdown("---")

    # Load data
    with st.spinner("Downloading stock data..."):
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("Failed to load stock data. Check configuration.")
        return

    # Check for multi-index and flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.dropna(inplace=True)
    
    if len(df) <= seq_length + 20:
        st.warning("Insufficient historical data points. Expand the date range.")
        return

    # Feature Engineering
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Momentum"] = df["Close"] / df["Close"].rolling(10).mean() - 1
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    features = ["Return", "Volatility", "Momentum", "RSI"]
    X_vals = df[features].values
    # Predict next day return
    y_vals = df["Return"].shift(-1).values
    
    # Drop last row since target is NaN
    X_vals = X_vals[:-1]
    y_vals = y_vals[:-1]
    df_model = df.iloc[:-1].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals)

    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X_scaled, y_vals, seq_length)

    # Train-test split
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Setup PyTorch model
    model = RAMSModel(input_dim=len(features), hidden_dim=32, num_experts=num_experts)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    with st.spinner("Training RAMS Mixture of Experts (PyTorch)..."):
        progress_bar = st.progress(0)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            output, _, _ = model(X_train_t)
            loss = criterion(output.squeeze(), y_train_t)
            loss.backward()
            optimizer.step()
            
            # Update progress
            progress_bar.progress((epoch + 1) / epochs)

    # Testing phase
    model.eval()
    with torch.no_grad():
        test_pred, expert_outputs, expert_weights = model(X_test_t)
        test_loss = criterion(test_pred.squeeze(), y_test_t).item()
        
        # Get active regime (argmax of weights)
        regimes = torch.argmax(expert_weights, dim=1).numpy()
        expert_weights_np = expert_weights.numpy()

    # Re-align dates for the test set
    test_dates = df_model.index[seq_length + split:]
    test_prices = df_model['Close'].iloc[seq_length + split:].values

    st.subheader("🔮 Market Regime Classification & Predictions")

    col_a, colb = st.columns([1, 2])
    with col_a:
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Testing Loss (MSE)</div>
            <div class="metric-value">{test_loss:.6f}</div>
            <div class="metric-delta delta-up">Experts Active: {num_experts}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Details of active regimes
        counts = pd.Series(regimes).value_counts()
        st.write("**Market Regime Sample Counts (Test Set):**")
        for r_id in range(num_experts):
            count = counts.get(r_id, 0)
            pct = (count / len(regimes)) * 100
            st.text(f"Regime/Expert {r_id}: {count} steps ({pct:.1f}%)")

    with colb:
        # Plot price timeline colored by regime
        fig_regime = go.Figure()
        # Add price line
        fig_regime.add_trace(go.Scatter(
            x=test_dates, y=test_prices,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=1.5),
            name='Stock Price'
        ))
        
        # Scatter markers colored by regime
        colors = ['#00FFCC', '#0077FF', '#FF3366', '#FFCC00']
        for r_id in range(num_experts):
            mask = regimes == r_id
            if np.any(mask):
                fig_regime.add_trace(go.Scatter(
                    x=test_dates[mask], y=test_prices[mask],
                    mode='markers',
                    marker=dict(color=colors[r_id % len(colors)], size=5),
                    name=f'Regime {r_id}'
                ))
                
        fig_regime.update_layout(
            title='Price Chart Segmented by Active Market Regime',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=1.02, x=0)
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    # Plot gating weights allocation
    st.subheader("📊 Dynamic Expert Allocation (Gating Weights)")
    
    fig_gate = go.Figure()
    for r_id in range(num_experts):
        fig_gate.add_trace(go.Scatter(
            x=test_dates, 
            y=expert_weights_np[:, r_id],
            name=f'Expert {r_id} Weight',
            stackgroup='one', # Creates stacked area plot
            line=dict(width=0.5, color=colors[r_id % len(colors)])
        ))
        
    fig_gate.update_layout(
        title='Gating Network Weight Allocations (Sum to 100%)',
        xaxis_title='Date',
        yaxis_title='Weight Allocation Probability',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_gate, use_container_width=True)

if __name__ == "__main__":
    display_page()
