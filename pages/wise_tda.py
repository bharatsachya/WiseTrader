import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import streamlit as st

# Attempt to import ripser and persim. If they fail, use our native sublevel set filtration homology.
TDA_MODE = "native"
try:
    from ripser import ripser
    from persim import PersistenceEntropy
    TDA_MODE = "ripser"
except ImportError:
    TDA_MODE = "native"


def compute_0d_persistence_native(x):
    """
    Computes 0-dimensional persistence diagrams for a 1D time series
    using sublevel set filtration and a Union-Find (Disjoint Set) structure.
    """
    n = len(x)
    indices = np.argsort(x)
    parent = list(range(n))
    born = np.copy(x)
    active = np.zeros(n, dtype=bool)
    pairs = []

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    for idx in indices:
        active[idx] = True
        neighbors = []
        if idx > 0 and active[idx - 1]:
            neighbors.append(find(idx - 1))
        if idx < n - 1 and active[idx + 1]:
            neighbors.append(find(idx + 1))

        if not neighbors:
            # Component born at idx
            pass
        elif len(neighbors) == 1:
            # Merge current point into the active neighbor component
            parent[idx] = neighbors[0]
        else:
            # Merge two active components. Elder rule: the younger (higher birth) component dies.
            n1, n2 = neighbors[0], neighbors[1]
            if n1 != n2:
                if born[n1] > born[n2]:
                    parent[n1] = n2
                    pairs.append((born[n1], x[idx]))
                else:
                    parent[n2] = n1
                    pairs.append((born[n2], x[idx]))
                parent[idx] = find(n1)
    
    # Return pairs (birth, death)
    return pairs


def calculate_entropy_native(pairs):
    """
    Calculates persistence entropy from the persistence diagram.
    """
    if not pairs:
        return 0.0
    lifetimes = [death - birth for birth, death in pairs if death > birth]
    if not lifetimes or sum(lifetimes) == 0:
        return 0.0
    
    total = sum(lifetimes)
    probs = [l / total for l in lifetimes]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy


def extract_tda_features(window):
    """
    Extracts persistence entropy features from a price window.
    """
    if TDA_MODE == "ripser":
        try:
            window_2d = window.reshape(-1, 1)
            diagrams = ripser(window_2d, maxdim=1)["dgms"]
            pe = PersistenceEntropy()
            entropy_0 = pe.fit_transform([diagrams[0]])[0]
            entropy_1 = pe.fit_transform([diagrams[1]])[0] if len(diagrams) > 1 else 0.0
            return [entropy_0, entropy_1]
        except Exception:
            # Fallback to native if ripser fails at runtime
            pairs = compute_0d_persistence_native(window)
            entropy = calculate_entropy_native(pairs)
            return [entropy, 0.0]
    else:
        # Native sublevel set filtration Homology
        pairs = compute_0d_persistence_native(window)
        entropy = calculate_entropy_native(pairs)
        # We can also compute persistence entropy on the inverted series (suplevel set filtration)
        pairs_inv = compute_0d_persistence_native(-window)
        entropy_inv = calculate_entropy_native(pairs_inv)
        return [entropy, entropy_inv]


def display_page():
    st.title('🔮 Topological Data Analysis (TDA)')
    st.markdown("""
    This module applies **Topological Data Analysis (TDA)** to identify structures in financial time-series.
    It constructs rolling time-delay windows, performs **Persistent Homology** (Sublevel and Suplevel Set Filtrations), 
    extracts **Persistence Entropy** as features, and feeds them into a **Support Vector Machine (SVM)** to forecast price direction.
    """)

    st.markdown("---")

    # Mode Indicator
    if TDA_MODE == "ripser":
        st.success("🔬 TDA Mode: Active (Using Ripser & Persim library)")
    else:
        st.info("⚡ TDA Mode: Active (Using native Python Sublevel Set DSU Filtration solver)")

    # Configurations
    st.subheader("⚙️ Model configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input('Stock Ticker Symbol', value='AAPL', key='tda_ticker').upper().strip()
        start_date = st.date_input('Start Date', pd.to_datetime('2019-01-01'), key='tda_start')
    with col2:
        end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'), key='tda_end')
        window_size = st.slider('Sliding Window Size (Days)', min_value=15, max_value=60, value=30, step=5, key='tda_window')
    with col3:
        svm_kernel = st.selectbox('SVM Kernel Function', ['rbf', 'linear', 'poly'], index=0, key='tda_svm')
        test_split = st.slider('Testing Split Size (%)', min_value=10, max_value=30, value=20, step=5, key='tda_split') / 100.0

    st.markdown("---")

    # Load data
    with st.spinner("Downloading close prices..."):
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("Failed to load stock data. Check configurations.")
        return

    # Check for multi-index and flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    prices = df["Close"].values.flatten()
    
    if len(prices) <= window_size + 20:
        st.warning("Insufficient data points. Adjust dates or reduce window size.")
        return

    # Construct sliding window persistence features
    X = []
    y = []
    
    with st.spinner("Performing Persistent Homology filtration on windows..."):
        # For efficiency, we scan
        progress_bar = st.progress(0)
        n_windows = len(prices) - window_size - 1
        
        for i in range(n_windows):
            window = prices[i:i+window_size]
            feat = extract_tda_features(window)
            X.append(feat)
            
            # Predict if tomorrow's price goes up compared to the last close of the window
            future_return = prices[i + window_size] - prices[i + window_size - 1]
            y.append(1 if future_return > 0 else 0)
            
            # Update progress occasionally
            if i % max(1, n_windows // 10) == 0:
                progress_bar.progress((i + 1) / n_windows)
        progress_bar.progress(1.0)

    X = np.array(X)
    y = np.array(y)

    # Train / Test split
    split = int((1 - test_split) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit SVM Model
    with st.spinner("Training Support Vector Machine..."):
        model = SVC(kernel=svm_kernel, probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # Predict next day
    latest_window = prices[-window_size:]
    latest_feat = np.array(extract_tda_features(latest_window)).reshape(1, -1)
    latest_feat_scaled = scaler.transform(latest_feat)
    prob_up = float(model.predict_proba(latest_feat_scaled)[0][1])

    direction = "UP 📈" if prob_up > 0.5 else "DOWN 📉"
    confidence = prob_up if prob_up > 0.5 else (1 - prob_up)

    st.subheader("📊 Homology-based SVM Classifier Results")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        card_color = "#00FFCC" if prob_up > 0.5 else "#FF3366"
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px; border-left: 5px solid {card_color};">
            <div class="metric-label">TDA Predictor Next Day Movement</div>
            <div class="metric-value" style="color: {card_color};">{direction}</div>
            <div class="metric-delta" style="color: #A0AEC0;">Up Probability: {prob_up*100:.2f}% | Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Testing Metrics</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #A0AEC0;">Accuracy: {acc*100:.2f}%</div>
            <div class="metric-delta" style="color: #718096;">Kernel: {svm_kernel} | Features: {X.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        # Plot persistence entropy features over time
        test_dates = df.index[window_size + split:-1]
        fig_feat = go.Figure()
        fig_feat.add_trace(go.Scatter(x=test_dates, y=X_test[:, 0], name='0D Homology Entropy', line=dict(color='#00FFCC', width=1.5)))
        fig_feat.add_trace(go.Scatter(x=test_dates, y=X_test[:, 1], name='Dual Homology Entropy', line=dict(color='#0077FF', width=1.5)))
        fig_vol = np.std(X_test, axis=0)
        fig_feat.update_layout(
            title='Persistence Entropy Features (Test Period)',
            xaxis_title='Date',
            yaxis_title='Entropy Value',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_feat, use_container_width=True)

if __name__ == "__main__":
    display_page()
