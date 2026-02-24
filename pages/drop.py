import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# 1. DOWNLOAD DATA
# =========================
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Feature Engineering
df["Return"] = df["Close"].pct_change()
df["Volatility"] = df["Return"].rolling(10).std()
df["Momentum"] = df["Close"] / df["Close"].rolling(10).mean()
df["RSI"] = 100 - (100 / (1 + df["Return"].rolling(14).mean() /
                           df["Return"].rolling(14).std()))

df = df.dropna()

features = ["Return", "Volatility", "Momentum", "RSI"]
target = "Return"

# =========================
# 2. SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
y = df[target].values

sequence_length = 30

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =========================
# 3. HYBRID MODEL
# =========================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        x = self.relu(self.fc1(attn_out))
        x = self.dropout(x)
        return self.fc2(x)

model = HybridModel(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 4. TRAINING
# =========================
epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train).squeeze()
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# =========================
# 5. PREDICTION
# =========================
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze().numpy()

# =========================
# 6. UNIQUE TRADING STRATEGY
# =========================
# Confidence-based dynamic position sizing

signals = []
capital = 10000
position = 0

for pred, actual in zip(predictions, y_test.numpy()):
    confidence = abs(pred)

    if pred > 0.002:  # buy threshold
        position = capital * min(confidence * 10, 1)
        signals.append(1)
    elif pred < -0.002:  # sell threshold
        position = -capital * min(confidence * 10, 1)
        signals.append(-1)
    else:
        signals.append(0)

print("Sample Signals:", signals[:20])
