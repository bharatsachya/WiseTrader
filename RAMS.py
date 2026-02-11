import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------
# 1. Temporal Convolution Block
# ---------------------------
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


# ---------------------------
# 2. Expert Network
# ---------------------------
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.tcn1 = TemporalBlock(input_dim, hidden_dim, dilation=1)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # to (batch, features, seq_len)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:, :, -1]  # last timestep
        return self.fc(x)


# ---------------------------
# 3. Regime Detector
# ---------------------------
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


# ---------------------------
# 4. Gating Network
# ---------------------------
class GatingNetwork(nn.Module):
    def __init__(self, num_regimes, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(num_regimes, 16)
        self.fc2 = nn.Linear(16, num_experts)

    def forward(self, regime_probs):
        x = F.relu(self.fc1(regime_probs))
        return F.softmax(self.fc2(x), dim=-1)


# ---------------------------
# 5. RAMS Model
# ---------------------------
class RAMS(nn.Module):
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
