import torch
import torch.nn as nn
class BidirectionalLSTMModel(nn.Module):
    """
    Bidirectional LSTM - learns from both past and future context
    Better for pattern recognition in time series
    """

    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Note: bidirectional doubles the output size
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Adjust input size for bidirectional output
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_features)

        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Bidirectional LSTM
        out, _ = self.lstm(x)

        # Last time step (concat of forward and backward)
        last_out = out[:, -1, :]  # (batch, hidden_size * 2)

        # Batch norm and dropout
        last_out = self.batch_norm(last_out)
        last_out = self.dropout(last_out)

        # Dense layers
        x1 = self.relu(self.fc1(last_out))
        x1 = self.dropout(x1)

        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)

        # Output - NO SIGMOID (let model learn the range naturally)
        y_hat = self.fc3(x2)

        return y_hat


if __name__ == "__main__":
    # Test models
    batch_size = 16
    seq_len = 12
    num_features = 10

    # Dummy input
    x = torch.randn(batch_size, seq_len, num_features)

    print("Testing LSTM Models")
    print("=" * 60)

    # Test basic LSTM
    print("\n1. Enhanced LSTM Model")
    model1 = LSTMModel(num_features=num_features)
    y1 = model1(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y1.shape}")
    print(f"   Parameters:   {sum(p.numel() for p in model1.parameters()):,}")

    # Test bidirectional LSTM
    print("\n2. Bidirectional LSTM Model")
    model2 = BidirectionalLSTMModel(num_features=num_features)
    y2 = model2(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y2.shape}")
    print(f"   Parameters:   {sum(p.numel() for p in model2.parameters()):,}")

    # Test attention LSTM
    print("\n3. Attention LSTM Model")
    model3 = AttentionLSTMModel(num_features=num_features)
    y3 = model3(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y3.shape}")
    print(f"   Parameters:   {sum(p.numel() for p in model3.parameters()):,}")

    print("\n" + "=" * 60)
    print("All models working correctly!")
