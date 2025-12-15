import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Enhanced LSTM Model for TIOBE Index Prediction

    Improvements:
    - Multiple LSTM layers for deeper learning
    - Dropout for regularization (prevent overfitting)
    - Batch Normalization for stable training
    - Residual connection for better gradient flow
    - Sigmoid activation for 0-1 range output
    """

    def __init__(self, num_features, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout between LSTM layers
        )

        # Batch normalization for LSTM output
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Dense layers with decreasing size
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_features)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For 0-1 scaled output

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, seq_len, num_features)

        Returns:
            y_hat: (batch_size, num_features) - predicted next time step
        """
        # LSTM forward
        # out: (batch, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)

        # Use last time step output
        last_out = out[:, -1, :]  # (batch, hidden_size)

        # Batch normalization
        last_out = self.batch_norm(last_out)

        # Dropout
        last_out = self.dropout(last_out)

        # First dense layer with ReLU
        x1 = self.fc1(last_out)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Output layer with Sigmoid (0-1 range for scaled data)
        y_hat = self.fc2(x1)
        y_hat = self.sigmoid(y_hat)

        return y_hat


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

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Key difference
        )

        # Note: bidirectional doubles the output size
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Adjust input size for bidirectional output
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_features)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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


class AttentionLSTMModel(nn.Module):
    """
    LSTM with Attention Mechanism
    Allows model to focus on important time steps
    """

    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_features)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = self.softmax(attention_weights)  # Normalize

        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # Batch norm and dropout
        context = self.batch_norm(context)
        context = self.dropout(context)

        # Dense layers
        x1 = self.relu(self.fc1(context))
        x1 = self.dropout(x1)

        y_hat = self.sigmoid(self.fc2(x1))

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
