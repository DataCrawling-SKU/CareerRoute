import torch
import torch.nn as nn
class BidirectionalLSTMModel(nn.Module):
    """
    Bidirectional LSTM
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
            batch_first=True, # 입력 형태를 (batch, seq, feature)로
            dropout=dropout if num_layers > 1 else 0, # 층이 2개 이상일때만 dropout
            bidirectional=True # 양방향 처리
        )

        self.batch_norm = nn.BatchNorm1d(hidden_size * 2) # 은닉층 2배
        self.dropout = nn.Dropout(dropout)

        # LSTM의 표현을 점진적 압축 -> 학습 안정
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_features) # 마지막 출력은 num_features으로하여 입력과 같은 차원으로(다음 시점 예측)

        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x) # 두번째 반환 값은 무시(마지막 은닉 상태와 셀 상태), 마지막 output의 시점만 필요

        last_out = out[:, -1, :]  # (batch, hidden_size * 2) = 256

        # 정규화 및 드롭아웃
        last_out = self.batch_norm(last_out)
        last_out = self.dropout(last_out)

        # Linear -> ReLU -> Dropout 반복
        x1 = self.relu(self.fc1(last_out))
        x1 = self.dropout(x1)

        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)

        # 마지막은 활성화 함수 x
        y_hat = self.fc3(x2)

        return y_hat


if __name__ == "__main__":
    # 임의의 하이퍼 파라미터
    batch_size = 16
    seq_len = 12
    num_features = 10

    x = torch.randn(batch_size, seq_len, num_features)

    print("Testing LSTM Models")
    print("=" * 60)

    # 테스트
    print("\nBidirectional LSTM Model")
    model2 = BidirectionalLSTMModel(num_features=num_features)
    y2 = model2(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y2.shape}")
    print(f"   Parameters:   {sum(p.numel() for p in model2.parameters()):,}")
