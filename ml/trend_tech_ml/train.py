import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import BidirectionalLSTMModel


class TiobeDataset(Dataset):
    def __init__(self, X, y):
        # 데이터를 텐서로 변환
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length=12):
    # 시계열 데이터를 모델의 입력 형태에 맞게 변환
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


# 학습데이터 : 0.9 / 검증 데이터 : 0.1 / 테스트 데이터 : 0.1
def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_epoch(model, dataloader, criterion, optimizer, device):
    # 학습코드 (1에폭 마다 수행)
    model.train()
    total_loss = 0
    total_mae = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 순전파
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # 역전파로 가중치 파라미터 조정
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() # 값만 추출하여 메모리 효율 보장
        total_mae += torch.mean(torch.abs(y_pred - y_batch)).item() # 평균 절대 오차

    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)

    return avg_loss, avg_mae

# 검증데이터로 모델 성능 평가
def validate_epoch(model, dataloader, criterion, device):
    model.eval() # 평가 모드
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item() 
            total_mae += torch.mean(torch.abs(y_pred - y_batch)).item()

    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)

    return avg_loss, avg_mae

# 학습 과정 시각화
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss 그래프
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE 그래프
    axes[1].plot(history['train_mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()

# 테스트 데이터로 모델 평가
def evaluate_model(model, test_loader, scaler, feature_names, device, save_path):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_test = np.concatenate(all_targets, axis=0)

    # 정규화 이전으로 변환 (알기 쉽게)
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)

    mse = np.mean((y_test_original - y_pred_original) ** 2) # 평균 제곱 오차
    mae = np.mean(np.abs(y_test_original - y_pred_original)) # 평균 오차
    mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100 # 퍼센트로 오차 표현

    print("\n" + "=" * 60)
    print("Test set evaluation")
    print("=" * 60)
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("=" * 60)

    # Plot predictions
    n_languages = len(feature_names)
    n_cols = 2
    n_rows = (n_languages + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, lang in enumerate(feature_names):
        axes[i].plot(y_test_original[:, i], label='Actual', marker='o', linewidth=2)
        axes[i].plot(y_pred_original[:, i], label='Predicted', marker='s', linewidth=2, alpha=0.7)
        axes[i].set_title(f'{lang.upper()} - Predictions', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Percentage (%)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    for i in range(n_languages, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Prediction plot saved to {save_path}")
    plt.close()

    return mse, mae, mape


def main():
    print("=" * 60)
    print("TIOBE LSTM MODEL TRAINING (PyTorch)")
    print("=" * 60)

    # 하이퍼 파라미터
    SEQUENCE_LENGTH = 36
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    LEARNING_RATE = 0.001 # 아담이 알아서 자동으로 조장해줄것임
    NUM_EPOCHS = 200
    PATIENCE = 30 

    data_dir = Path("/Users/parkjuyong/Desktop/4-1/CareerRoute/ml/trend_tech_ml/data")
    scaled_data_path = data_dir / "scaled_tiobe_data.csv"
    scaler_path = data_dir / "scaler.pkl"

    output_dir = data_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # GPU 사용시
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 데이터 로드
    print("\n1. Loading data")
    df = pd.read_csv(scaled_data_path)
    dates = df['date'].values
    data = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist() # 첫번째 열이 특징(언어들)

    print(f"   Data shape: {data.shape}")
    print(f"   Features: {feature_names}")

    # 정규화 데이터 로드 
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Create sequences
    print("\n2. Creating sequences")
    X, y = create_sequences(data, seq_length=SEQUENCE_LENGTH)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    # 데이터 분할
    print("\n3. Splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")

    # Create datasets and dataloaders
    train_dataset = TiobeDataset(X_train, y_train)
    val_dataset = TiobeDataset(X_val, y_val)
    test_dataset = TiobeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 구축
    print("\n4. Building model...")
    num_features = data.shape[1]

    # 양방향 LSTM 모델 
    model = BidirectionalLSTMModel(
        num_features=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print(f"   Model: BidirectionalLSTMModel")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 손실함수 및 최적화 함수
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # 아담으로 학습률 자동조정
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 학습 시작
    print("\n5. Training model")
    print("=" * 60)

    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

        # 최적화 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   -> Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # 강제종료
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 60)
    print("Training completed")
    print("=" * 60)

    # 학습 과정 시각화
    print("\n6. Plotting training history")
    plot_training_history(history, save_path=str(output_dir / "training_history.png"))

    # 최적 모델 로드 및 평가
    print("\n7. Evaluating best model")
    model.load_state_dict(torch.load(best_model_path))

    mse, mae, mape = evaluate_model(
        model, test_loader, scaler, feature_names, device,
        save_path=str(output_dir / "predictions.png")
    )

    # 학습 정보 저장
    info = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': num_features,
        'feature_names': feature_names,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs_trained': epoch + 1,
        'test_mse': float(mse),
        'test_mae': float(mae),
        'test_mape': float(mape),
        'best_val_loss': float(best_val_loss)
    }

    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining info saved to {output_dir / 'training_info.json'}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"Best model: {best_model_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
