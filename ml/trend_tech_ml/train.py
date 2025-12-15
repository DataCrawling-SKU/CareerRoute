import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import LSTMModel, BidirectionalLSTMModel, AttentionLSTMModel


class TiobeDataset(Dataset):
    """PyTorch Dataset for TIOBE time series data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length=12):
    """
    Create sequences for LSTM training

    Args:
        data: numpy array (n_samples, n_features)
        seq_length: number of time steps to look back

    Returns:
        X: (n_samples - seq_length, seq_length, n_features)
        y: (n_samples - seq_length, n_features)
    """
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation, and test sets"""
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
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mae = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_mae += torch.mean(torch.abs(y_pred - y_batch)).item()

    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)

    return avg_loss, avg_mae


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
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


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
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


def evaluate_model(model, test_loader, scaler, feature_names, device, save_path):
    """Evaluate model and plot predictions"""
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

    # Inverse transform to original scale
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)

    # Calculate metrics
    mse = np.mean((y_test_original - y_pred_original) ** 2)
    mae = np.mean(np.abs(y_test_original - y_pred_original))
    mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (Original Scale)")
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

    # Configuration
    SEQUENCE_LENGTH = 12
    BATCH_SIZE = 8
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    PATIENCE = 30

    # Paths
    data_dir = Path("/Users/parkjuyong/Desktop/4-1/CareerRoute/ml/trend_tech_ml/data")
    scaled_data_path = data_dir / "scaled_tiobe_data.csv"
    scaler_path = data_dir / "scaler.pkl"

    output_dir = data_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(scaled_data_path)
    dates = df['date'].values
    data = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist()

    print(f"   Data shape: {data.shape}")
    print(f"   Features: {feature_names}")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Create sequences
    print("\n2. Creating sequences...")
    X, y = create_sequences(data, seq_length=SEQUENCE_LENGTH)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    # Split data
    print("\n3. Splitting data...")
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

    # Build model
    print("\n4. Building model...")
    num_features = data.shape[1]

    # Choose model (change here to try different models)
    model = BidirectionalLSTMModel(
        num_features=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print(f"   Model: BidirectionalLSTMModel")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training
    print("\n5. Training model...")
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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   -> Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Plot training history
    print("\n6. Plotting training history...")
    plot_training_history(history, save_path=str(output_dir / "training_history.png"))

    # Load best model and evaluate
    print("\n7. Evaluating best model...")
    model.load_state_dict(torch.load(best_model_path))

    mse, mae, mape = evaluate_model(
        model, test_loader, scaler, feature_names, device,
        save_path=str(output_dir / "predictions.png")
    )

    # Save training info
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
    print("ALL DONE!")
    print("=" * 60)
    print(f"Best model: {best_model_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
