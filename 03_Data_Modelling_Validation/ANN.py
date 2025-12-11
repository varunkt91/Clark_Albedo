# ann_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




# class RegressionNet(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], dropout_p=0.1):
#         super(RegressionNet, self).__init__()
#         layers = []
#         prev_dim = input_dim

#         # Hidden layers: Linear → BatchNorm → ReLU → Dropout
#         for h in hidden_dims:
#             layers.append(nn.Linear(prev_dim, h))
#             layers.append(nn.BatchNorm1d(h))   # batch normalization
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(p=dropout_p))
#             prev_dim = h

#         # Final output layer (no BatchNorm, no Dropout, no ReLU)
#         layers.append(nn.Linear(prev_dim, output_dim))

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32], dropout_p=0.10):
        super(RegressionNet, self).__init__()
        layers = []
        prev_dim = input_dim

        # Hidden layers: Linear → BatchNorm → ReLU
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev_dim = h

        # 🔹 Apply Dropout only before final output layer
        layers.append(nn.Dropout(p=dropout_p))

        # Final output layer (no BatchNorm, no ReLU)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Regression Metrics
# -----------------------------
def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mean_obs = np.mean(y_true)
    perc_rmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan
    ubrmse = np.sqrt(np.mean(((y_pred - y_true) - np.mean(y_pred - y_true))**2))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MSE": mse,
        "RMSE%": perc_rmse,
        "PBIAS%": pbias,
        "ubRMSE": ubrmse,
    }


# -----------------------------
# Training Function
# # -----------------------------
# def train_model(X_train, y_train, X_val, y_val,
#                 input_dim, output_dim,
#                 batch_size=512, epochs=20, lr=0.001,
#                 hidden_dims=[128, 64]):

#     # Datasets
#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # Model
#     model = RegressionNet(input_dim, output_dim, hidden_dims)
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr)

#     history = {"train_loss": [], "val_loss": []}

#     for epoch in range(epochs):
#         # Train
#         model.train()
#         train_losses = []
#         for Xb, yb in train_loader:
#             optimizer.zero_grad()
#             preds = model(Xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             train_losses.append(loss.item())

#         # Validation
#         model.eval()
#         val_losses = []
#         with torch.no_grad():
#             for Xb, yb in val_loader:
#                 preds = model(Xb)
#                 loss = criterion(preds, yb)
#                 val_losses.append(loss.item())

#         train_loss = np.mean(train_losses)
#         val_loss = np.mean(val_losses)
#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#     return model, history


from torch.utils.tensorboard import SummaryWriter
import os

# def train_model(X_train, y_train, X_val, y_val,
#                 input_dim, output_dim,
#                 batch_size=512, epochs=20, lr=0.001,
#                 hidden_dims=[128, 64],
#                 log_dir="runs/NN_experiment"):
    
#     # Create log directory
#     os.makedirs(log_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=log_dir)

#     # Datasets
#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # Model
#     model = RegressionNet(input_dim, output_dim, hidden_dims)
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr)

#     history = {"train_loss": [], "val_loss": []}

#     for epoch in range(epochs):
#         # Train
#         model.train()
#         train_losses = []
#         for Xb, yb in train_loader:
#             optimizer.zero_grad()
#             preds = model(Xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             train_losses.append(loss.item())

#         # Validation
#         model.eval()
#         val_losses = []
#         with torch.no_grad():
#             for Xb, yb in val_loader:
#                 preds = model(Xb)
#                 loss = criterion(preds, yb)
#                 val_losses.append(loss.item())

#         train_loss = np.mean(train_losses)
#         val_loss = np.mean(val_losses)
#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)

#         # Log losses to TensorBoard
#         writer.add_scalar("Loss/Train", train_loss, epoch)
#         writer.add_scalar("Loss/Validation", val_loss, epoch)

#         # Log weights and gradients
#         for name, param in model.named_parameters():
#             writer.add_histogram(f"Weights/{name}", param, epoch)
#             if param.grad is not None:
#                 writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#     writer.close()
#     return model, history




######### WITH TENSARBOARD IMPLEMENTED
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_absolute_error

# # -----------------------------
# # Training Function
# # -----------------------------
# def train_model(X_train, y_train, X_val, y_val,
#                 input_dim, output_dim,
#                 batch_size=512, epochs=20, lr=0.001,
#                 hidden_dims=[128, 64],
#                 dropout_p=0.1,
#                 log_dir="runs/NN_experiment"):

#     # Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Create log directory
#     os.makedirs(log_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=log_dir)

#     # Datasets & loaders
#     train_dataset = TensorDataset(X_train, y_train)
#     val_dataset = TensorDataset(X_val, y_val)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # Model
#     model = RegressionNet(input_dim, output_dim, hidden_dims=hidden_dims, dropout_p=dropout_p).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr)

#     history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": [], "train_mse": [], "val_mse": []}

#     for epoch in range(epochs):
#         # ---------------- Train ----------------
#         model.train()
#         train_losses, y_true_train, y_pred_train = [], [], []

#         for Xb, yb in train_loader:
#             Xb, yb = Xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             preds = model(Xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()

#             train_losses.append(loss.item())
#             y_true_train.append(yb.cpu().numpy())
#             y_pred_train.append(preds.detach().cpu().numpy())

#         y_true_train = np.vstack(y_true_train)
#         y_pred_train = np.vstack(y_pred_train)

#         train_loss = np.mean(train_losses)
#         train_r2 = r2_score(y_true_train, y_pred_train)
#         train_mse = mean_squared_error(y_true_train, y_pred_train)

#         # ---------------- Validation ----------------
#         model.eval()
#         val_losses, y_true_val, y_pred_val = [], [], []
#         with torch.no_grad():
#             for Xb, yb in val_loader:
#                 Xb, yb = Xb.to(device), yb.to(device)
#                 preds = model(Xb)
#                 loss = criterion(preds, yb)
#                 val_losses.append(loss.item())
#                 y_true_val.append(yb.cpu().numpy())
#                 y_pred_val.append(preds.cpu().numpy())

#         y_true_val = np.vstack(y_true_val)
#         y_pred_val = np.vstack(y_pred_val)

#         val_loss = np.mean(val_losses)
#         val_r2 = r2_score(y_true_val, y_pred_val)
#         val_mse = mean_squared_error(y_true_val, y_pred_val)

#         # Save history
#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["train_r2"].append(train_r2)
#         history["val_r2"].append(val_r2)
#         history["train_mse"].append(train_mse)
#         history["val_mse"].append(val_mse)

#         # Log to TensorBoard
#         writer.add_scalar("Loss/Train", train_loss, epoch)
#         writer.add_scalar("Loss/Validation", val_loss, epoch)
#         writer.add_scalar("R2/Train", train_r2, epoch)
#         writer.add_scalar("R2/Validation", val_r2, epoch)
#         writer.add_scalar("MSE/Train", train_mse, epoch)
#         writer.add_scalar("MSE/Validation", val_mse, epoch)

#         print(f"Epoch {epoch+1}/{epochs} | "
#               f"Train Loss: {train_loss:.4f}, R2: {train_r2:.4f}, MSE: {train_mse:.4f} | "
#               f"Val Loss: {val_loss:.4f}, R2: {val_r2:.4f}, MSE: {val_mse:.4f}")

#     writer.close()
#     return model, history


###################### NN with epoch stop

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_model(X_train, y_train, X_val, y_val,
                input_dim, output_dim,
                batch_size=512, epochs=100, lr=0.001,
                hidden_dims=[128, 64], dropout_p=0.1,
                patience=10):  # <-- stop if no improvement for N epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = RegressionNet(input_dim, output_dim, hidden_dims=hidden_dims, dropout_p=dropout_p).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        # ---------------- Train ----------------
        model.train()
        train_losses, y_true_train, y_pred_train = [], [], []

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            y_true_train.append(yb.cpu().numpy())
            y_pred_train.append(preds.detach().cpu().numpy())

        train_loss = np.mean(train_losses)
        train_r2 = r2_score(np.vstack(y_true_train), np.vstack(y_pred_train))

        # ---------------- Validation ----------------
        model.eval()
        val_losses, y_true_val, y_pred_val = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
                y_true_val.append(yb.cpu().numpy())
                y_pred_val.append(preds.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_r2 = r2_score(np.vstack(y_true_val), np.vstack(y_pred_val))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_r2"].append(train_r2)
        history["val_r2"].append(val_r2)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, R2: {train_r2:.4f} | "
              f"Val Loss: {val_loss:.4f}, R2: {val_r2:.4f}")

        # ---------------- Early Stopping ----------------
        if val_loss < best_val_loss - 1e-5:  # tiny tolerance to avoid floating jitter
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⛔ Stopping early at epoch {epoch+1}: no improvement for {patience} epochs\n")
            break

    # Load best weights
    model.load_state_dict(best_state)
    return model, history




# -----------------------------
# Plot Training History
# -----------------------------
def plot_history(history, save_path=None):
    epochs = len(history["train_loss"])
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history["train_loss"], label="Train Loss")
    plt.plot(range(epochs), history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs"); plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend(); plt.grid(True)

    # R2 Score
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history["train_r2"], label="Train R2")
    plt.plot(range(epochs), history["val_r2"], label="Val R2")
    plt.xlabel("Epochs"); plt.ylabel("R2 Score")
    plt.title("Training and Validation R2")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()