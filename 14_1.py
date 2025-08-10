import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as ncDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
INPUT_DAYS = 14  # increased input days
OUTPUT_DAYS = 1
CITY_NAME = "Mumbai"
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 7
DATA_FOLDER = "Dataset/2024"


# =========================
# Dataset Class
# =========================
class TempDataset(Dataset):
    def __init__(self, folder_path, input_days, output_days):
        self.input_days = input_days
        self.output_days = output_days
        self.data = []

        for file in os.listdir(folder_path):
            if file.endswith(".nc"):
                ds = ncDataset(os.path.join(folder_path, file))
                temp = np.array(ds.variables["t2m"][:], dtype=np.float32) - 273.15
                temp = temp.flatten()
                temp = temp[~np.isnan(temp)]
                self.data.extend(temp.tolist())

        if len(self.data) == 0:
            raise ValueError("No valid temperature data found in the dataset folder.")

        self.data = np.array(self.data, dtype=np.float32)
        self.min_val = np.nanmin(self.data)
        self.max_val = np.nanmax(self.data)
        self.data = (self.data - self.min_val) / (self.max_val - self.min_val)

    def __len__(self):
        return len(self.data) - self.input_days - self.output_days + 1

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.input_days]
        y = self.data[idx + self.input_days : idx + self.input_days + self.output_days]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# =========================
# Model
# =========================
class TempModel(nn.Module):
    def __init__(self, input_days, output_days):
        super(TempModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_days, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_days),
        )

    def forward(self, x):
        return self.fc(x)


# =========================
# Early Stopping
# =========================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# =========================
# Main Script
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = TempDataset(DATA_FOLDER, INPUT_DAYS, OUTPUT_DAYS)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

model = TempModel(INPUT_DAYS, OUTPUT_DAYS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
early_stopping = EarlyStopping(patience=PATIENCE)

train_losses, val_losses = [], []

print("Starting training...")
for epoch in tqdm(range(EPOCHS), desc="Epochs", position=0):
    model.train()
    train_loss = 0
    batch_bar = tqdm(
        train_loader, desc=f"Train Epoch {epoch+1}", leave=False, position=1
    )
    for x, y in batch_bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_bar.set_postfix(loss=loss.item())

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    tqdm.write(
        f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
    )

    early_stopping.step(avg_val_loss)
    if early_stopping.early_stop:
        tqdm.write("Early stopping triggered.")
        break

# =========================
# Generate Predictions and Plots
# =========================
print("Generating plots...")
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        output = model(x).cpu().numpy()
        preds.extend(output.flatten())
        actuals.extend(y.numpy().flatten())

# Metrics
mae = mean_absolute_error(actuals, preds)
mse = mean_squared_error(actuals, preds)
r2 = r2_score(actuals, preds)

# Plot only first cycle (e.g., first 30 points)
cycle_len = min(30, len(actuals))
plt.figure(figsize=(8, 5))
plt.plot(range(cycle_len), actuals[:cycle_len], label="Actual", marker="o")
plt.plot(range(cycle_len), preds[:cycle_len], label="Predicted", marker="x")
plt.title(
    f"Actual vs Predicted Temperatures\n{CITY_NAME} ({INPUT_DAYS}in-{OUTPUT_DAYS}out)\nMAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}"
)
plt.xlabel("Time Step")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"actual_vs_predicted_{CITY_NAME}_{INPUT_DAYS}in_{OUTPUT_DAYS}out.png", dpi=300
)
plt.close()

# Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"train_vs_val_loss_{CITY_NAME}_{INPUT_DAYS}in_{OUTPUT_DAYS}out.png", dpi=300
)
plt.close()

print(
    f"Plots saved:\n- actual_vs_predicted_{CITY_NAME}_{INPUT_DAYS}in_{OUTPUT_DAYS}out.png\n- train_vs_val_loss_{CITY_NAME}_{INPUT_DAYS}in_{OUTPUT_DAYS}out.png"
)
