import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ncDataset
import os


# Dataset class
class TempDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        for file in os.listdir(folder_path):
            if file.endswith(".nc"):
                try:
                    ds = ncDataset(os.path.join(folder_path, file))
                    temp = np.array(ds.variables["t2m"][:], dtype=np.float32)
                    temp = np.nan_to_num(temp, nan=np.nanmean(temp))  # Replace NaN
                    self.data.extend(temp.flatten())
                except Exception as e:
                    print(f"Skipping {file}: {e}")

        self.data = np.array(self.data, dtype=np.float32)
        if len(self.data) == 0:
            raise ValueError("No valid temperature data found.")

        self.min_val = np.nanmin(self.data)
        self.max_val = np.nanmax(self.data)
        if self.max_val == self.min_val:
            raise ValueError("Data has no variation; normalization will fail.")

        # Normalize to [0,1]
        self.data = (self.data - self.min_val) / (self.max_val - self.min_val)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return torch.tensor([self.data[idx]]), torch.tensor([self.data[idx + 1]])


# Model
class TempPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = TempDataset("Dataset/2024")
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TempPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR to prevent NaN

epochs = 50
losses = []
actual_vals = []
predicted_vals = []

print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        if torch.isnan(loss):
            raise ValueError("Loss became NaN. Check your data for invalid values.")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

# Predictions for plotting
with torch.no_grad():
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        actual_vals.extend(y.cpu().numpy())
        predicted_vals.extend(outputs.cpu().numpy())

# Denormalize
actual_vals = (
    np.array(actual_vals) * (dataset.max_val - dataset.min_val) + dataset.min_val
)
predicted_vals = (
    np.array(predicted_vals) * (dataset.max_val - dataset.min_val) + dataset.min_val
)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.plot(actual_vals[:200], label="Actual")
plt.plot(predicted_vals[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Temperatures")
plt.savefig("actual_vs_predicted.png")
plt.close()

# Plot Error
errors = actual_vals - predicted_vals
plt.figure(figsize=(8, 5))
plt.plot(errors[:200], label="Error")
plt.legend()
plt.title("Prediction Error")
plt.savefig("error_plot.png")
plt.close()

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.savefig("loss_plot.png")
plt.close()

print("Plots saved: actual_vs_predicted.png, error_plot.png, loss_plot.png")
