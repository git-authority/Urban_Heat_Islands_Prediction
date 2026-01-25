import os
import glob
import re
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from skimage.metrics import structural_similarity as ssim_metric

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Current Script: .../New_Abs/new.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset: .../Dataset (Two levels up)
base_dir = "../../../../Dataset"

# Model: .../Best_SwinLSTM/Model/SwinLSTM_t2m_best.pth (Sibling folder)
model_dir = Path(script_dir).resolve().parents[0] / "Best_SwinLSTM" / "Model"
model_path = model_dir / "SwinLSTM_t2m_best.pth"

# Output folders
heatmap_folder = os.path.join(script_dir, "Temp_Heatmaps_2024")
eval_out = os.path.join(script_dir, "Final_Results_Absolute")
os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

# Settings (Must match training architecture)
input_len = 8
img_size = 224
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Dataset Path: {base_dir}")
print(f"✅ Model Path:   {model_path}")
print(f"✅ Output Path:  {eval_out}")

# ==========================================
# 2. RE-CALCULATE GLOBAL STATS (CRITICAL)
# ==========================================
# We need the exact Min/Max used during training to denormalize correctly.
print("\n[Step 1/5] scanning Dataset to calculate Temperature Range (Kelvin)...")

years = ["2020", "2021", "2022", "2023", "2024"]
nc_files = []

for y in years:
    year_dir = os.path.join(str(base_dir), y)
    if os.path.exists(year_dir):
        for root, _, files in os.walk(year_dir):
            for f in files:
                if f.endswith(".nc"):
                    nc_files.append(os.path.join(root, f))
nc_files = sorted(nc_files)

if not nc_files:
    print("❌ ERROR: No .nc files found! Check your Dataset path.")
    exit()

# Load Data to find Global Min/Max
all_t2m = []
# We only need a quick scan, but we need all data for accurate Min/Max
for path in tqdm(nc_files, desc="Reading NetCDFs"):
    try:
        ds = xr.open_dataset(path)
        da = ds["t2m"]

        # Standardize Dimensions
        if "valid_time" in da.dims: da = da.rename({"valid_time": "time"})
        if "time" in da.dims and "step" in da.dims:
            if "valid_time" in ds:
                new_time = ds["valid_time"].values.reshape(-1)
            else:
                new_time = np.arange(da.sizes["time"] * da.sizes["step"])
            da = da.stack(new_time=("time", "step"))
            da = da.assign_coords(new_time=("new_time", new_time))
            da = da.rename({"new_time": "time"})
            da = da.drop_vars(["step"], errors="ignore")

        if "latitude" in da.coords: da = da.sortby("latitude")
        if "longitude" in da.coords: da = da.sortby("longitude")

        # Keep only required years to save memory if needed, but training used all
        all_t2m.append(da)
    except Exception as e:
        print(f"Skipping {path}: {e}")

t2m_all = xr.concat(all_t2m, dim="time")

# Subsample (Odd indices only, same as training)
t2m_sub = t2m_all.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))
t2m_arr = t2m_sub.values

# CALCULATE GLOBAL STATS
GLOBAL_MIN = float(np.nanmin(t2m_arr))
GLOBAL_MAX = float(np.nanmax(t2m_arr))
TEMP_RANGE = GLOBAL_MAX - GLOBAL_MIN
print(f"🌡️ Global Min: {GLOBAL_MIN:.4f} K")
print(f"🌡️ Global Max: {GLOBAL_MAX:.4f} K")
print(f"🌡️ Range:      {TEMP_RANGE:.4f} K")

# Calculate Sea Mask (NaNs are sea)
sea_mask_np = np.isnan(t2m_arr).all(axis=0)
land_mask = Image.fromarray((~sea_mask_np).astype(np.uint8)*255).resize((224, 224), Image.NEAREST)
land_mask = np.array(land_mask) > 0

# ==========================================
# 3. GENERATE TEST IMAGES (2024 ONLY)
# ==========================================
print("\n[Step 2/5] Generating 2024 Test Images...")
times = t2m_sub.time.values
count = 0

for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
    ts_str = np.datetime_as_string(times[i], unit="h").replace(":", "-")
    # Only save 2024 files (and late 2023 for sequence context if needed)
    if "2024" in ts_str:
        temp = t2m_arr[i]
        # Normalize to 0-1
        norm = np.nan_to_num((temp - GLOBAL_MIN) / TEMP_RANGE, nan=0.0)
        norm = np.clip(norm, 0.0, 1.0)
        img_arr = (norm * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        img.save(os.path.join(heatmap_folder, f"t2m_{i:05d}_{ts_str}.png"))
        count += 1

print(f"✅ Generated {count} images for 2024.")

# ==========================================
# 4. MODEL SETUP
# ==========================================
class SwinLSTM(nn.Module):
    def __init__(self, hidden_dim=256, input_len=8, img_size=224):
        super().__init__()
        # Matches training architecture
        self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        self.lstm = nn.LSTM(input_size=self.encoder.num_features, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, img_size * img_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W)
        feats = self.encoder(x).view(B, T, -1)
        _, (h_n, _) = self.lstm(feats)
        return self.fc(h_n[-1]).view(B, 1, 224, 224)

class HeatmapSeqDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq_paths, tgt_path = self.samples[idx]
        seq = torch.stack([self.load_img(p) for p in seq_paths], dim=0)
        tgt = self.load_img(tgt_path)
        return seq, tgt, tgt_path
    def load_img(self, p):
        img = Image.open(p).convert("L")
        return transforms.ToTensor()(self.transform(img)) if self.transform else transforms.ToTensor()(img)

print("\n[Step 3/5] Loading Model...")
model = SwinLSTM().to(device)

if os.path.exists(model_path):
    print(f"⚡ Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"❌ ERROR: Model not found at {model_path}")
    exit()

# Prepare Dataset
pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
test_tuples = []
for i in range(len(pngs) - input_len):
    s_paths = pngs[i : i + input_len]
    t_path = pngs[i + input_len]
    if "2024" in t_path:
        test_tuples.append((s_paths, t_path))

transform = transforms.Resize((224, 224))
test_ds = HeatmapSeqDataset(test_tuples, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# ==========================================
# 5. EVALUATION (DENORMALIZATION)
# ==========================================
print("\n[Step 4/5] Evaluating and Denormalizing...")
model.eval()
results = []
dt_re = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2})")
alpha = 1.0 # Pure model prediction

with torch.no_grad():
    for seq, tgt, path in tqdm(test_loader, desc="Testing"):
        pred = model(seq.to(device)).cpu().numpy()[0,0]
        actual = tgt.numpy()[0,0]

        # 1. Denormalize to Kelvin (Land Only)
        p_abs = (pred[land_mask] * TEMP_RANGE) + GLOBAL_MIN
        a_abs = (actual[land_mask] * TEMP_RANGE) + GLOBAL_MIN

        # 2. Calc Metrics
        mae = np.mean(np.abs(p_abs - a_abs))
        rmse = np.sqrt(np.mean((p_abs - a_abs)**2))

        # 3. Calc SSIM (Use normalized 0-1 range for structure comparison)
        # Neutralize sea pixels for SSIM calculation
        a_ssim = actual.astype(np.float64)
        p_ssim = pred.astype(np.float64)
        p_ssim[~land_mask] = a_ssim[~land_mask]
        ssim = ssim_metric(a_ssim, p_ssim, data_range=1.0)

        # 4. Get Timestamp
        m = dt_re.search(os.path.basename(path[0]))
        ts = m.group(1) if m else "Unknown"

        results.append({"time": ts, "mae": mae, "rmse": rmse, "ssim": ssim})

# ==========================================
# 6. PLOTTING (Stem-like Style)
# ==========================================
print("\n[Step 5/5] Saving Plots...")
df = pd.DataFrame(results)
df.to_csv(os.path.join(eval_out, "absolute_test_metrics_2024.csv"), index=False)

# Convert string time to datetime objects for better plotting
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H')

def plot_timeseries(times, values, title, ylabel, filename, color):
    plt.figure(figsize=(15, 5))

    # Use standard plot with markers to simulate the "stem" look
    # linewidth=0.5 makes lines thin, marker='.' adds the dots
    plt.plot(times, values, color=color, marker='.', linestyle='-', linewidth=0.5, markersize=4)

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Time (2024)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate() # Rotation

    plt.tight_layout()
    plt.savefig(os.path.join(eval_out, filename), dpi=300)
    plt.close()

# Plot RMSE
plot_timeseries(df['time'], df['rmse'],
                "Figure 3: Time series of RMSE for test data (2024) - Absolute Kelvin",
                "RMSE (Kelvin)", "Figure3_RMSE_timeseries_2024.png", 'blue')

# Plot MAE
plot_timeseries(df['time'], df['mae'],
                "Figure 2: Time series of MAE for test data (2024) - Absolute Kelvin",
                "MAE (Kelvin)", "Figure2_MAE_timeseries_2024.png", 'tab:red')

# Plot SSIM
plot_timeseries(df['time'], df['ssim'],
                "Figure 1: Time series of SSIM for test data (2024)",
                "SSIM", "Figure1_SSIM_timeseries_2024.png", 'green')

print(f"\n🎉 SUCCESS! Results saved to: {eval_out}")