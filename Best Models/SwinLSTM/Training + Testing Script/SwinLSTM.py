import os
import re
import glob
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Data Handling Imports
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from skimage.metrics import structural_similarity as ssim_metric

# ==========================================
# 1. CLASSES & DEFINITIONS (Global Scope)
# ==========================================

class HeatmapSeqFromPaths(Dataset):
    def __init__(self, tuples, transform=None):
        self.tuples = tuples
        self.transform = transform
    def __len__(self): return len(self.tuples)
    def _load(self, p):
        img = Image.open(p).convert("L")
        if self.transform:
            return self.transform(img)
        return transforms.ToTensor()(img)
    def __getitem__(self, idx):
        seq_paths, tgt_path = self.tuples[idx]
        seq = torch.stack([self._load(p) for p in seq_paths], dim=0)
        tgt = self._load(tgt_path)
        return seq, tgt

class SwinLSTM(nn.Module):
    def __init__(self, hidden_dim=256, input_len=8, img_size=224):
        super().__init__()
        self.input_len = input_len
        self.img_size = img_size
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, global_pool='avg')
        F = self.encoder.num_features
        self.lstm = nn.LSTM(input_size=F, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, img_size * img_size)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.repeat(1,1,3,1,1).view(B*T,3,H,W)
        feats = self.encoder(x).view(B, T, -1)
        _, (h_n, _) = self.lstm(feats)
        return self.fc(h_n[-1]).view(B, 1, self.img_size, self.img_size)

# ==========================================
# 2. IMAGE GENERATION FUNCTION
# ==========================================
def generate_images(base_dataset_dir, output_folder):
    print(f"⚠️ No images found. Starting automatic generation from: {base_dataset_dir}")

    # 1. Find NetCDF Files
    years = ['2020', '2021', '2022', '2023', '2024']
    nc_files = []
    for y in years:
        year_dir = os.path.join(base_dataset_dir, y)
        if os.path.exists(year_dir):
            for root, _, files in os.walk(year_dir):
                for f in files:
                    if f.endswith('.nc'):
                        nc_files.append(os.path.join(root, f))

    if not nc_files:
        raise RuntimeError(f"❌ Could not find any .nc files in {base_dataset_dir}. Check your path!")

    nc_files = sorted(nc_files)
    print(f"   Found {len(nc_files)} NetCDF files. Processing...")

    # 2. Load Data
    all_t2m = []
    for path in tqdm(nc_files, desc="Reading NetCDFs"):
        try:
            ds = xr.open_dataset(path)
            da = ds['t2m']

            # Standardize Dimensions
            if 'valid_time' in da.dims: da = da.rename({'valid_time': 'time'})
            if 'time' in da.dims and 'step' in da.dims:
                if 'valid_time' in ds:
                    new_time = ds['valid_time'].values.reshape(-1)
                else:
                    new_time = np.arange(da.sizes['time'] * da.sizes['step'])
                da = da.stack(new_time=('time', 'step'))
                da = da.assign_coords(new_time=('new_time', new_time))
                da = da.rename({'new_time': 'time'})
                da = da.drop_vars(['step'], errors='ignore')

            if 'latitude' in da.coords: da = da.sortby('latitude')
            if 'longitude' in da.coords: da = da.sortby('longitude')
            all_t2m.append(da)
        except Exception as e:
            print(f"Skipping corrupt file {path}: {e}")

    # 3. Concatenate & Subsample
    t2m_all = xr.concat(all_t2m, dim='time')
    # Keep odd indices (grid reduction)
    t2m_sub = t2m_all.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))

    # 4. Normalize & Save
    t2m_arr = t2m_sub.values
    times = t2m_sub.time.values

    global_min = float(np.nanmin(t2m_arr))
    global_max = float(np.nanmax(t2m_arr))
    print(f"   Global Temp Range: {global_min:.2f}K - {global_max:.2f}K")

    # Save Stats for later Denormalization
    stats_path = os.path.join(os.path.dirname(output_folder), "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({"min": global_min, "max": global_max}, f)
    print(f"   Stats saved to {stats_path}")

    for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
        temp = t2m_arr[i]
        # Normalize to 0-1
        norm = (temp - global_min) / (global_max - global_min)
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0) # Handle sea/NaNs

        # Save as Grayscale
        img_arr = (norm * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")

        ts_str = np.datetime_as_string(times[i], unit='h').replace(":", "-")
        fname = f"t2m_{i:05d}_{ts_str}.png"
        img.save(os.path.join(output_folder, fname))

    print("✅ Image generation complete!")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # ---------- Config ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = "../../../Dataset"

    heatmap_folder = os.path.join(script_dir, "Heatmaps_t2m_gray_fast")
    model_save_path = os.path.join(script_dir, "SwinLSTM_t2m_best.pth")
    eval_out = os.path.join(script_dir, "Eval_SwinLSTM_2024")
    stats_path = os.path.join(script_dir, "dataset_stats.json")

    os.makedirs(heatmap_folder, exist_ok=True)
    os.makedirs(eval_out, exist_ok=True)

    # Params
    input_len = 8
    img_size = 224
    batch_size = 4
    num_epochs = 5
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Using device: {device}")

    # ---------- 1. Check & Generate Images ----------
    pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    if len(pngs) == 0:
        generate_images(base_dir, heatmap_folder)
        pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))

    print(f"✅ Ready with {len(pngs)} images.")

    # ---------- 1.5 Load Stats for Absolute Metrics ----------
    # If stats file missing (images existed but no json), perform one-time scan
    if not os.path.exists(stats_path):
        print("⚠️ Stats file missing. Scanning NetCDF to recover global min/max...")
        # Re-using generation logic just for stats
        years = ['2020', '2021', '2022', '2023', '2024']
        nc_files_scan = []
        for y in years:
            yd = os.path.join(base_dir, y)
            if os.path.exists(yd):
                for r, _, f in os.walk(yd):
                    for file in f:
                        if file.endswith('.nc'): nc_files_scan.append(os.path.join(r, file))

        all_vals = []
        for p in tqdm(nc_files_scan, desc="Scanning for Stats"):
            try:
                ds = xr.open_dataset(p)['t2m']
                # Subsample to match training grid
                ds = ds.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))
                all_vals.append(ds.values.flatten())
            except: pass

        full_arr = np.concatenate(all_vals)
        global_min = float(np.nanmin(full_arr))
        global_max = float(np.nanmax(full_arr))
        with open(stats_path, 'w') as f:
            json.dump({"min": global_min, "max": global_max}, f)
        print(f"✅ Stats recovered: {global_min} - {global_max}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)
        global_min = stats['min']
        global_max = stats['max']

    # Parse timestamps
    dt_re = re.compile(r'.*?_(\d{4}-\d{2}-\d{2}T\d{2})')
    file_times = []
    for p in pngs:
        m = dt_re.search(os.path.basename(p))
        if not m: continue
        ts = np.datetime64(m.group(1)).astype('datetime64[m]').astype(object)
        file_times.append((p, ts))

    file_times = sorted(file_times, key=lambda x: x[1])
    paths = [ft[0] for ft in file_times]
    times = [ft[1] for ft in file_times]
    years = [t.year for t in times]

    # ---------- 2. Prepare Sequences ----------
    samples_by_year = {}
    for i in range(len(paths) - input_len):
        seq_idxs = list(range(i, i + input_len))
        targ_idx = i + input_len
        y_seq = {years[j] for j in seq_idxs + [targ_idx]}
        if len(y_seq) == 1:
            yr = years[targ_idx]
            samples_by_year.setdefault(yr, []).append((seq_idxs, targ_idx))

    train_years = [2020, 2021, 2022, 2023]
    test_year = 2024

    train_samples = []
    for y in train_years:
        train_samples += samples_by_year.get(y, [])

    test_tuples = [([paths[idx] for idx in s], paths[t]) for s,t in samples_by_year.get(test_year, [])]
    train_tuples = [([paths[idx] for idx in s], paths[t]) for s,t in train_samples]

    print(f"Train Samples: {len(train_tuples)} | Test Samples: {len(test_tuples)}")

    # Dataloaders
    transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    full_train_ds = HeatmapSeqFromPaths(train_tuples, transform=transform)
    test_ds = HeatmapSeqFromPaths(test_tuples, transform=transform)

    # 👇 SEQUENTIAL SPLIT (80% Train, 20% Val)
    n_total = len(full_train_ds)
    n_train = int(0.8 * n_total) # 80% split

    # Create indices list [0, 1, 2, ... n_train] and [n_train, ... end]
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_total))

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds, val_indices)

    print(f"Split: {len(train_ds)} Training / {len(val_ds)} Validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- 3. Training / Loading ----------
    model = SwinLSTM(hidden_dim=256, input_len=input_len, img_size=img_size).to(device)

    if os.path.exists(model_save_path):
        print(f"⚡ Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("🚀 Starting Training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_val = float('inf')

        for epoch in range(1, num_epochs+1):
            model.train()
            train_loss = 0.0
            for seq, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
                seq, tgt = seq.to(device), tgt.to(device)
                optimizer.zero_grad()
                loss = criterion(model(seq), tgt)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, tgt in val_loader:
                    val_loss += criterion(model(seq.to(device)), tgt.to(device)).item()

            print(f"Epoch {epoch}: Train Loss {train_loss/len(train_loader):.5f} | Val Loss {val_loss/len(val_loader):.5f}")

            if (val_loss/len(val_loader)) < best_val:
                best_val = val_loss/len(val_loader)
                torch.save(model.state_dict(), model_save_path)
                print("🔥 Model Saved!")

    # ---------- 4. Evaluation ----------
    print("📊 Evaluating...")
    model.eval()

    # Store absolute metrics
    ssim_scores = []
    mae_scores = []
    rmse_scores = []
    test_dates = []

    with torch.no_grad():
        for i, (seq, tgt) in enumerate(tqdm(test_loader)):
            pred = model(seq.to(device)).cpu().numpy().squeeze()
            tgt_np = tgt.numpy().squeeze()

            # 1. SSIM (Standard on normalized 0-1)
            try:
                s = ssim_metric(tgt_np, pred, data_range=1.0)
            except:
                s = 0.0
            ssim_scores.append(s)

            # 2. Denormalize to Absolute Kelvin
            pred_abs = pred * (global_max - global_min) + global_min
            tgt_abs = tgt_np * (global_max - global_min) + global_min

            # 3. Absolute Metrics
            diff = pred_abs - tgt_abs
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(np.mean(diff**2))

            mae_scores.append(mae)
            rmse_scores.append(rmse)

            # Timestamp
            fn = os.path.basename(test_tuples[i][1])
            m = dt_re.search(fn)
            if m: test_dates.append(np.datetime64(m.group(1)))
            else: test_dates.append(np.datetime64("NaT"))

    # ---------- 5. Plotting (3 Figures) ----------

    # Figure 1: SSIM
    plt.figure(figsize=(12,4))
    plt.plot(test_dates, ssim_scores, marker='.', linestyle='-', linewidth=0.5, markersize=2, color='tab:blue')
    plt.title("SSIM over 2024")
    plt.xlabel("Time")
    plt.ylabel("SSIM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_out, "Figure1_SSIM_2024.png"))
    plt.close()

    # Figure 2: Absolute MAE
    plt.figure(figsize=(12,4))
    plt.plot(test_dates, mae_scores, marker='.', linestyle='-', linewidth=0.5, markersize=2, color='tab:orange')
    plt.title("Absolute MAE (Kelvin) over 2024")
    plt.xlabel("Time")
    plt.ylabel("MAE (K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_out, "Figure2_MAE_Absolute_2024.png"))
    plt.close()

    # Figure 3: Absolute RMSE
    plt.figure(figsize=(12,4))
    plt.plot(test_dates, rmse_scores, marker='.', linestyle='-', linewidth=0.5, markersize=2, color='tab:red')
    plt.title("Absolute RMSE (Kelvin) over 2024")
    plt.xlabel("Time")
    plt.ylabel("RMSE (K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_out, "Figure3_RMSE_Absolute_2024.png"))
    plt.close()

    # Save summary CSV
    df = pd.DataFrame({
        "timestamp": test_dates,
        "ssim": ssim_scores,
        "mae_abs": mae_scores,
        "rmse_abs": rmse_scores
    })
    df.to_csv(os.path.join(eval_out, "evaluation_metrics_absolute.csv"), index=False)

    print(f"🎉 Done! Results in {eval_out}")

# ==========================================
# 4. WINDOWS GUARD
# ==========================================
if __name__ == '__main__':
    main()