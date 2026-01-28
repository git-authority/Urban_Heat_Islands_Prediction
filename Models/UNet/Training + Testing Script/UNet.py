import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from PIL import Image
from pathlib import Path

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_metric
import xarray as xr

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, "../../../Dataset"))

heatmap_folder = os.path.join(script_dir, "Heatmaps_t2m_gray_fast")
model_save_path = os.path.join(script_dir, "UNetConvLSTM_t2m_best.pth")
eval_out = os.path.join(script_dir, "Eval_UNetConvLSTM_2024")
stats_path = os.path.join(script_dir, "dataset_stats.json")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

# Hyperparameters
input_len = 8
img_size = 224
batch_size = 4
num_epochs = 5
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Device: {device}")
print(f"✅ Dataset Path: {base_dir}")

# ==========================================
# 2. DATA GENERATION & STATS
# ==========================================
def ensure_data_ready():
    """
    Ensures PNGs exist and Global Min/Max stats are available for denormalization.
    """
    # 1. Check if we need to scan for stats (Min/Max)
    if not os.path.exists(stats_path):
        print("⚠️ Stats file missing. Scanning NetCDF to calculate Global Min/Max...")
        years = ['2020', '2021', '2022', '2023', '2024']
        nc_files = []
        for y in years:
            year_dir = os.path.join(base_dir, y)
            if os.path.exists(year_dir):
                for root, _, files in os.walk(year_dir):
                    for f in files:
                        if f.endswith('.nc'):
                            nc_files.append(os.path.join(root, f))

        if not nc_files:
            raise RuntimeError(f"❌ No NetCDF files found in {base_dir}")

        # Scan all files to find global min/max
        all_vals = []
        for path in tqdm(sorted(nc_files), desc="Scanning Stats"):
            try:
                ds = xr.open_dataset(path)
                da = ds['t2m']
                # Subsample to match grid
                da = da.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))
                all_vals.append(da.values.flatten())
            except: pass

        full_arr = np.concatenate(all_vals)
        g_min = float(np.nanmin(full_arr))
        g_max = float(np.nanmax(full_arr))

        with open(stats_path, 'w') as f:
            json.dump({"min": g_min, "max": g_max}, f)
        print(f"✅ Stats saved: Min={g_min:.2f}K, Max={g_max:.2f}K")

    # Load Stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        g_min, g_max = stats['min'], stats['max']

    # 2. Check if PNGs exist
    pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    if len(pngs) > 1000:
        print(f"✅ Found {len(pngs)} existing images. Skipping generation.")
        return g_min, g_max

    print("⚠️ Images missing. Generating PNGs...")

    # Re-gather files for generation
    years = ['2020', '2021', '2022', '2023', '2024']
    nc_files = []
    for y in years:
        year_dir = os.path.join(base_dir, y)
        if os.path.exists(year_dir):
            for root, _, files in os.walk(year_dir):
                for f in files:
                    if f.endswith('.nc'): nc_files.append(os.path.join(root, f))

    all_t2m = []
    for path in tqdm(sorted(nc_files), desc="Reading NetCDF"):
        try:
            ds = xr.open_dataset(path)
            da = ds['t2m']
            if 'valid_time' in da.dims: da = da.rename({'valid_time': 'time'})
            # Flatten Step dimensions if present
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
            print(f"Skipping {path}: {e}")

    t2m_all = xr.concat(all_t2m, dim='time')
    t2m_sub = t2m_all.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))
    t2m_arr = t2m_sub.values
    times = t2m_sub.time.values

    for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
        temp = t2m_arr[i]
        norm = (temp - g_min) / (g_max - g_min)
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)

        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        ts_str = np.datetime_as_string(times[i], unit='h').replace(":", "-")
        img.save(os.path.join(heatmap_folder, f"t2m_{i:05d}_{ts_str}.png"))

    return g_min, g_max

# ==========================================
# 3. DATASET CLASS
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

# ==========================================
# 4. MODEL DEFINITION (UNet + ConvLSTM)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch: mid_ch = out_ch
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv_op(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size//2, bias=bias)
        self.hidden_dim = hidden_dim
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        A = self.conv(combined)
        (i, f, o, g) = torch.split(A, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, True)
        self.hidden_dim = hidden_dim
    def forward(self, input_seq):
        T, B, C, H, W = input_seq.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=input_seq.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=input_seq.device)
        for t in range(T):
            h, c = self.cell(input_seq[t], h, c)
        return h

class UNetConvLSTM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, lstm_hidden=256):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)

        bottleneck_ch = base_ch * 8
        self.convlstm = ConvLSTM(bottleneck_ch, lstm_hidden)
        self.post_lstm = nn.Conv2d(lstm_hidden, bottleneck_ch, 1)

        self.up1 = Up(bottleneck_ch + base_ch*8, base_ch*8)
        self.up2 = Up(base_ch*8 + base_ch*4, base_ch*4)
        self.up3 = Up(base_ch*4 + base_ch*2, base_ch*2)
        self.up4 = Up(base_ch*2 + base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, out_channels, 1)

    def encode_single(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4], x5

    def forward(self, x):
        B, T, C, H, W = x.shape
        bottlenecks = []
        last_skips = None
        for t in range(T):
            skips, bott = self.encode_single(x[:, t])
            bottlenecks.append(bott)
            last_skips = skips

        seq = torch.stack(bottlenecks, dim=0) # T, B, C, H, W
        h_last = self.convlstm(seq)
        dec_feat = self.post_lstm(h_last)

        x = self.up1(dec_feat, last_skips[3])
        x = self.up2(x, last_skips[2])
        x = self.up3(x, last_skips[1])
        x = self.up4(x, last_skips[0])
        return torch.sigmoid(self.outc(x))

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    # 1. Setup Data & Stats
    g_min, g_max = ensure_data_ready()
    temp_range = g_max - g_min
    print(f"📉 Denormalization Key: Min={g_min:.2f}K, Range={temp_range:.2f}K")

    # 2. Gather Files
    pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    dt_re = re.compile(r'.*?_(\d{4}-\d{2}-\d{2}T\d{2})')

    file_times = []
    for p in pngs:
        m = dt_re.search(os.path.basename(p))
        if m:
            ts = np.datetime64(m.group(1)).astype('datetime64[m]').astype(object)
            file_times.append((p, ts))

    file_times.sort(key=lambda x: x[1])
    paths = [x[0] for x in file_times]
    years = [x[1].year for x in file_times]

    # 3. Create Sequences
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
    for y in train_years: train_samples.extend(samples_by_year.get(y, []))

    test_tuples = [([paths[idx] for idx in s], paths[t]) for s,t in samples_by_year.get(test_year, [])]
    train_tuples = [([paths[idx] for idx in s], paths[t]) for s,t in train_samples]

    print(f"Train Samples: {len(train_tuples)} | Test Samples: {len(test_tuples)}")

    # 4. Sequential Split (80/20)
    transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    full_train_ds = HeatmapSeqFromPaths(train_tuples, transform=transform)
    test_ds = HeatmapSeqFromPaths(test_tuples, transform=transform)

    n_total = len(full_train_ds)
    n_train = int(0.8 * n_total)

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_total))

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds, val_indices)

    print(f"Sequential Split: {len(train_ds)} Training / {len(val_ds)} Validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Model & Training
    model = UNetConvLSTM().to(device)

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
                pred = model(seq)
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, tgt in val_loader:
                    val_loss += criterion(model(seq.to(device)), tgt.to(device)).item()

            avg_train = train_loss / len(train_loader.dataset)
            avg_val = val_loss / len(val_loader.dataset)
            print(f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f}")

            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), model_save_path)
                print("🔥 Saved Best Model")

    # 6. Evaluation (Absolute Metrics)
    print("📊 Evaluating (Calculating Absolute Metrics)...")
    model.eval()
    mse_scores, mae_scores, rmse_scores, ssim_scores = [], [], [], []
    dates = []

    with torch.no_grad():
        for i, (seq, tgt) in enumerate(tqdm(test_loader)):
            pred = model(seq.to(device)).cpu().numpy().squeeze()
            tgt_np = tgt.numpy().squeeze()

            # Handle potential NaNs in raw output
            pred = np.nan_to_num(pred, nan=0.0)
            tgt_np = np.nan_to_num(tgt_np, nan=0.0)

            # 1. Denormalize to Kelvin (Abs = Norm * Range + Min)
            pred_abs = pred * temp_range + g_min
            tgt_abs = tgt_np * temp_range + g_min

            # 2. Calculate Absolute Metrics
            diff = pred_abs - tgt_abs
            mae = np.mean(np.abs(diff))
            mse = np.mean(diff**2)
            rmse = np.sqrt(mse)

            # 3. Calculate SSIM (Use normalized 0-1 for stability, matches structure)
            try: s = ssim_metric(tgt_np, pred, data_range=1.0)
            except: s = 0.0

            mse_scores.append(mse)
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            ssim_scores.append(s)

            # Timestamp
            fn = os.path.basename(test_tuples[i][1])
            m = dt_re.search(fn)
            if m: dates.append(np.datetime64(m.group(1)))
            else: dates.append(np.datetime64('NaT'))

    # 7. Save & Plot
    df = pd.DataFrame({
        'time': dates,
        'mae_abs': mae_scores,
        'rmse_abs': rmse_scores,
        'ssim': ssim_scores
    })
    df.to_csv(os.path.join(eval_out, "absolute_results.csv"), index=False)

    # Helper to plot
    def plot_metric(metric_name, values, color, title):
        plt.figure(figsize=(12, 4))
        plt.plot(dates, values, marker='.', linewidth=0.5, markersize=2, color=color)
        plt.title(title)
        plt.xlabel("Time (2024)")
        plt.ylabel(metric_name)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_out, f"{title.replace(' ', '_')}.png"))
        plt.close()

    plot_metric("SSIM", ssim_scores, "tab:blue", "SSIM 2024")
    plot_metric("MAE (Kelvin)", mae_scores, "tab:orange", "Absolute MAE 2024")
    plot_metric("RMSE (Kelvin)", rmse_scores, "tab:red", "Absolute RMSE 2024")

    print(f"🎉 Done! Absolute metrics saved in {eval_out}")

if __name__ == '__main__':
    main()