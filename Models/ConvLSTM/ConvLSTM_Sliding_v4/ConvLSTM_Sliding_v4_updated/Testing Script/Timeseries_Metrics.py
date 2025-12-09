#!/usr/bin/env python3
"""
ConvLSTM (Residual + Refine) — INFERENCE-ONLY SCRIPT (2024 ONLY)
Loads:
    ./best_model.pth
    ./climatology.npy
Runs inference on 2024 dataset and produces:
    - SSIM time series
    - MAE time series
    - RMSE time series
Timestamps are synthetic but span ONLY Jan–Dec 2024.
"""

import os
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"

# ---------------- CONFIG ----------------
folder_path = "../../../../Dataset/2024"
model_path = "./best_model.pth"
clim_path = "./climatology.npy"

out_dir = "./Timeseries_Metrics_2024"
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3
seed = 42

USE_NORMALIZATION = True
hidden_dim = 192
num_layers = 3
kernel_size = 5
dropout_p = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- MODEL ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, dropout_p=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad
        )
        self.dropout_p = dropout_p
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)

    def forward(self, x, hidden):
        h, c = hidden
        comb = torch.cat([x, h], dim=1)
        conv = self.conv(comb)

        ci, cf, co, cg = torch.split(conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        o = torch.sigmoid(co)
        g = torch.tanh(cg)

        cnext = f * c + i * g
        hnext = o * torch.tanh(cnext)
        hnext = self.gn(hnext)
        return hnext, cnext

    def init_hidden(self, b, spatial, device):
        H, W = spatial
        return (
            torch.zeros(b, self.hidden_dim, H, W, device=device),
            torch.zeros(b, self.hidden_dim, H, W, device=device),
        )


class ResidualConvLSTMWithRefine(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, kernel_size, dropout_p):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                ConvLSTMCell(
                    in_dim, hidden_dim, kernel_size=kernel_size, dropout_p=dropout_p
                )
            )
        self.layers = nn.ModuleList(layers)

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        hiddens = [l.init_hidden(B, (H, W), x.device) for l in self.layers]

        last = None
        for t in range(C):
            inp = x[:, t : t + 1]
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + inp
                inp = hnext
                hiddens[li] = (hnext, cnext)
            last = inp

        return self.refine(last)


# ---------------- DATASET ----------------
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, folder, input_len, target_offset, sample_step):

        self.frames = []
        files = sorted([f for f in os.listdir(folder) if f.endswith(".nc")])
        if not files:
            raise RuntimeError("No .nc files found")

        for fn in files:
            ds = Dataset(os.path.join(folder, fn))
            if "t2m" not in ds.variables:
                ds.close()
                continue

            arr = np.array(ds["t2m"][:], dtype=np.float32)
            if arr.ndim == 3:
                for k in range(arr.shape[0]):
                    self.frames.append(arr[k])
            else:
                self.frames.append(arr)
            ds.close()

        # temporal downsampling
        self.frames = self.frames[::sample_step]

        # ---- NEW: synthetic timestamps that stay inside 2024 only ----
        n_frames = len(self.frames)
        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 12, 31, 23, 59)
        if n_frames <= 1:
            self.times = [start] * n_frames
        else:
            total_sec = (end - start).total_seconds()
            self.times = [
                start + timedelta(seconds=total_sec * i / (n_frames - 1))
                for i in range(n_frames)
            ]
        # ---------------------------------------------------------------

        stack = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stack).all(axis=0)
        self.H, self.W = self.frames[0].shape

        self.input_len = input_len
        self.target_offset = target_offset

        self.starts = [
            s for s in range(len(self.frames) - input_len - target_offset + 1)
        ]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len

        X = np.stack(self.frames[s:e], axis=0).astype(np.float32)
        y = self.frames[e - 1 + self.target_offset].astype(np.float32)

        mask = self.sea_mask

        # Fill NaNs
        X_f = np.empty_like(X)
        for t in range(X.shape[0]):
            land_vals = X[t][~mask]
            mean_fill = np.nanmean(land_vals) if land_vals.size else 0.0
            X_f[t] = np.where(np.isnan(X[t]), mean_fill, X[t])

        land_vals_y = y[~mask]
        mean_fill_y = np.nanmean(land_vals_y) if land_vals_y.size else 0.0
        y_f = np.where(np.isnan(y), mean_fill_y, y).astype(np.float32)

        timestamp = self.times[e - 1 + self.target_offset]

        return (
            torch.from_numpy(X_f).float(),
            torch.from_numpy(y_f).unsqueeze(0).float(),
            timestamp,
        )


# ---------------- NORMALIZATION ----------------
def compute_norm(dataset, train_idx, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    C = climatology.astype(np.float64)

    for i in train_idx:
        X, y, _ = dataset[i]
        Xn = X.numpy() - C
        yn = y.numpy()[0] - C
        arr = np.concatenate([Xn.ravel(), yn.ravel()])
        s += arr.sum()
        ss += (arr**2).sum()
        cnt += arr.size

    mean = s / cnt
    std = np.sqrt(max(ss / cnt - mean * mean, 1e-12))
    return float(mean), float(std)


# ---------------- LOAD DATASET ----------------
print("Loading dataset...")
dataset = EvalDataset(folder_path, input_len, target_offset, SAMPLE_STEP)
sea_mask = dataset.sea_mask
H, W = dataset.H, dataset.W

# Load climatology
if not os.path.exists(clim_path):
    raise FileNotFoundError("climatology.npy missing!")

climatology = np.load(clim_path).astype(np.float32)
clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)

# norm stats from random split (same as training style)
np.random.seed(seed)
indices = np.arange(len(dataset))
np.random.shuffle(indices)

split = int(len(dataset) * 0.18)
val_idx = indices[:split]
train_idx = indices[split:]

norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    print("Computing normalization...")
    norm_mean, norm_std = compute_norm(dataset, train_idx, climatology)
print("norm_mean, norm_std:", norm_mean, norm_std)


# ---------------- LOAD MODEL ----------------
print("Loading model weights...")
model = ResidualConvLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout_p=dropout_p,
).to(device)

state = torch.load(model_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()


# ---------------- INFERENCE ----------------
times = []
ssim_list = []
mae_list = []
rmse_list = []

print("Running inference...")
with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        X, y, ts = dataset[idx]
        X = X.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        Xn = X - clim_t
        if USE_NORMALIZATION:
            Xn = (Xn - norm_mean) / norm_std

        pred = model(Xn)

        if USE_NORMALIZATION:
            pred = pred * norm_std + norm_mean
        pred = pred + clim_t

        p = pred[0, 0].cpu().numpy()
        a = y[0, 0].cpu().numpy()

        p_l = p[~sea_mask]
        a_l = a[~sea_mask]

        diff = a_l - p_l
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(mse)

        # SSIM
        if ssim_fn is not None:
            p2 = p.copy()
            p2[sea_mask] = a[sea_mask]
            dr = a.max() - a.min()
            dr = dr if dr > 0 else 1e-6
            try:
                ssim_val = ssim_fn(a, p2, data_range=dr)
            except Exception:
                ssim_val = np.nan
        else:
            ssim_val = np.nan

        times.append(ts)
        ssim_list.append(ssim_val)
        mae_list.append(mae)
        rmse_list.append(rmse)


# ---------------- PLOTS ----------------
def plot_timeseries(times, vals, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(times, vals, "o", markersize=3)
    for t, v in zip(times, vals):
        ax.vlines(t, 0, v, colors="C0", alpha=0.35, linewidth=0.7)

    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (2024)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=35)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


plot_timeseries(
    times,
    ssim_list,
    "Figure 1: SSIM Time Series (2024) - ConvLSTM",
    "SSIM",
    "Figure1_SSIM_2024.png",
)

plot_timeseries(
    times,
    mae_list,
    "Figure 2: MAE Time Series (2024) - ConvLSTM",
    "MAE",
    "Figure2_MAE_2024.png",
)

plot_timeseries(
    times,
    rmse_list,
    "Figure 3: RMSE Time Series (2024) - ConvLSTM",
    "RMSE",
    "Figure3_RMSE_2024.png",
)

print("\n🎉 DONE — All 3 time-series plots saved!")
