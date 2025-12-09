#!/usr/bin/env python3
# SwinLSTM_Sliding_v1_with_timeseries_metrics.py
# Updated: dataset returns timestamps; training/validation adjusted; test-time series metrics + plots saved.

import os
import csv
import numpy as np
from netCDF4 import Dataset, num2date
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
import pandas as pd

# Optional imports
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

# ---------------- CONFIG ----------------
folder_path = "../.././Dataset"
out_dir = "SwinLSTM_Sliding_v4"
os.makedirs(out_dir, exist_ok=True)

# Temporal downsampling and mapping
input_len = 8
target_offset = 4
SAMPLE_STEP = 3  # downsample the raw frames by 3 (every 3 hours)

# Training hyperparams
batch_size = 6
lr = 8e-5
epochs = 80
val_split = 0.20
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True  # climatology-based zscore on anomalies
PRINT_EVERY = 1

# Model hyperparams
hidden_dim = 192
num_layers = 3
dropout_p = 0.05

# SwinLSTM-specific
WINDOW_SIZE = 4
ATTN_HEADS = 4

# Loss weights
SSIM_WEIGHT = 0.0
GRAD_WEIGHT = 0.8

plt.rcParams["font.family"] = "Times New Roman"
DPI = 200
# ---------------- END CONFIG ----------------

torch.manual_seed(seed)
np.random.seed(seed)
print(f"✅ Using device: {device}")


# ---------------- Dataset (now returns timestamp per sample) ----------------
class SlidingMaskDataset(TorchDataset):
    """
    Returns (inp_tensor, tgt_tensor, ts_epoch_int)
    Tries to read time variable from NetCDF; if missing, creates placeholders properly spaced by sample_step hours.
    """

    def __init__(
        self,
        folder_path,
        input_len=8,
        target_offset=4,
        sample_step=1,
        include_years=None,
        exclude_years=None,
        fallback_year=None,
    ):
        import os

        self.frames = []
        self.frame_times = []  # python datetimes or None
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

        month_map = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        entries = sorted(os.listdir(folder_path))
        year_dirs = []
        for e in entries:
            p = os.path.join(folder_path, e)
            if os.path.isdir(p):
                year_dirs.append(e)

        if include_years is not None:
            year_dirs = [y for y in year_dirs if str(y) in set(map(str, include_years))]
        if exclude_years is not None:
            year_dirs = [
                y for y in year_dirs if str(y) not in set(map(str, exclude_years))
            ]

        try:
            year_dirs = sorted(year_dirs, key=lambda x: int(x))
        except Exception:
            year_dirs = sorted(year_dirs)

        files_paths = []
        for y in year_dirs:
            yd = os.path.join(folder_path, y)
            if not os.path.isdir(yd):
                continue
            month_files = [f for f in sorted(os.listdir(yd)) if f.endswith(".nc")]

            def month_key(fn):
                name = os.path.splitext(fn)[0].lower()
                return month_map.get(name, 999), fn

            month_files = sorted(month_files, key=month_key)
            for mf in month_files:
                files_paths.append(os.path.join(yd, mf))

        if len(files_paths) == 0:
            raise ValueError(
                f"No .nc files found in {folder_path} for years {year_dirs}"
            )

        # iterate files, read frames and times if available
        for path in files_paths:
            fn = os.path.basename(path)
            try:
                ds = Dataset(path)
            except Exception as e:
                print(f"Skipping {fn} (open failed): {e}")
                continue
            if "t2m" not in ds.variables:
                ds.close()
                continue
            var = ds.variables["t2m"]
            arr = np.array(var[:])
            # handle fill/missing
            if hasattr(var, "_FillValue"):
                arr = np.where(arr == var._FillValue, np.nan, arr)
            if hasattr(var, "missing_value"):
                arr = np.where(arr == var.missing_value, np.nan, arr)

            # try to read time var from file (common name 'time')
            times = None
            if "time" in ds.variables:
                try:
                    tvar = ds.variables["time"]
                    times = num2date(tvar[:], units=getattr(tvar, "units", None))
                    # convert to python datetime if needed
                    times = [(t if hasattr(t, "year") else None) for t in times]
                except Exception:
                    times = None

            if arr.ndim == 3:
                Tloc = arr.shape[0]
                for t in range(Tloc):
                    self.frames.append(arr[t].astype(np.float32))
                    if times is not None and len(times) == Tloc:
                        self.frame_times.append(times[t])
                    else:
                        self.frame_times.append(None)
            elif arr.ndim == 2:
                self.frames.append(arr.astype(np.float32))
                if times is not None and len(times) >= 1:
                    self.frame_times.append(times[0])
                else:
                    self.frame_times.append(None)
            ds.close()

        if len(self.frames) == 0:
            raise ValueError("No frames loaded")

        # downsample in time (apply equally to frames and times)
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]
            self.frame_times = self.frame_times[:: self.sample_step]

        # ensure consistent shape
        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        # if any timestamps missing, create placeholders spaced by sample_step hours
        if any(t is None for t in self.frame_times):
            # pick fallback year
            fy = fallback_year if fallback_year is not None else None
            if fy is None and len(year_dirs) > 0:
                try:
                    fy = int(year_dirs[0])
                except Exception:
                    fy = 2000
            if fy is None:
                fy = 2000
            total = len(self.frames)
            base = datetime(fy, 1, 1, tzinfo=timezone.utc)
            # crucial fix: placeholders spaced by sample_step hours (not 1 hour)
            times = [base + timedelta(hours=i * self.sample_step) for i in range(total)]
            for i in range(total):
                if self.frame_times[i] is None:
                    self.frame_times[i] = times[i]

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)

        # sliding windows
        starts = []
        for s in range(len(self.frames) - input_len - target_offset + 1):
            starts.append(s)
        if len(starts) == 0:
            raise ValueError("Not enough frames for chosen input_len/target_offset")
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len
        inp = np.stack(self.frames[s:e], axis=0)  # C,H,W
        tgt = self.frames[e - 1 + self.target_offset]  # H,W

        # fill NaNs using land mean on that frame (as before)
        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~self.sea_mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        land_vals_tgt = tgt[~self.sea_mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        ts_dt = self.frame_times[e - 1 + self.target_offset]
        # ensure tz-aware
        if ts_dt is None:
            ts_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        ts_epoch = int(ts_dt.timestamp())

        return (
            torch.from_numpy(inp_filled).float(),  # C,H,W
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),  # 1,H,W
            np.int64(ts_epoch),
        )


# ---------------- SwinLSTM cell & model + refinement head (unchanged) ----------------
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class SwinLSTMCell(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, window_size=4, num_heads=4, dropout_p=0.05
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.in_proj = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.attn = LocalWindowAttention(
            dim=hidden_dim, window_size=window_size, num_heads=num_heads
        )
        self.gates_conv = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        assert (
            H % ws == 0 and W % ws == 0
        ), "Height and Width must be divisible by window_size"
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        num_win = (H // ws) * (W // ws)
        x = x.view(B * num_win, ws * ws, C)
        return x, (H // ws, W // ws)

    def window_unpartition(self, x_w, grid_hw, H, W):
        B_numwin, N, C = x_w.shape
        ws = self.window_size
        num_h, num_w = grid_hw
        B = B_numwin // (num_h * num_w)
        x = x_w.view(B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, x, hidden):
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.in_proj(comb)
        pad_h = (self.window_size - (H % self.window_size)) % self.window_size
        pad_w = (self.window_size - (W % self.window_size)) % self.window_size
        if pad_h or pad_w:
            proj_padded = F.pad(proj, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            proj_padded = proj
        Hp, Wp = proj_padded.shape[2], proj_padded.shape[3]
        x_win, grid_hw = self.window_partition(proj_padded)
        attn_out = self.attn(x_win)
        attn_map = self.window_unpartition(attn_out, grid_hw, Hp, Wp)
        if pad_h or pad_w:
            attn_map = attn_map[:, :, :H, :W]
        attn_map = self.gn(attn_map)
        if self.training and self.dropout_p > 0:
            attn_map = F.dropout2d(attn_map, p=self.dropout_p)
        conv = self.gates_conv(attn_map)
        ci, cf, co, cg = torch.chunk(conv, 4, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        o = torch.sigmoid(co)
        g = torch.tanh(cg)
        cnext = f * c + i * g
        hnext = o * torch.tanh(cnext)
        hnext = self.gn(hnext)
        if self.training and self.dropout_p > 0:
            hnext = F.dropout2d(hnext, p=self.dropout_p)
        return hnext, cnext

    def init_hidden(self, b, spatial, device):
        H, W = spatial
        return (
            torch.zeros(b, self.hidden_dim, H, W, device=device),
            torch.zeros(b, self.hidden_dim, H, W, device=device),
        )


class ResidualSwinLSTMWithRefine(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        num_layers=3,
        window_size=4,
        num_heads=4,
        dropout_p=0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                SwinLSTMCell(
                    in_dim,
                    hidden_dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    dropout_p=dropout_p,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device
        hiddens = [l.init_hidden(B, (H, W), device) for l in self.layers]
        last = None
        for t in range(C):
            frame = x[:, t : t + 1, :, :]
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + inp
                hiddens[li] = (hnext, cnext)
                inp = hnext
            last = inp
        out = self.refine(last)
        return out


# ---------------- Helper losses and metrics ----------------
def pretty_cb(cb, fmt="%.2f"):
    cb.ax.tick_params(labelsize=9)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))


def compute_mean_ssim(preds, actuals, sea_mask):
    if ssim_fn is None:
        return None
    # preds/actuals shapes: (N,1,H,W) or (N,H,W)
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)
    if isinstance(actuals, torch.Tensor):
        actuals_np = actuals.detach().cpu().numpy()
    else:
        actuals_np = np.asarray(actuals)
    mask_np = np.asarray(sea_mask)
    # ensure shapes (N,H,W)
    if preds_np.ndim == 4:
        preds_np = preds_np[:, 0]
    if actuals_np.ndim == 4:
        actuals_np = actuals_np[:, 0]
    N, H, W = preds_np.shape
    ssim_vals = []
    for i in range(N):
        a = actuals_np[i].astype(np.float64).copy()
        p = preds_np[i].astype(np.float64).copy()
        m = mask_np.astype(bool)
        try:
            p[m] = a[m]
        except Exception:
            pass
        dr = float(a.max() - a.min())
        if dr == 0:
            dr = 1e-6
        try:
            s = ssim_fn(a, p, data_range=dr)
        except Exception:
            s = np.nan
        ssim_vals.append(s)
    ssim_vals = np.array(ssim_vals, dtype=np.float64)
    if np.all(np.isnan(ssim_vals)):
        return None
    return float(np.nanmean(ssim_vals))


def gradient_loss_torch(pred, target, mask):
    sobel_x = (
        torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
            device=pred.device,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = sobel_x.transpose(2, 3)
    grad_px = F.conv2d(pred, sobel_x, padding=1)
    grad_py = F.conv2d(pred, sobel_y, padding=1)
    grad_tx = F.conv2d(target, sobel_x, padding=1)
    grad_ty = F.conv2d(target, sobel_y, padding=1)
    gdiff = torch.abs(grad_px - grad_tx) + torch.abs(grad_py - grad_ty)
    gmask = mask.expand_as(gdiff)
    sel = gdiff.masked_select(gmask)
    if sel.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return sel.mean()


def fit_columnwise_bias(actuals, preds, sea_mask, smooth_sigma=2.0):
    N, H, W = preds.shape
    slope_cols = np.ones(W, dtype=np.float32)
    intercept_cols = np.zeros(W, dtype=np.float32)
    for j in range(W):
        p_col = preds[:, :, j].reshape(-1)
        a_col = actuals[:, :, j].reshape(-1)
        lat_land_mask = ~sea_mask[:, j]
        if lat_land_mask.sum() == 0:
            slope_cols[j] = 1.0
            intercept_cols[j] = 0.0
            continue
        sel = np.tile(lat_land_mask, N)
        p_sel = p_col[sel]
        a_sel = a_col[sel]
        if p_sel.size >= 5 and np.nanstd(p_sel) > 1e-6:
            s, b = np.polyfit(p_sel, a_sel, 1)
        else:
            s, b = 1.0, 0.0
        slope_cols[j] = float(s)
        intercept_cols[j] = float(b)
    if gaussian_filter is not None:
        slope_cols = gaussian_filter(slope_cols, sigma=smooth_sigma)
        intercept_cols = gaussian_filter(intercept_cols, sigma=smooth_sigma)
    slope_cols = np.clip(slope_cols, 0.75, 1.25)
    slope_map = np.tile(slope_cols[np.newaxis, :], (H, 1))
    intercept_map = np.tile(intercept_cols[np.newaxis, :], (H, 1))
    return slope_map.astype(np.float32), intercept_map.astype(np.float32)


# ---------------- Prepare data: build datasets for train/val from 2020-2023 and test=2024 ----------------
print("Loading dataset and preparing train/val/test splits (chronological)...")
all_years = sorted(
    [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
)
print("Detected year folders:", all_years)

TEST_YEAR = "2024" if "2024" in all_years else all_years[-1]
trainval_years = [y for y in all_years if y != TEST_YEAR]
if not trainval_years:
    raise ValueError("No years available for train/val after excluding test year.")

print(
    f"Using years {trainval_years} for train+val (chronological). Using {TEST_YEAR} as TEST set (kept separate)."
)

dataset_trainval = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=trainval_years,
    fallback_year=int(TEST_YEAR),
)
dataset_test = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=[TEST_YEAR],
    fallback_year=int(TEST_YEAR),
)

n_tv = len(dataset_trainval)
print(f"Train+Val samples (chronological across years {trainval_years}): total={n_tv}")

indices_tv = np.arange(n_tv)
split_idx = int(np.floor((1.0 - val_split) * n_tv))
train_idx = indices_tv[:split_idx]
val_idx = indices_tv[split_idx:]

train_set = Subset(dataset_trainval, train_idx)
val_set = Subset(dataset_trainval, val_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train/Val sizes -> train={len(train_set)}, val={len(val_set)}")
print(f"Test samples from {TEST_YEAR}: {len(dataset_test)} (kept separate)")

sea_mask = dataset_trainval.sea_mask
H, W = dataset_trainval.H, dataset_trainval.W

# ---------------- climatology computed from training targets ----------------
print("Computing per-grid climatology from training targets...")
clim_sum = np.zeros((H, W), dtype=np.float64)
count = 0
for i in train_idx:
    X, y, _ = dataset_trainval[i]
    clim_sum += y.numpy().squeeze(0)
    count += 1
climatology = (clim_sum / max(1, count)).astype(np.float32)
np.save(os.path.join(out_dir, "climatology.npy"), climatology)
print("Saved climatology to:", os.path.join(out_dir, "climatology.npy"))


# ---------------- normalization (anomaly mean/std) ----------------
def compute_norm_from_anomalies(dataset_obj, train_idx_local, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    clim = climatology.astype(np.float64)
    for i in train_idx_local:
        X, y, _ = dataset_obj[i]
        Xn = X.numpy() - clim[np.newaxis, :, :]
        yn = y.numpy().squeeze(0) - clim
        arr = np.concatenate([Xn.ravel(), yn.ravel()]).astype(np.float64)
        s += arr.sum()
        ss += (arr**2).sum()
        cnt += arr.size
    mean = s / cnt
    var = ss / cnt - mean**2
    std = np.sqrt(max(var, 1e-12))
    return float(mean), float(std)


norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    print("Computing normalization (mean/std) from anomaly training set...")
    norm_mean, norm_std = compute_norm_from_anomalies(
        dataset_trainval, train_idx, climatology
    )
    print("norm mean, std (anomalies):", norm_mean, norm_std)

clim_t = (
    torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)
)  # 1x1xHxW
sea_mask_t = torch.from_numpy(sea_mask).to(device)
land_mask_t = (~sea_mask_t).to(device).unsqueeze(0).unsqueeze(0)  # 1x1xHxW

# ---------------- build model / optimizer ----------------
model = ResidualSwinLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    window_size=WINDOW_SIZE,
    num_heads=ATTN_HEADS,
    dropout_p=dropout_p,
).to(device)
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)
criterion_map = nn.SmoothL1Loss(reduction="none")

# --- load checkpoint if exists (quick eval) ---
checkpoint_path = os.path.join(out_dir, "best_model.pth")
SKIP_TRAIN = False
if os.path.exists(checkpoint_path):
    print(
        f"⚡ Found trained model at: {checkpoint_path}  — loading and running a quick eval (no training)."
    )
    state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception as e:
        print("Warning strict load failed:", e)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    try:
        inputs, actuals, _ = next(iter(val_loader))
    except Exception as e:
        print("ERROR: Could not fetch a batch from val_loader:", e)
        raise SystemExit(1)
    inputs, actuals = inputs.to(device), actuals.to(device)
    with torch.no_grad():
        X_anom = inputs - clim_t
        if USE_NORMALIZATION:
            X_anom = (X_anom - norm_mean) / norm_std
        preds = model(X_anom)
        if preds.shape[2:] != actuals.shape[2:]:
            preds = F.interpolate(
                preds, size=actuals.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            preds = preds * norm_std + norm_mean
        preds_abs = preds + clim_t
    print("Loaded checkpoint and finished quick eval. Continuing (TRAIN SKIPPED).")
    SKIP_TRAIN = True

if SKIP_TRAIN:
    epochs = 0
    print("SKIP_TRAIN is True -> setting epochs = 0 to skip training loop.")

# ---------------- Training loop (unpacked ts but ignored in train) ----------------
train_losses = []
val_losses = []

best_val_rmse = 1e9
best_state = None

for epoch in range(1, epochs + 1):
    model.train()
    run = 0.0
    seen = 0
    pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{epochs}")
    for X, y, _ in pbar:
        X = X.to(device)
        y = y.to(device)
        X_anom = X - clim_t
        y_anom = y - clim_t
        if USE_NORMALIZATION:
            X_anom = (X_anom - norm_mean) / norm_std
            y_anom = (y_anom - norm_mean) / norm_std
        opt.zero_grad()
        out = model(X_anom)
        if out.shape != y_anom.shape:
            out = F.interpolate(
                out, size=y_anom.shape[2:], mode="bilinear", align_corners=False
            )
        map_loss = criterion_map(out, y_anom)
        mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
        masked_vals = map_loss.masked_select(mask)
        loss_basic = (
            masked_vals.mean()
            if masked_vals.numel()
            else torch.tensor(0.0, device=device)
        )
        if USE_NORMALIZATION:
            out_abs = out * norm_std + norm_mean
        else:
            out_abs = out
        out_abs = out_abs + clim_t
        y_abs = y
        grad_loss = gradient_loss_torch(
            out_abs, y_abs, land_mask_t.expand(out_abs.shape[0], 1, H, W)
        )
        loss = loss_basic + GRAD_WEIGHT * grad_loss
        # optional SSIM loss
        if SSIM_WEIGHT > 0 and ssim_fn is not None:
            out_np = out_abs.detach().cpu().numpy()
            y_np = y_abs.detach().cpu().numpy()
            ssim_batch = 0.0
            B = out_np.shape[0]
            for bi in range(B):
                a = y_np[bi, 0].astype(np.float64).copy()
                p = out_np[bi, 0].astype(np.float64).copy()
                p[sea_mask] = a[sea_mask]
                dr = float(a.max() - a.min())
                dr = dr if dr != 0 else 1e-6
                try:
                    ssim_val = ssim_fn(a, p, data_range=dr)
                except Exception:
                    ssim_val = 0.0
                ssim_batch += 1.0 - ssim_val
            ssim_batch = ssim_batch / float(B)
            loss = loss + SSIM_WEIGHT * torch.tensor(
                ssim_batch, device=device, dtype=loss.dtype
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += float(loss.item()) * X.size(0)
        seen += X.size(0)
        pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})
    train_loss = run / max(1, seen)
    train_losses.append(train_loss)

    # validation
    model.eval()
    vr = 0.0
    vseen = 0
    all_val_preds = []
    all_val_actuals = []
    with torch.no_grad():
        for Xv, yv, _ in val_loader:
            Xv = Xv.to(device)
            yv = yv.to(device)
            Xv_anom = Xv - clim_t
            yv_anom = yv - clim_t
            if USE_NORMALIZATION:
                Xv_anom = (Xv_anom - norm_mean) / norm_std
                yv_anom = (yv_anom - norm_mean) / norm_std
            outv = model(Xv_anom)
            if outv.shape != yv_anom.shape:
                outv = F.interpolate(
                    outv, size=yv_anom.shape[2:], mode="bilinear", align_corners=False
                )
            map_loss = criterion_map(outv, yv_anom)
            mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
            masked_vals = map_loss.masked_select(mask)
            lv = (
                masked_vals.mean()
                if masked_vals.numel()
                else torch.tensor(0.0, device=device)
            )
            vr += float(lv.item()) * Xv.size(0)
            vseen += Xv.size(0)
            if USE_NORMALIZATION:
                outv = outv * norm_std + norm_mean
            outv_abs = outv + clim_t
            all_val_preds.append(outv_abs.cpu().numpy())
            all_val_actuals.append(yv.cpu().numpy())
    val_loss = vr / max(1, vseen)
    val_losses.append(val_loss)
    sched.step(val_loss)

    preds_arr = (
        np.concatenate(all_val_preds, axis=0)
        if len(all_val_preds)
        else np.empty((0, 1, H, W))
    )
    actuals_arr = (
        np.concatenate(all_val_actuals, axis=0)
        if len(all_val_actuals)
        else np.empty((0, 1, H, W))
    )
    if preds_arr.size:
        mask_flat = (~dataset_trainval.sea_mask).ravel()
        preds_flat = preds_arr.reshape(preds_arr.shape[0], -1)
        actuals_flat = actuals_arr.reshape(actuals_arr.shape[0], -1)
        diffs = actuals_flat[:, mask_flat] - preds_flat[:, mask_flat]
        mse = float(np.nanmean(diffs**2))
        mae = float(np.nanmean(np.abs(diffs)))
        rmse = float(np.sqrt(mse))
        mean_ssim_val = compute_mean_ssim(
            preds_arr, actuals_arr, dataset_trainval.sea_mask
        )
    else:
        mse = mae = rmse = float("nan")
        mean_ssim_val = None

    if epoch % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch:03d} Train={train_loss:.6f} Val={val_loss:.6f} | VAL MSE={mse:.6f} MAE={mae:.6f} RMSE={rmse:.6f} SSIM={mean_ssim_val}"
        )

    if preds_arr.size and rmse < best_val_rmse:
        best_val_rmse = rmse
        best_state = model.state_dict().copy()
        torch.save(best_state, os.path.join(out_dir, "best_model.pth"))

# Save model at end
if best_state is not None:
    try:
        torch.save(best_state, os.path.join(out_dir, "best_model.pth"))
        print("Saved best model to:", os.path.join(out_dir, "best_model.pth"))
    except Exception as e:
        print(
            "Warning: failed to save best_state, attempting to save current model state:",
            e,
        )
        torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
        print("Saved current model state to:", os.path.join(out_dir, "best_model.pth"))
else:
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
    print(
        "No best_state found; saved current model state to:",
        os.path.join(out_dir, "best_model.pth"),
    )

# Save loss curve
loss_fig_path = os.path.join(out_dir, "Train_Val_Loss.png")
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (SmoothL1 on anomalies + grad)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_fig_path, dpi=200)
plt.close()
print("Saved loss curve to:", loss_fig_path)

# ---------------- inference on validation for bias correction & blending ----------------
print("Running inference on validation set for bias correction & blending...")
model.eval()
all_preds = []
all_actuals = []
with torch.no_grad():
    for Xv, yv, _ in val_loader:
        Xv = Xv.to(device)
        yv = yv.to(device)
        Xv_anom = Xv - clim_t
        if USE_NORMALIZATION:
            Xv_anom = (Xv_anom - norm_mean) / norm_std
        outv = model(Xv_anom)
        if outv.shape != yv.shape:
            outv = F.interpolate(
                outv, size=yv.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            outv = outv * norm_std + norm_mean
        outv_abs = outv + clim_t
        all_preds.append(outv_abs.cpu().numpy())
        all_actuals.append(yv.cpu().numpy())

preds = np.concatenate(all_preds, axis=0) if len(all_preds) else np.empty((0, 1, H, W))
actuals = (
    np.concatenate(all_actuals, axis=0) if len(all_actuals) else np.empty((0, 1, H, W))
)

if preds.size:
    mask_flat = (~dataset_trainval.sea_mask).ravel()
    preds_flat = preds.reshape(preds.shape[0], -1)
    actuals_flat = actuals.reshape(actuals.shape[0], -1)
    diffs = actuals_flat[:, mask_flat] - preds_flat[:, mask_flat]
    mse = float(np.nanmean(diffs**2))
    mae = float(np.nanmean(np.abs(diffs)))
    rmse = float(np.sqrt(mse))
    print(
        "Model VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            mse, mae, rmse
        )
    )
    mean_ssim_val = compute_mean_ssim(preds, actuals, dataset_trainval.sea_mask)
    if mean_ssim_val is not None:
        print(f"Model VAL SSIM (land-only): SSIM={mean_ssim_val:.6f}")
else:
    print("No validation predictions available.")
    mean_ssim_val = None

# ---------------- persistence baseline & blending coefficient ----------------
print("Computing persistence baseline and optimal blending coefficient (alpha)...")
pers_preds = []
pers_actuals = []
with torch.no_grad():
    for Xv, yv, _ in val_loader:
        last = Xv[:, -1:, :, :].cpu().numpy()  # Bx1xHxW
        pers_preds.append(last)
        pers_actuals.append(yv.cpu().numpy())
pers_preds = (
    np.concatenate(pers_preds, axis=0) if len(pers_preds) else np.empty((0, 1, H, W))
)
pers_actuals = (
    np.concatenate(pers_actuals, axis=0)
    if len(pers_actuals)
    else np.empty((0, 1, H, W))
)

if pers_preds.size and preds.size:
    Pp = pers_preds.reshape(pers_preds.shape[0], -1)[:, mask_flat]  # persistence
    Pm = preds.reshape(preds.shape[0], -1)[:, mask_flat]  # model
    A = actuals.reshape(actuals.shape[0], -1)[:, mask_flat]
    numer = np.nansum((A - Pp) * (Pm - Pp))
    denom = np.nansum((Pm - Pp) ** 2)
    if denom <= 1e-12:
        alpha = 1.0
    else:
        alpha = float(numer / denom)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    print(f"Blending alpha (model weight) learned on val: {alpha:.4f}")
    preds_blend = alpha * preds[:, 0] + (1.0 - alpha) * pers_preds[:, 0]
    preds_blend = preds_blend[:, None, :, :]  # N,1,H,W
else:
    alpha = 1.0
    preds_blend = preds.copy()

# ---------------- Fit column-wise bias correction on blended preds ----------------
print("Fitting column-wise (longitude) bias correction on blended preds...")
if preds_blend.size:
    Pblend = preds_blend[:, 0]
    A = actuals[:, 0]
    slope_map, intercept_map = fit_columnwise_bias(
        A, Pblend, dataset_trainval.sea_mask, smooth_sigma=2.0
    )
    np.save(os.path.join(out_dir, "bias_slope_map.npy"), slope_map)
    np.save(os.path.join(out_dir, "bias_intercept_map.npy"), intercept_map)
    print("Saved slope/intercept maps to:", out_dir)
    preds_corr = preds_blend.copy()
    preds_corr[:, :] = preds_blend * slope_map[None, :, :] + intercept_map[None, :, :]
    pc_flat = preds_corr.reshape(preds_corr.shape[0], -1)[:, mask_flat]
    a_flat_mat = actuals.reshape(actuals.shape[0], -1)[:, mask_flat]
    dif_corr = a_flat_mat - pc_flat
    mse_corr = float(np.nanmean(dif_corr**2))
    mae_corr = float(np.nanmean(np.abs(dif_corr)))
    rmse_corr = float(np.sqrt(mse_corr))
    print(
        "Bias-corrected (blended) VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            mse_corr, mae_corr, rmse_corr
        )
    )
    mean_ssim_corr = compute_mean_ssim(preds_corr, actuals, dataset_trainval.sea_mask)
    if mean_ssim_corr is not None:
        print(
            f"Bias-corrected (blended) VAL SSIM (land-only): SSIM={mean_ssim_corr:.6f}"
        )
else:
    slope_map = np.ones((H, W), dtype=np.float32)
    intercept_map = np.zeros((H, W), dtype=np.float32)
    preds_corr = preds_blend
    mean_ssim_corr = None
    print("No preds to fit bias correction.")

# ---------------- Test-time inference -> per-sample metrics + daily aggregation + plots ----------------
print("Running inference on TEST set and computing time-series metrics for 2024...")
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

test_mae = []
test_rmse = []
test_ssim = []
test_timestamps = []

model.eval()
with torch.no_grad():
    for Xb, yb, tsb in test_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        Xb_anom = Xb - clim_t
        if USE_NORMALIZATION:
            Xb_anom = (Xb_anom - norm_mean) / norm_std
        outb = model(Xb_anom)
        if outb.shape != yb.shape:
            outb = F.interpolate(
                outb, size=yb.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            outb = outb * norm_std + norm_mean
        outb_abs = outb + clim_t
        pred_np = outb_abs.cpu().numpy()[0, 0]
        actual_np = yb.cpu().numpy()[0, 0]
        mask = dataset_trainval.sea_mask
        land_pred = pred_np[~mask]
        land_actual = actual_np[~mask]
        if land_actual.size:
            mae_v = float(np.nanmean(np.abs(land_actual - land_pred)))
            mse_v = float(np.nanmean((land_actual - land_pred) ** 2))
            rmse_v = float(np.sqrt(mse_v))
        else:
            mae_v = float("nan")
            rmse_v = float("nan")
        if ssim_fn is not None:
            a = actual_np.astype(np.float64).copy()
            p = pred_np.astype(np.float64).copy()
            try:
                p[mask] = a[mask]
            except Exception:
                pass
            dr = float(a.max() - a.min())
            dr = dr if dr != 0 else 1e-6
            try:
                s_v = float(ssim_fn(a, p, data_range=dr))
            except Exception:
                s_v = float("nan")
        else:
            s_v = float("nan")
        test_mae.append(mae_v)
        test_rmse.append(rmse_v)
        test_ssim.append(s_v)
        # tsb is np.int64 epoch
        try:
            epoch = int(tsb.item())
            ts_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        except Exception:
            ts_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        test_timestamps.append(ts_dt)

# Save per-sample CSV
csv_path = os.path.join(out_dir, f"test_metrics_{TEST_YEAR}.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "timestamp_utc", "MAE", "RMSE", "SSIM"])
    for i, (t, mae_v, rmse_v, s_v) in enumerate(
        zip(test_timestamps, test_mae, test_rmse, test_ssim)
    ):
        w.writerow([i, t.isoformat(), mae_v, rmse_v, s_v])
print("Saved per-sample test CSV:", csv_path)

# Aggregate to daily mean
df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
# restrict to TEST_YEAR if needed
df = df[df["timestamp_utc"].dt.year == int(TEST_YEAR)].copy()
df["date"] = df["timestamp_utc"].dt.floor("D")
daily = df.groupby("date")[["MAE", "RMSE", "SSIM"]].mean().reset_index()

# ensure full-year coverage Jan 1 -> Dec 31
full_idx = pd.date_range(
    start=f"{TEST_YEAR}-01-01", end=f"{TEST_YEAR}-12-31", freq="D", tz="UTC"
)
daily_full = pd.DataFrame({"date": full_idx})
daily = (
    pd.merge(daily_full, daily, on="date", how="left")
    .sort_values("date")
    .reset_index(drop=True)
)

# plotting: SSIM (Figure4), MAE (Figure5), RMSE (Figure6)
dates = pd.to_datetime(daily["date"])
try:
    dates_naive = dates.dt.tz_convert(None)
except Exception:
    try:
        dates_naive = dates.dt.tz_localize(None)
    except Exception:
        dates_naive = dates

# fallback columns if missing
if "SSIM" not in daily.columns:
    daily["SSIM"] = np.nan

month_locator = mdates.MonthLocator()
month_formatter = mdates.DateFormatter("%b")


def save_daily_plot(dates, vals, metric_name, out_name, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot_date(mdates.date2num(dates), vals, "-", lw=1.0, color=color)
    ax.xaxis.set_major_locator(month_locator)
    ax.xaxis.set_major_formatter(month_formatter)
    ax.set_xlim([mdates.date2num(dates.min()), mdates.date2num(dates.max())])
    ax.set_xlabel("Months →", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"Figure: Time series of {metric_name} for test data ({TEST_YEAR}) of SwinLSTM",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_name)


fig_ssim = os.path.join(out_dir, f"Figure4_SSIM_{TEST_YEAR}.png")
fig_mae = os.path.join(out_dir, f"Figure5_MAE_{TEST_YEAR}.png")
fig_rmse = os.path.join(out_dir, f"Figure6_RMSE_{TEST_YEAR}.png")

save_daily_plot(dates_naive, daily["SSIM"].values, "SSIM", fig_ssim, color="#de2d26")
save_daily_plot(
    dates_naive, daily["MAE"].values, "MAE (units)", fig_mae, color="#7fc97f"
)
save_daily_plot(
    dates_naive, daily["RMSE"].values, "RMSE (units)", fig_rmse, color="#2c7fb8"
)

print("Done. Time-series figures + CSV saved to:", out_dir)

# ---------------- (rest of script) sample visualization unchanged (kept for compatibility) ----------------
# (I keep the sample-building + plotting code below unchanged from your original script
#  so you still get the 'Actual/Predicted/Error' map for the requested sample.)

# Build requested sample in downsampled timeline
ORIG_INPUT_INDICES = [0, 3, 6, 9, 12, 15, 18, 21]
ORIG_TARGET_INDEX = 33
ds_input_idxs = [i // SAMPLE_STEP for i in ORIG_INPUT_INDICES]
ds_target_idx = ORIG_TARGET_INDEX // SAMPLE_STEP
assert ds_input_idxs == list(
    range(input_len)
), f"Expected contiguous downsampled inputs 0..{input_len-1}, got {ds_input_idxs}"

frames_ds = dataset_trainval.frames
T_ds = len(frames_ds)
if ds_target_idx < 0 or ds_target_idx >= T_ds:
    raise IndexError(
        f"Downsampled target index {ds_target_idx} out of range (0..{T_ds-1})"
    )

input_arr = np.stack([frames_ds[i] for i in ds_input_idxs], axis=0).astype(np.float32)
target_arr = frames_ds[ds_target_idx].astype(np.float32)

# fill NaNs
inp_filled = np.empty_like(input_arr, dtype=np.float32)
for i in range(input_arr.shape[0]):
    frame = input_arr[i]
    land_vals = frame[~dataset_trainval.sea_mask]
    fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
    inp_filled[i] = np.where(np.isnan(frame), fill, frame)
land_vals_tgt = target_arr[~dataset_trainval.sea_mask]
fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
tgt_filled = np.where(np.isnan(target_arr), fill_t, target_arr).astype(np.float32)

# inference on single sample
model.eval()
with torch.no_grad():
    X_sample = torch.from_numpy(inp_filled).unsqueeze(0).float().to(device)
    y_sample = torch.from_numpy(tgt_filled).unsqueeze(0).unsqueeze(0).float().to(device)
    X_sample_anom = X_sample - clim_t
    if USE_NORMALIZATION:
        X_sample_anom = (X_sample_anom - norm_mean) / norm_std
    out_sample = model(X_sample_anom)
    if out_sample.shape[2:] != tgt_filled.shape:
        out_sample = F.interpolate(
            out_sample, size=tgt_filled.shape, mode="bilinear", align_corners=False
        )
    if USE_NORMALIZATION:
        out_sample = out_sample * norm_std + norm_mean
    out_sample_abs = out_sample + clim_t
    pred_sample = out_sample_abs.cpu().numpy()[0, 0]

pers_sample = inp_filled[-1]
pred_sample_blend = alpha * pred_sample + (1.0 - alpha) * pers_sample
pred_sample_bc = pred_sample_blend * slope_map + intercept_map

actual_sample = tgt_filled
mask = dataset_trainval.sea_mask
actual_masked = np.ma.masked_where(mask, actual_sample)
pred_masked = np.ma.masked_where(mask, pred_sample_bc)
error_masked = np.ma.masked_where(mask, actual_sample - pred_sample_bc)

land_actual = actual_sample[~mask]
land_pred_bc = pred_sample_bc[~mask]
if land_actual.size:
    difc = land_actual - land_pred_bc
    mse_sample_bc = float(np.nanmean(difc**2))
    mae_sample_bc = float(np.nanmean(np.abs(difc)))
    rmse_sample_bc = float(np.sqrt(mse_sample_bc))
else:
    mse_sample_bc = mae_sample_bc = rmse_sample_bc = float("nan")

if ssim_fn is not None:
    try:
        a = actual_sample.astype(np.float64).copy()
        pbc = pred_sample_bc.astype(np.float64).copy()
        pbc[mask] = a[mask]
        dr = float(a.max() - a.min())
        dr = dr if dr != 0 else 1e-6
        s_sample_bc = float(ssim_fn(a, pbc, data_range=dr))
    except Exception:
        s_sample_bc = float("nan")
else:
    s_sample_bc = float("nan")

print(
    f"Sample metrics (downsampled target idx {ds_target_idx}): AFTER BC+BLEND -> MSE={mse_sample_bc:.6f}, MAE={mae_sample_bc:.6f}, RMSE={rmse_sample_bc:.6f}, SSIM={s_sample_bc}"
)


# plotting the sample maps (unchanged)
def find_first_nc(folder):
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".nc"):
            return os.path.join(folder, fn)
    raise FileNotFoundError


nc0 = find_first_nc(folder_path)
ds0 = Dataset(nc0)
lats = ds0.variables["latitude"][:].astype(float)
lons = ds0.variables["longitude"][:].astype(float)
ds0.close()

origin = "lower"
if lats[0] > lats[-1]:
    origin = "upper"

lon_min = float(lons.min())
lon_max = float(lons.max())
lat_min = float(lats.min())
lat_max = float(lats.max())
extent = [lon_min, lon_max, lat_min, lat_max]

num_xticks = min(6, lons.size)
num_yticks = min(6, lats.size)
xtick_idxs = np.linspace(0, lons.size - 1, num_xticks).astype(int)
ytick_idxs = np.linspace(0, lats.size - 1, num_yticks).astype(int)
xticks = lons[xtick_idxs]
yticks = lats[ytick_idxs]

cmap_temp = plt.get_cmap("gray")
try:
    cmap_temp = cmap_temp.copy()
except Exception:
    cmap_temp = mpl.colors.ListedColormap(cmap_temp(np.linspace(0, 1, cmap_temp.N)))
cmap_temp.set_bad("white")

cmap_err = plt.get_cmap("gray")
try:
    cmap_err = cmap_err.copy()
except Exception:
    cmap_err = mpl.colors.ListedColormap(cmap_err(np.linspace(0, 1, cmap_err.N)))
cmap_err.set_bad("white")

combined = np.concatenate(
    [actual_masked.filled(np.nan).ravel(), pred_masked.filled(np.nan).ravel()]
)
combined = combined[~np.isnan(combined)]
if combined.size == 0:
    raise RuntimeError("No valid land pixels to plot.")
vmin = float(np.nanmin(combined))
vmax = float(np.nanmax(combined))

err_data = (actual_masked - pred_masked).filled(np.nan)
err_vals = err_data[~np.isnan(err_data)]
if err_vals.size == 0:
    vmin_err, vmax_err = -0.5, 0.5
else:
    vmin_err = float(np.nanmin(err_vals))
    vmax_err = float(np.nanmax(err_vals))
    if np.isclose(vmin_err, vmax_err):
        pad = max(1e-4, abs(vmin_err) * 0.001)
        vmin_err -= pad
        vmax_err += pad

err_ticks = [vmin_err, 0.0, vmax_err]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "\n\nInput: 1st April, 2024 | 12am - 9pm       |       Output: 2nd April, 2024 | 9am\n\n",
    fontsize=14,
    fontweight="bold",
    y=0.96,
)

im0 = axes[0].imshow(
    actual_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[0].set_title("Actual", fontsize=14)
axes[0].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[0].set_xticks(xticks)
axes[0].set_yticks(yticks)

im1 = axes[1].imshow(
    pred_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[1].set_title("Predicted", fontsize=14)
axes[1].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[1].set_xticks(xticks)
axes[1].set_yticks(yticks)

im2 = axes[2].imshow(
    error_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_err,
    vmin=vmin_err,
    vmax=vmax_err,
    interpolation="nearest",
)
axes[2].set_title("Error = Actual - Predicted", fontsize=14)
axes[2].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[2].set_xticks(xticks)
axes[2].set_yticks(yticks)

for ax, im in zip(axes, [im0, im1, im2]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    if im is im2:
        cbar.set_label("Error (units)", fontsize=10)
        cbar.set_ticks(err_ticks)
        cbar.ax.set_yticklabels([f"{v:.003f}" for v in err_ticks])
        pretty_cb(cbar, fmt="%.3f")
    else:
        cbar.set_label("2m Temperature (units)", fontsize=10)
        pretty_cb(cbar, fmt="%.2f")

metrics_text = f"MSE: {mse_sample_bc:.6f}   MAE: {mae_sample_bc:.6f}   RMSE: {rmse_sample_bc:.6f}   SSIM: {s_sample_bc:.6f}"
plt.tight_layout(rect=[0, 0.15, 1, 0.94])

fig.text(
    0.5,
    0.02,
    metrics_text,
    fontsize=14,
    va="center",
    ha="center",
    fontname="Times New Roman",
    bbox=dict(
        facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.3"
    ),
)

save_path = os.path.join(out_dir, "Actual_Predicted_Error.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved plot to:", save_path)
print("Saved slope/intercept maps and climatology in:", out_dir)
print("Done.")
