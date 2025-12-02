#!/usr/bin/env python3
"""
swinlstm_metrics_full.py

Complete script: dataset -> model -> train (or load checkpoint) -> test -> daily metrics -> 3 time-series plots (SSIM, MAE, RMSE)
Designed for the folder layout:
  top_folder/
      2020/*.nc
      2021/*.nc
      ...
      2024/*.nc   <- test year (kept separate)

Adapted to be robust if nc time metadata is missing.
"""
import os
import sys
import math
import time
import copy
import numpy as np
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Any, Optional
import datetime as dt
import warnings

# Optional imports
try:
    from skimage.metrics import structural_similarity as ssim_fn

    SSIM_AVAILABLE = True
except Exception:
    ssim_fn = None
    SSIM_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

# ---------------- CONFIG ----------------
# Path to top-level dataset folder (contains year subfolders)
FOLDER_PATH = "../../Dataset"  # change if needed
OUT_DIR = "SwinLSTM_Sliding_v6"
os.makedirs(OUT_DIR, exist_ok=True)

# Temporal downsample step (must match the model training downsample used earlier)
SAMPLE_STEP = 3  # every 3 hours -> keep 1 sample of 3
INPUT_LEN = 8
TARGET_OFFSET = 4

# Train/Validation/Test year split
TEST_YEAR = "2024"

# Training hyperparams
BATCH_SIZE = 6
LR = 8e-5
EPOCHS = 80
VAL_SPLIT = 0.20  # within 2020-2023 chronological
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True  # climatology-based zscore on anomalies
PRINT_EVERY = 1
EARLY_STOPPING_PATIENCE = 12

# Model hyperparams (you can tune)
HIDDEN_DIM = 160
NUM_LAYERS = 2
WINDOW_SIZE = 4
ATTN_HEADS = 4
DROPOUT_P = 0.08
WEIGHT_DECAY = 1e-6
GRAD_WEIGHT = 0.7  # gradient loss weight
SSIM_WEIGHT = 0.0  # keep 0 unless you want to use SSIM in training (slow)

mpl.rcParams["font.family"] = "Times New Roman"
# ---------------- END CONFIG ----------------

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Device:", DEVICE)
print("SSIM available:", SSIM_AVAILABLE)


# ---------------- Dataset ----------------
class SlidingMaskDatasetWithTime(TorchDataset):
    """
    Slides windows across all .nc files in given year directories (chronological order).
    Tries to extract per-frame datetimes from the NetCDF 'time' variable if present.
    Returns (input_tensor: CxHxW, target_tensor: 1xHxW, timestamp: datetime or None)
    """

    def __init__(
        self,
        folder_path,
        input_len=8,
        target_offset=4,
        sample_step=1,
        include_years=None,
    ):
        import os
        from netCDF4 import Dataset as ncDataset

        self.frames = []  # list of 2D arrays (H,W)
        self.timestamps = []  # list of datetimes or None (same length as frames)
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

        # collect year directories
        entries = sorted(os.listdir(folder_path))
        years = [e for e in entries if os.path.isdir(os.path.join(folder_path, e))]
        if include_years:
            years = [y for y in years if str(y) in set(map(str, include_years))]
        # numeric sort if possible
        try:
            years = sorted(years, key=lambda x: int(x))
        except Exception:
            years = sorted(years)

        if not years:
            raise ValueError(f"No year directories found in {folder_path}")

        # helper to convert netcdf time arrays to python datetimes when available
        def times_from_ncvar(var):
            # handle common CF time formats (units like "hours since 1970-01-01 00:00:00")
            try:
                units = getattr(var, "units", None)
                calendar = getattr(var, "calendar", "standard")
                vals = np.array(var[:])
                # netCDF4 has num2date if available; try to use it
                from netCDF4 import num2date

                dates = num2date(vals, units, calendar=calendar)
                # convert numpy.datetime64 or datetime objects -> python datetimes (naive UTC)
                out = []
                for d in dates:
                    if isinstance(d, np.datetime64):
                        out.append(
                            np.datetime64(d)
                            .astype("datetime64[ms]")
                            .astype(dt.datetime)
                        )
                    elif isinstance(d, dt.datetime):
                        out.append(d)
                    else:
                        try:
                            out.append(
                                dt.datetime(
                                    d.year, d.month, d.day, d.hour, d.minute, d.second
                                )
                            )
                        except Exception:
                            out.append(None)
                return out
            except Exception:
                return None

        for year in years:
            ydir = os.path.join(folder_path, year)
            nc_files = sorted([f for f in os.listdir(ydir) if f.endswith(".nc")])
            # order months by filename if possible (Jan..Dec)
            month_order = {
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

            def key_fn(fn):
                base = os.path.splitext(fn)[0].lower()
                return month_order.get(base, 999), fn

            nc_files = sorted(nc_files, key=key_fn)

            for fn in nc_files:
                p = os.path.join(ydir, fn)
                try:
                    ds = ncDataset(p)
                except Exception as e:
                    print(f"Warning: failed to read {p}: {e}")
                    continue

                # find t2m variable
                if "t2m" not in ds.variables:
                    ds.close()
                    continue
                var = ds.variables["t2m"]
                arr = np.array(var[:])  # typically (T, lat, lon) or (lat, lon)
                # mask fill values -> NaN
                if hasattr(var, "_FillValue"):
                    arr = np.where(arr == var._FillValue, np.nan, arr)
                if hasattr(var, "missing_value"):
                    arr = np.where(arr == var.missing_value, np.nan, arr)

                # attempt to read time variable for this file
                times_for_file = None
                # many datasets name it 'time' or 'valid_time' etc.
                possible_time_vars = [
                    v for v in ds.variables.keys() if "time" in v.lower()
                ]
                for tname in possible_time_vars:
                    try:
                        times_for_file = times_from_ncvar(ds.variables[tname])
                        if times_for_file is not None:
                            break
                    except Exception:
                        times_for_file = None

                # if arr is 3D => iterate over time axis
                if arr.ndim == 3:
                    T = arr.shape[0]
                    for t in range(T):
                        self.frames.append(arr[t].astype(np.float32))
                        if times_for_file is not None and len(times_for_file) == T:
                            self.timestamps.append(times_for_file[t])
                        else:
                            # try to read "time" dimension from variable if per-timestep metadata exists
                            self.timestamps.append(None)
                elif arr.ndim == 2:
                    self.frames.append(arr.astype(np.float32))
                    self.timestamps.append(
                        times_for_file[0]
                        if (times_for_file and len(times_for_file) >= 1)
                        else None
                    )
                ds.close()

        if len(self.frames) == 0:
            raise ValueError("No frames loaded in dataset.")

        # downsample frames and timestamps
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]
            self.timestamps = self.timestamps[:: self.sample_step]

        # ensure consistent shapes
        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent spatial shapes found: {shapes}")
        self.H, self.W = self.frames[0].shape

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)  # True where always NaN (sea)

        # compute sliding start indices
        starts = []
        for s in range(len(self.frames) - input_len - target_offset + 1):
            starts.append(s)
        if not starts:
            raise ValueError("Not enough frames for chosen input_len/target_offset")
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len
        inp = np.stack(self.frames[s:e], axis=0)  # C,H,W
        tgt = self.frames[e - 1 + self.target_offset]  # H,W

        # fill NaNs in each frame using land-mean for that frame
        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~self.sea_mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        land_vals_t = tgt[~self.sea_mask]
        fill_t = float(np.nanmean(land_vals_t)) if land_vals_t.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        # timestamp of the target frame (if any)
        ts = (
            self.timestamps[e - 1 + self.target_offset]
            if len(self.timestamps) > (e - 1 + self.target_offset)
            else None
        )

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
            ts,
        )


# ---------------- SwinLSTM components ----------------
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: B, N, C
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # B,N,heads,hd
        q = q.permute(0, 2, 1, 3)  # B,heads,N,hd
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # B,heads,N,hd
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class SwinLSTMCell(nn.Module):
    """
    A simplified SwinLSTM cell that:
     - projects input+hidden to feature map
     - partitions into non-overlapping windows and applies LocalWindowAttention
     - aggregates and computes Conv-like LSTM gates
    """

    def __init__(self, in_ch, hidden_ch, window_size=4, num_heads=4, dropout_p=0.05):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.input_proj = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size=1)
        self.attn = LocalWindowAttention(dim=hidden_ch, num_heads=num_heads)
        self.gates = nn.Conv2d(hidden_ch, 4 * hidden_ch, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, hidden_ch), num_channels=hidden_ch)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        # pad so divisible
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        num_win = (Hp // ws) * (Wp // ws)
        x = x.view(B * num_win, ws * ws, C)
        return x, (Hp, Wp, pad_h, pad_w, Hp // ws, Wp // ws)

    def window_unpartition(self, xw, grid_info):
        Hp, Wp, pad_h, pad_w, nh, nw = (
            grid_info[0],
            grid_info[1],
            grid_info[2],
            grid_info[3],
            grid_info[4],
            grid_info[5],
        )
        B_numwin, N, C = xw.shape
        ws = self.window_size
        B = B_numwin // (nh * nw)
        x = xw.view(B, nh, nw, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, : Hp - pad_h, : Wp - pad_w]
        return x

    def forward(self, x, hidden):
        # x: B x in_ch x H x W
        # hidden: (h,c)
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.input_proj(comb)  # B, hidden, H, W

        x_win, grid_info = self.window_partition(proj)  # (B*num_win, N, C)
        attn_out = self.attn(x_win)  # same shape
        attn_map = self.window_unpartition(
            attn_out, grid_info
        )  # B,hidden,Hp',Wp' -> trimmed inside

        attn_map = self.norm(attn_map)
        if self.training and self.dropout_p > 0:
            attn_map = F.dropout2d(attn_map, p=self.dropout_p)

        conv_g = self.gates(attn_map)
        ci, cf, co, cg = torch.chunk(conv_g, 4, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        o = torch.sigmoid(co)
        g = torch.tanh(cg)
        cnext = f * c + i * g
        hnext = o * torch.tanh(cnext)
        hnext = self.norm(hnext)
        if self.training and self.dropout_p > 0:
            hnext = F.dropout2d(hnext, p=self.dropout_p)
        return hnext, cnext

    def init_hidden(self, b, spatial, device):
        H, W = spatial
        return (
            torch.zeros(b, self.hidden_ch, H, W, device=device),
            torch.zeros(b, self.hidden_ch, H, W, device=device),
        )


class ResidualSwinLSTM(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        num_layers=2,
        window_size=4,
        num_heads=4,
        dropout_p=0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else hidden_dim
            layers.append(
                SwinLSTMCell(
                    in_ch,
                    hidden_dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    dropout_p=dropout_p,
                )
            )
        self.layers = nn.ModuleList(layers)
        # refinement decoder: small conv residual stack -> 1 channel
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: B x C x H x W  (C = time length)
        B, C, H, W = x.shape
        device = x.device
        hiddens = [layer.init_hidden(B, (H, W), device) for layer in self.layers]
        last = None
        for t in range(C):
            frame = x[:, t : t + 1, :, :]  # B x 1 x H x W
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + inp  # residual across layers
                hiddens[li] = (hnext, cnext)
                inp = hnext
            last = inp
        out = self.refine(last)
        return out


# ---------------- losses / metrics ----------------
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


def compute_mean_ssim(preds, actuals, sea_mask):
    if ssim_fn is None:
        return None
    # preds, actuals: numpy arrays or tensors with shape (N,1,H,W) or (N,H,W)
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)
    if isinstance(actuals, torch.Tensor):
        actuals_np = actuals.detach().cpu().numpy()
    else:
        actuals_np = np.asarray(actuals)
    mask_np = np.asarray(sea_mask)
    # format to N,H,W
    if preds_np.ndim == 4:
        preds_np = preds_np[:, 0, :, :]
    if actuals_np.ndim == 4:
        actuals_np = actuals_np[:, 0, :, :]
    N = preds_np.shape[0]
    out_vals = []
    for i in range(N):
        a = actuals_np[i].astype(np.float64).copy()
        p = preds_np[i].astype(np.float64).copy()
        # force sea pixels equal to actuals so they don't affect SSIM
        try:
            p[mask_np] = a[mask_np]
        except Exception:
            pass
        dr = float(a.max() - a.min())
        if dr == 0.0:
            dr = 1e-6
        try:
            s = ssim_fn(a, p, data_range=dr)
        except Exception:
            s = np.nan
        out_vals.append(s)
    out_vals = np.array(out_vals, dtype=np.float64)
    if np.all(np.isnan(out_vals)):
        return None
    return float(np.nanmean(out_vals))


# ---------------- collate_fn to allow None timestamps ----------------
def collate_keep_ts(batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]):
    inputs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    times = [b[2] for b in batch]
    inputs_tensor = torch.stack(inputs, dim=0)
    targets_tensor = torch.stack(targets, dim=0)
    return inputs_tensor, targets_tensor, times


# ---------------- helper: month ticks for plotting ----------------
def month_ticks_for_dates(dates):
    # dates: list of python datetimes (for every sample) - choose monthly tick locations (first of month)
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ticks = []
    labels = []
    if len(dates) == 0:
        return [], []
    start = dates[0].replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # produce index of first occurrence of each month in dates
    for m in months:
        found = None
        for i, d in enumerate(dates):
            if d is None:
                continue
            try:
                if d.month == m:
                    found = i
                    break
            except Exception:
                pass
        if found is not None:
            ticks.append(found)
            labels.append(dt.date(1900, m, 1).strftime("%b"))
    return ticks, labels


# ---------------- main flow ----------------
def main():
    print("Preparing datasets and splits...")
    years = sorted(
        [
            d
            for d in os.listdir(FOLDER_PATH)
            if os.path.isdir(os.path.join(FOLDER_PATH, d))
        ]
    )
    print("Found years:", years)
    if TEST_YEAR not in years:
        raise ValueError(f"Test year {TEST_YEAR} not found in {FOLDER_PATH}")
    trainval_years = [y for y in years if y != TEST_YEAR]
    print("Train/Val years (chronological):", trainval_years, "; Test year:", TEST_YEAR)

    # build datasets
    ds_tv = SlidingMaskDatasetWithTime(
        FOLDER_PATH,
        input_len=INPUT_LEN,
        target_offset=TARGET_OFFSET,
        sample_step=SAMPLE_STEP,
        include_years=trainval_years,
    )
    ds_test = SlidingMaskDatasetWithTime(
        FOLDER_PATH,
        input_len=INPUT_LEN,
        target_offset=TARGET_OFFSET,
        sample_step=SAMPLE_STEP,
        include_years=[TEST_YEAR],
    )

    N_tv = len(ds_tv)
    print(f"Train+Val samples: {N_tv}  Test samples: {len(ds_test)}")

    # chronological split 80/20
    indices = np.arange(N_tv)
    split_idx = int(np.floor((1.0 - VAL_SPLIT) * N_tv))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_set = Subset(ds_tv, train_idx)
    val_set = Subset(ds_tv, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keep_ts,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_keep_ts,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_keep_ts,
    )

    print(f"Train {len(train_set)}, Val {len(val_set)}, Test {len(ds_test)}")

    H, W = ds_tv.H, ds_tv.W
    sea_mask = ds_tv.sea_mask

    # compute climatology from training targets (per-grid mean)
    print("Computing climatology from training targets...")
    clim_sum = np.zeros((H, W), dtype=np.float64)
    cnt = 0
    for i in train_idx:
        _, y, _ = ds_tv[i]
        clim_sum += y.numpy().squeeze(0)
        cnt += 1
    climatology = (clim_sum / max(1, cnt)).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "climatology.npy"), climatology)
    print("Saved climatology.")

    # compute normalization (mean/std) from anomaly training set
    def compute_norm():
        s = 0.0
        ss = 0.0
        cnt = 0
        for i in train_idx:
            X, y, _ = ds_tv[i]
            Xn = X.numpy() - climatology[np.newaxis, :, :]
            yn = y.numpy().squeeze(0) - climatology
            arr = np.concatenate([Xn.ravel(), yn.ravel()]).astype(np.float64)
            s += arr.sum()
            ss += (arr**2).sum()
            cnt += arr.size
        mean = s / cnt
        var = ss / cnt - mean * mean
        std = math.sqrt(max(var, 1e-12))
        return float(mean), float(std)

    norm_mean, norm_std = 0.0, 1.0
    if USE_NORMALIZATION:
        print("Computing anomaly normalization from training set...")
        norm_mean, norm_std = compute_norm()
        print("norm mean, std:", norm_mean, norm_std)

    # prepare tensors for climatology broadcasting
    clim_t = (
        torch.from_numpy(climatology).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    )  # 1x1xHxW
    land_mask_t = (
        (~torch.from_numpy(sea_mask)).to(DEVICE).unsqueeze(0).unsqueeze(0)
    )  # 1x1xHxW

    # build model and optimizer
    model = ResidualSwinLSTM(
        in_channels=INPUT_LEN,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        window_size=WINDOW_SIZE,
        num_heads=ATTN_HEADS,
        dropout_p=DROPOUT_P,
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=6
    )
    criterion = nn.SmoothL1Loss(reduction="none")

    checkpoint_path = os.path.join(OUT_DIR, "best_model.pth")
    start_epoch = 1
    best_val_rmse = 1e9
    best_state = None
    early_counter = 0

    # If checkpoint exists, load and skip training
    if os.path.exists(checkpoint_path):
        print("Found checkpoint at:", checkpoint_path, "- loading.")
        state = torch.load(checkpoint_path, map_location=DEVICE)
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.to(DEVICE)
        print("Loaded checkpoint. Skipping training and proceeding to test.")
    else:
        print("No checkpoint found. Starting training...")

        # training loop
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
            for Xb, yb, _ in pbar:
                Xb = Xb.to(DEVICE)  # BxCxHxW
                yb = yb.to(DEVICE)  # Bx1xHxW

                # anomalies
                Xan = Xb - clim_t
                yan = yb - clim_t
                if USE_NORMALIZATION:
                    Xan = (Xan - norm_mean) / norm_std
                    yan = (yan - norm_mean) / norm_std

                opt.zero_grad()
                out = model(Xan)  # predicted normalized anomalies
                if out.shape != yan.shape:
                    out = F.interpolate(
                        out, size=yan.shape[2:], mode="bilinear", align_corners=False
                    )

                map_loss = criterion(out, yan)  # Bx1xHxW
                mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
                vals = map_loss.masked_select(mask)
                if vals.numel() == 0:
                    loss_basic = torch.tensor(0.0, device=DEVICE)
                else:
                    loss_basic = vals.mean()

                # gradient loss on absolute scale
                if USE_NORMALIZATION:
                    out_abs = out * norm_std + norm_mean
                else:
                    out_abs = out
                out_abs = out_abs + clim_t
                y_abs = yb
                grad_loss = gradient_loss_torch(
                    out_abs, y_abs, land_mask_t.expand(out_abs.shape[0], 1, H, W)
                )

                loss = loss_basic + GRAD_WEIGHT * grad_loss

                # optional SSIM regularization (slow)
                if SSIM_WEIGHT > 0 and SSIM_AVAILABLE:
                    batch_ssim = 0.0
                    out_np = out_abs.detach().cpu().numpy()
                    y_np = y_abs.detach().cpu().numpy()
                    B = out_np.shape[0]
                    for bi in range(B):
                        a = y_np[bi, 0].astype(np.float64).copy()
                        p = out_np[bi, 0].astype(np.float64).copy()
                        p[sea_mask] = a[sea_mask]
                        dr = float(a.max() - a.min())
                        dr = dr if dr != 0 else 1e-6
                        try:
                            s = ssim_fn(a, p, data_range=dr)
                        except Exception:
                            s = 0.0
                        batch_ssim += 1.0 - s
                    batch_ssim /= float(B)
                    loss = loss + SSIM_WEIGHT * torch.tensor(
                        batch_ssim, device=DEVICE, dtype=loss.dtype
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running_loss += float(loss.item()) * Xb.size(0)
                seen += Xb.size(0)
                pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

            train_loss = running_loss / max(1, seen)

            # validation
            model.eval()
            vr = 0.0
            vseen = 0
            all_preds = []
            all_truths = []
            with torch.no_grad():
                for Xv, yv, _ in val_loader:
                    Xv = Xv.to(DEVICE)
                    yv = yv.to(DEVICE)
                    Xv_an = Xv - clim_t
                    yv_an = yv - clim_t
                    if USE_NORMALIZATION:
                        Xv_an = (Xv_an - norm_mean) / norm_std
                        yv_an = (yv_an - norm_mean) / norm_std
                    outv = model(Xv_an)
                    if outv.shape != yv_an.shape:
                        outv = F.interpolate(
                            outv,
                            size=yv_an.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    map_loss = criterion(outv, yv_an)
                    mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
                    vals = map_loss.masked_select(mask)
                    if vals.numel() == 0:
                        lv = torch.tensor(0.0, device=DEVICE)
                    else:
                        lv = vals.mean()
                    vr += float(lv.item()) * Xv.size(0)
                    vseen += Xv.size(0)

                    if USE_NORMALIZATION:
                        outv = outv * norm_std + norm_mean
                    outv_abs = outv + clim_t
                    all_preds.append(outv_abs.cpu().numpy())
                    all_truths.append(yv.cpu().numpy())

            val_loss = vr / max(1, vseen)
            sched.step(val_loss)

            preds_arr = (
                np.concatenate(all_preds, axis=0)
                if len(all_preds)
                else np.empty((0, 1, H, W))
            )
            actuals_arr = (
                np.concatenate(all_truths, axis=0)
                if len(all_truths)
                else np.empty((0, 1, H, W))
            )

            if preds_arr.size:
                mask_flat = (~ds_tv.sea_mask).ravel()
                preds_flat = preds_arr.reshape(preds_arr.shape[0], -1)
                actuals_flat = actuals_arr.reshape(actuals_arr.shape[0], -1)
                diffs = actuals_flat[:, mask_flat] - preds_flat[:, mask_flat]
                mse = float(np.nanmean(diffs**2))
                mae = float(np.nanmean(np.abs(diffs)))
                rmse = float(np.sqrt(mse))
                mean_ssim_val = compute_mean_ssim(
                    preds_arr, actuals_arr, ds_tv.sea_mask
                )
            else:
                mse = mae = rmse = float("nan")
                mean_ssim_val = None

            if epoch % PRINT_EVERY == 0:
                print(
                    f"Epoch {epoch:03d} Train={train_loss:.6f} ValLoss={val_loss:.6f} | VAL RMSE={rmse:.6f} MAE={mae:.6f} SSIM={mean_ssim_val}"
                )

            # save best
            if preds_arr.size and rmse < best_val_rmse:
                best_val_rmse = rmse
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, checkpoint_path)
                print(
                    f"Saved new best model (val RMSE={rmse:.6f}) -> {checkpoint_path}"
                )
                early_counter = 0
            else:
                early_counter += 1

            if early_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # after training, ensure we have a saved best_state
        if best_state is None:
            torch.save(model.state_dict(), checkpoint_path)
            print("No improvement observed; saved final model as checkpoint.")

    # ---------------- Testing / Inference on test set ----------------
    print("Running inference on TEST set and collecting per-sample metrics...")
    model.eval()
    per_sample_metrics = []  # list of dicts {ts, mae, rmse, ssim}
    with torch.no_grad():
        for Xb, yb, ts_list in tqdm(test_loader, desc="Test infer"):
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            B = Xb.shape[0]
            Xan = Xb - clim_t
            if USE_NORMALIZATION:
                Xan = (Xan - norm_mean) / norm_std
            outb = model(Xan)
            if outb.shape != yb.shape:
                outb = F.interpolate(
                    outb, size=yb.shape[2:], mode="bilinear", align_corners=False
                )
            if USE_NORMALIZATION:
                outb = outb * norm_std + norm_mean
            out_abs = outb + clim_t  # Bx1xHxW
            # compute metrics (land-only)
            for bi in range(B):
                pred = out_abs[bi : bi + 1].cpu().numpy()  # 1x1xHxW
                truth = yb[bi : bi + 1].cpu().numpy()
                mask_flat = (~ds_tv.sea_mask).ravel()
                pred_flat = pred.reshape(-1)
                truth_flat = truth.reshape(-1)
                pred_land = pred_flat[mask_flat]
                truth_land = truth_flat[mask_flat]
                if truth_land.size:
                    dif = truth_land - pred_land
                    mae = float(np.nanmean(np.abs(dif)))
                    rmse = float(np.sqrt(np.nanmean(dif**2)))
                else:
                    mae = float("nan")
                    rmse = float("nan")
                if SSIM_AVAILABLE:
                    try:
                        a = truth[0, 0].astype(np.float64).copy()
                        p = pred[0, 0].astype(np.float64).copy()
                        p[ds_tv.sea_mask] = a[ds_tv.sea_mask]
                        dr = float(a.max() - a.min())
                        dr = dr if dr != 0 else 1e-6
                        s = float(ssim_fn(a, p, data_range=dr))
                    except Exception:
                        s = float("nan")
                else:
                    s = float("nan")
                ts = ts_list[bi] if isinstance(ts_list, list) else None
                per_sample_metrics.append(
                    {"ts": ts, "mae": mae, "rmse": rmse, "ssim": s}
                )

    print(f"Collected {len(per_sample_metrics)} test-sample metrics.")

    # ---------------- Aggregate to daily means ----------------
    print(
        "Aggregating per-sample metrics to daily mean values (if timestamps available)..."
    )
    # build arrays
    ts_list_all = [p["ts"] for p in per_sample_metrics]
    # check fraction of non-None timestamps
    non_none = sum(1 for t in ts_list_all if t is not None)
    print(f"Timestamps present for {non_none}/{len(ts_list_all)} samples.")
    # If timestamps exist, group by UTC date
    daily_dates = []
    daily_mae = []
    daily_rmse = []
    daily_ssim = []
    if non_none >= max(
        30, len(ts_list_all) // 10
    ):  # enough timestamps -> group by date
        # convert all to python datetimes (naive) and date
        ts_py = []
        for t in ts_list_all:
            if t is None:
                ts_py.append(None)
            elif isinstance(t, dt.datetime):
                ts_py.append(t.replace(tzinfo=None))
            else:
                try:
                    # numpy.datetime64 -> python
                    ts_py.append(
                        np.datetime64(t).astype("datetime64[s]").astype(dt.datetime)
                    )
                except Exception:
                    ts_py.append(None)
        # build list of (date, metrics)
        rows = []
        for i, p in enumerate(per_sample_metrics):
            t = ts_py[i]
            if t is None:
                continue
            date = dt.date(t.year, t.month, t.day)
            rows.append((date, p["mae"], p["rmse"], p["ssim"]))
        # group by date
        from collections import defaultdict

        grp = defaultdict(list)
        for date, mae, rmse, ssimv in rows:
            grp[date].append((mae, rmse, ssimv))
        # produce sorted dates across 2024 Jan-Dec (even if some dates missing)
        all_dates_sorted = sorted(grp.keys())
        for d in all_dates_sorted:
            arr = np.array(grp[d], dtype=np.float64)
            mae_mean = float(np.nanmean(arr[:, 0]))
            rmse_mean = float(np.nanmean(arr[:, 1]))
            ssim_mean = float(np.nanmean(arr[:, 2])) if SSIM_AVAILABLE else float("nan")
            daily_dates.append(d)
            daily_mae.append(mae_mean)
            daily_rmse.append(rmse_mean)
            daily_ssim.append(ssim_mean)
    else:
        # fallback: no timestamps -> treat sequence indices and create approximate daily bins (365)
        print(
            "Not enough timestamps: falling back to sample-index -> daily approximate mapping."
        )
        N = len(per_sample_metrics)
        # try to map to 366/365 points across a year
        days_in_year = 366 if ((int(TEST_YEAR) % 4 == 0)) else 365
        # create bins (equal sized)
        preds_arr = per_sample_metrics
        bin_size = max(1, N // days_in_year)
        # aggregate by contiguous bins and assign a pseudo-date
        start_date = dt.date(int(TEST_YEAR), 1, 1)
        idx = 0
        day_counter = 0
        while idx < N and day_counter < 365 + 1:
            end = min(N, idx + bin_size)
            block = preds_arr[idx:end]
            mae_vals = [b["mae"] for b in block if not np.isnan(b["mae"])]
            rmse_vals = [b["rmse"] for b in block if not np.isnan(b["rmse"])]
            ssim_vals = [b["ssim"] for b in block if not np.isnan(b["ssim"])]
            date = start_date + dt.timedelta(days=day_counter)
            if len(mae_vals) == 0:
                mae_mean = float("nan")
            else:
                mae_mean = float(np.nanmean(mae_vals))
            if len(rmse_vals) == 0:
                rmse_mean = float("nan")
            else:
                rmse_mean = float(np.nanmean(rmse_vals))
            if SSIM_AVAILABLE:
                ssim_mean = (
                    float(np.nanmean(ssim_vals)) if len(ssim_vals) > 0 else float("nan")
                )
            else:
                ssim_mean = float("nan")
            daily_dates.append(date)
            daily_mae.append(mae_mean)
            daily_rmse.append(rmse_mean)
            daily_ssim.append(ssim_mean)
            idx = end
            day_counter += 1

    # Convert daily_dates to datetimes for plotting positions
    x_dates = daily_dates
    # prepare x-axis tick locations for months
    # we will convert dates to indices for plotting
    x_idx = list(range(len(x_dates)))
    # month ticks
    month_ticks = []
    month_labels = []
    for i, d in enumerate(x_dates):
        if d.day == 1:
            month_ticks.append(i)
            month_labels.append(d.strftime("%b"))
    # ensure Jan tick exists at start
    if 0 not in month_ticks and len(x_dates) > 0:
        month_ticks.insert(0, 0)
        month_labels.insert(0, x_dates[0].strftime("%b"))

    # ---------------- Plotting: SSIM, MAE, RMSE separately (three PNGs) ----------------
    plt.rcParams["figure.figsize"] = (18, 5)
    # SSIM
    if SSIM_AVAILABLE:
        fig = plt.figure(figsize=(16, 5))
        plt.plot(x_idx, daily_ssim, linewidth=1.0)
        plt.title(
            f"Figure: Time series of SSIM for test data ({TEST_YEAR}) of SwinLSTM",
            fontsize=18,
            fontweight="bold",
        )
        plt.ylabel("SSIM", fontsize=16)
        plt.xlabel("Months →", fontsize=16)
        plt.ylim(-0.1, 1.02)
        plt.xticks(month_ticks, month_labels, fontsize=14)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.25)
        fpath = os.path.join(OUT_DIR, f"Figure_SSIM_{TEST_YEAR}.png")
        plt.tight_layout()
        plt.savefig(fpath, dpi=200)
        plt.close(fig)
        print("Saved SSIM timeseries:", fpath)
    else:
        print("SSIM not available (scikit-image not installed). Skipping SSIM plot.")

    # MAE
    fig = plt.figure(figsize=(16, 5))
    plt.plot(x_idx, daily_mae, linewidth=1.0)
    plt.title(
        f"Figure: Time series of MAE (units) for test data ({TEST_YEAR}) of SwinLSTM",
        fontsize=18,
        fontweight="bold",
    )
    plt.ylabel("MAE (units)", fontsize=16)
    plt.xlabel("Months →", fontsize=16)
    plt.xticks(month_ticks, month_labels, fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.25)
    fpath = os.path.join(OUT_DIR, f"Figure_MAE_{TEST_YEAR}.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=200)
    plt.close(fig)
    print("Saved MAE timeseries:", fpath)

    # RMSE
    fig = plt.figure(figsize=(16, 5))
    plt.plot(x_idx, daily_rmse, linewidth=1.0)
    plt.title(
        f"Figure: Time series of RMSE (units) for test data ({TEST_YEAR}) of SwinLSTM",
        fontsize=18,
        fontweight="bold",
    )
    plt.ylabel("RMSE (units)", fontsize=16)
    plt.xlabel("Months →", fontsize=16)
    plt.xticks(month_ticks, month_labels, fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.25)
    fpath = os.path.join(OUT_DIR, f"Figure_RMSE_{TEST_YEAR}.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=200)
    plt.close(fig)
    print("Saved RMSE timeseries:", fpath)

    # Save daily metrics csv
    csv_path = os.path.join(OUT_DIR, f"daily_metrics_{TEST_YEAR}.csv")
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "date": [d.isoformat() for d in daily_dates],
                "mae": daily_mae,
                "rmse": daily_rmse,
                "ssim": daily_ssim,
            }
        )
        df.to_csv(csv_path, index=False)
        print("Saved daily metrics CSV:", csv_path)
    except Exception:
        print("pandas not available - skipping CSV save.")

    print("Done.")


if __name__ == "__main__":
    main()
