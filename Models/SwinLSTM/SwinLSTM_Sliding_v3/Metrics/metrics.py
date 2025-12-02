#!/usr/bin/env python3
"""
make_2024_timeseries_plots.py

Produce daily-aggregated SSIM / MAE / RMSE time-series plots for test year (2024)
from a saved SwinLSTM model and dataset layout.

Place next to your model folder (SwinLSTM_Sliding_v3) and run.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from netCDF4 import Dataset as ncDataset, num2date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

# optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

# ---------------- CONFIG - adjust only if you moved files ----------------
DATA_ROOT = "../../../Dataset"  # top-level dataset folder (2020,2021,...)
MODEL_DIR = (
    "SwinLSTM_Sliding_v3"  # folder containing best_model.pth, climatology.npy, etc.
)
OUT_DIR = MODEL_DIR
TEST_YEAR = "2024"
SAMPLE_STEP = 3
INPUT_LEN = 8
TARGET_OFFSET = 4
USE_NORMALIZATION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # inference sample-by-sample
DPI = 200
# ------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

print(f"Using device: {DEVICE}")


# ---------------- Dataset class (matches your pipeline) ----------------
class SlidingMaskDataset(TorchDataset):
    def __init__(
        self,
        root_folder,
        input_len=8,
        target_offset=4,
        sample_step=1,
        include_years=None,
    ):
        import os

        self.frames = []
        self.frame_times = []
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

        entries = sorted(os.listdir(root_folder))
        year_dirs = []
        for e in entries:
            p = os.path.join(root_folder, e)
            if os.path.isdir(p):
                year_dirs.append(e)

        if include_years is not None:
            year_dirs = [y for y in year_dirs if str(y) in set(map(str, include_years))]

        try:
            year_dirs = sorted(year_dirs, key=lambda x: int(x))
        except Exception:
            year_dirs = sorted(year_dirs)

        files_paths = []
        for y in year_dirs:
            yd = os.path.join(root_folder, y)
            if not os.path.isdir(yd):
                continue
            month_files = [f for f in sorted(os.listdir(yd)) if f.endswith(".nc")]

            def month_key(fn):
                name = os.path.splitext(fn)[0].lower()
                return month_map.get(name, 999), fn

            month_files = sorted(month_files, key=month_key)
            for mf in month_files:
                files_paths.append(os.path.join(yd, mf))

        if not files_paths:
            raise ValueError(
                f"No .nc files found under {root_folder} for years {year_dirs}"
            )

        for path in files_paths:
            fn = os.path.basename(path)
            try:
                ds = ncDataset(path)
            except Exception as e:
                print(f"Skipping {fn} (open failed): {e}")
                continue
            if "t2m" not in ds.variables:
                ds.close()
                continue
            var = ds.variables["t2m"]
            arr = np.array(var[:])
            if hasattr(var, "_FillValue"):
                arr = np.where(arr == var._FillValue, np.nan, arr)
            if hasattr(var, "missing_value"):
                arr = np.where(arr == var.missing_value, np.nan, arr)

            times = None
            if "time" in ds.variables:
                try:
                    tvar = ds.variables["time"]
                    times = num2date(tvar[:], units=tvar.units)
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

        # downsample in time
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]
            self.frame_times = self.frame_times[:: self.sample_step]

        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent frame shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        # fill missing times with regular placeholders spaced by sample_step hours
        if any(t is None for t in self.frame_times):
            base = datetime(
                int(year_dirs[0]) if year_dirs else 2000, 1, 1, tzinfo=timezone.utc
            )
            total = len(self.frames)
            times = [base + timedelta(hours=i * self.sample_step) for i in range(total)]
            for i in range(total):
                if self.frame_times[i] is None:
                    self.frame_times[i] = times[i]

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)

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
        inp = np.stack(self.frames[s:e], axis=0)
        tgt = self.frames[e - 1 + self.target_offset]

        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~self.sea_mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        land_vals_tgt = tgt[~self.sea_mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        ts = self.frame_times[e - 1 + self.target_offset]
        if ts is None:
            ts = datetime.utcnow().replace(tzinfo=timezone.utc)
        if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        epoch = int(ts.timestamp())

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
            epoch,
        )


# ---------------- Model classes (copied minimal - must match your saved state) ----------------
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
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
        return self.proj(out)


class SwinLSTMCell(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, window_size=8, num_heads=8, dropout_p=0.05
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
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
        ph = (ws - (H % ws)) % ws
        pw = (ws - (W % ws)) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        num_win = (Hp // ws) * (Wp // ws)
        x = x.view(B * num_win, ws * ws, C)
        return x, (Hp, Wp, Hp // ws, Wp // ws)

    def window_unpartition(self, x_w, grid_info):
        Hp, Wp, num_h, num_w = grid_info
        ws = self.window_size
        B_numwin, N, C = x_w.shape
        B = B_numwin // (num_h * num_w)
        x = x_w.view(B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)
        return x

    def forward(self, x, hidden):
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.in_proj(comb)
        x_win, grid_info = self.window_partition(proj)
        attn_out = self.attn(x_win)
        attn_map = self.window_unpartition(attn_out, grid_info)
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
        return torch.zeros(b, self.hidden_dim, H, W, device=device), torch.zeros(
            b, self.hidden_dim, H, W, device=device
        )


class ResidualSwinLSTMWithRefine(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_layers=3,
        window_size=8,
        num_heads=8,
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
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.shortcut = nn.Conv2d(hidden_dim, 1, kernel_size=1)

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
        out = out + self.shortcut(last)
        return out


# ---------------- load datasets ----------------
all_years = sorted(
    [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
)
if not all_years:
    raise RuntimeError("No year folders found in DATA_ROOT")
if TEST_YEAR not in all_years:
    print(
        f"Warning: TEST_YEAR {TEST_YEAR} not found under DATA_ROOT. Using last available year {all_years[-1]}"
    )
    TEST_YEAR_USED = all_years[-1]
else:
    TEST_YEAR_USED = TEST_YEAR

train_years = [y for y in all_years if y != TEST_YEAR_USED]
print("Train years:", train_years, "Test year:", TEST_YEAR_USED)

dataset_trainval = SlidingMaskDataset(
    DATA_ROOT,
    input_len=INPUT_LEN,
    target_offset=TARGET_OFFSET,
    sample_step=SAMPLE_STEP,
    include_years=train_years,
)
dataset_test = SlidingMaskDataset(
    DATA_ROOT,
    input_len=INPUT_LEN,
    target_offset=TARGET_OFFSET,
    sample_step=SAMPLE_STEP,
    include_years=[TEST_YEAR_USED],
)

print(f"Train+Val samples: {len(dataset_trainval)}  Test samples: {len(dataset_test)}")
H, W = dataset_trainval.H, dataset_trainval.W
sea_mask = dataset_trainval.sea_mask
land_mask = ~sea_mask

# ---------------- climatology ----------------
clim_path = os.path.join(MODEL_DIR, "climatology.npy")
if os.path.exists(clim_path):
    climatology = np.load(clim_path)
    print("Loaded climatology:", clim_path)
else:
    print("Computing climatology from training portion...")
    # use first 80% of trainval for climatology by default
    n_tv = len(dataset_trainval)
    split_idx = int(np.floor(0.8 * n_tv))
    clim_sum = np.zeros((H, W), dtype=np.float64)
    count = 0
    for i in range(split_idx):
        # FIX: dataset returns (X, y, epoch)
        _, y, _ = dataset_trainval[i]
        clim_sum += y.numpy().squeeze(0)
        count += 1
    climatology = (clim_sum / max(1, count)).astype(np.float32)
    np.save(clim_path, climatology)
    print("Saved climatology to:", clim_path)


# ---------------- normalization ----------------
def compute_norm_from_anomalies(dataset_obj, train_idx_local, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    clim = climatology.astype(np.float64)
    for i in train_idx_local:
        # FIX: dataset returns (X,y,epoch)
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
    n_tv = len(dataset_trainval)
    split_idx = int(np.floor(0.8 * n_tv))
    train_idx = np.arange(0, split_idx)
    print("Computing anomaly mean/std from training set...")
    norm_mean, norm_std = compute_norm_from_anomalies(
        dataset_trainval, train_idx, climatology
    )
    print("norm_mean, norm_std:", norm_mean, norm_std)

clim_t = (
    torch.from_numpy(climatology).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
)  # 1x1xHxW

# ---------------- build model and load checkpoint ----------------
model = ResidualSwinLSTMWithRefine(
    in_channels=INPUT_LEN,
    hidden_dim=256,
    num_layers=3,
    window_size=8,
    num_heads=8,
    dropout_p=0.05,
).to(DEVICE)

# attempt to find checkpoint in MODEL_DIR or CWD
ckpt = None
for fn in os.listdir(MODEL_DIR):
    if fn.endswith(".pth"):
        ckpt = os.path.join(MODEL_DIR, fn)
        break
if ckpt is None:
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".pth"):
                ckpt = os.path.join(root, f)
                break
        if ckpt:
            break

if ckpt is None:
    raise FileNotFoundError(
        "No .pth checkpoint found in MODEL_DIR or CWD. Put best_model.pth in MODEL_DIR."
    )

print("Loading checkpoint:", ckpt)
state = torch.load(ckpt, map_location=DEVICE)
# handle wrapped formats
if isinstance(state, dict):
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
# try load
try:
    model.load_state_dict(state)
except Exception:
    model.load_state_dict(state, strict=False)
model.eval()
print("Model loaded.")

# ---------------- load bias maps if present ----------------
slope_path = os.path.join(MODEL_DIR, "bias_slope_map.npy")
intercept_path = os.path.join(MODEL_DIR, "bias_intercept_map.npy")
if os.path.exists(slope_path) and os.path.exists(intercept_path):
    slope_map = np.load(slope_path)
    intercept_map = np.load(intercept_path)
    print("Loaded bias slope/intercept maps.")
else:
    slope_map = np.ones((H, W), dtype=np.float32)
    intercept_map = np.zeros((H, W), dtype=np.float32)
    print("Bias maps not found; using identity.")

# ---------------- compute blending alpha (on validation portion of trainval) ----------------
print("Computing blending alpha on validation subset (chronological)...")
n_tv = len(dataset_trainval)
split_idx = int(np.floor(0.8 * n_tv))
val_idx = np.arange(split_idx, n_tv)
val_set = Subset(dataset_trainval, val_idx)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

vals_preds = []
vals_pers = []
vals_actuals = []
with torch.no_grad():
    for Xv, yv, _ in val_loader:
        Xv = Xv.to(DEVICE)
        yv = yv.to(DEVICE)
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
        vals_preds.append(outv_abs.cpu().numpy()[0, 0])
        vals_actuals.append(yv.cpu().numpy()[0, 0])
        vals_pers.append(Xv.cpu().numpy()[0, -1])

if len(vals_preds) > 0:
    mask_flat = land_mask.ravel()
    Pp = np.stack(vals_pers, axis=0).reshape(len(vals_pers), -1)[:, mask_flat]
    Pm = np.stack(vals_preds, axis=0).reshape(len(vals_preds), -1)[:, mask_flat]
    A = np.stack(vals_actuals, axis=0).reshape(len(vals_actuals), -1)[:, mask_flat]
    numer = np.nansum((A - Pp) * (Pm - Pp))
    denom = np.nansum((Pm - Pp) ** 2)
    if denom <= 1e-12:
        alpha = 1.0
    else:
        alpha = float(numer / denom)
    alpha = float(np.clip(alpha, 0.0, 1.0))
else:
    alpha = 1.0

print("Blending alpha:", alpha)

# ---------------- inference on test set (sample-by-sample) ----------------
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

mae_list = []
rmse_list = []
ssim_list = []
timestamps = []

with torch.no_grad():
    for Xb, yb, epoch in tqdm(test_loader, desc="Inference"):
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
        X_anom = Xb - clim_t
        if USE_NORMALIZATION:
            X_anom = (X_anom - norm_mean) / norm_std
        out = model(X_anom)
        if out.shape != yb.shape:
            out = F.interpolate(
                out, size=yb.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            out = out * norm_std + norm_mean
        out_abs = out + clim_t
        pred = out_abs.cpu().numpy()[0, 0]
        actual = yb.cpu().numpy()[0, 0]
        pers = Xb.cpu().numpy()[0, -1]

        # blend + bias correct
        pred_blend = alpha * pred + (1.0 - alpha) * pers
        pred_bc = pred_blend * slope_map + intercept_map

        land_actual = actual[land_mask]
        land_pred = pred_bc[land_mask]
        if land_actual.size:
            diff = land_actual - land_pred
            mae = float(np.nanmean(np.abs(diff)))
            rmse = float(np.sqrt(np.nanmean(diff**2)))
        else:
            mae = float("nan")
            rmse = float("nan")

        if ssim_fn is not None:
            a = actual.astype(np.float64).copy()
            p = pred_bc.astype(np.float64).copy()
            try:
                p[sea_mask] = a[sea_mask]
            except Exception:
                pass
            dr = float(a.max() - a.min())
            dr = dr if dr != 0 else 1e-6
            try:
                s = float(ssim_fn(a, p, data_range=dr))
            except Exception:
                s = float("nan")
        else:
            s = float("nan")

        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(s)
        try:
            ts = int(epoch.item())
            ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            ts_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        timestamps.append(ts_dt)

# ---------------- build dataframe and aggregate to daily means ----------------
df = pd.DataFrame(
    {"timestamp": timestamps, "MAE": mae_list, "RMSE": rmse_list, "SSIM": ssim_list}
)
df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC").dt.floor("D")
# restrict to TEST_YEAR_USED
df = df[df["date"].dt.year == int(TEST_YEAR_USED)].copy()

# daily mean
daily = df.groupby("date")[["MAE", "RMSE", "SSIM"]].mean().reset_index()

# full-year daily index and merge to fill missing days
start = datetime(int(TEST_YEAR_USED), 1, 1, tzinfo=timezone.utc)
end = datetime(int(TEST_YEAR_USED), 12, 31, tzinfo=timezone.utc)
full_idx = pd.date_range(start=start, end=end, freq="D", tz="UTC")
daily_full = pd.DataFrame({"date": full_idx})
daily = (
    pd.merge(daily_full, daily, on="date", how="left")
    .sort_values("date")
    .reset_index(drop=True)
)

# ---------------- plotting helpers (match your original aesthetic) ----------------
plt.rcParams["font.family"] = "Times New Roman"
LABEL_FONT = 18
TICK_FONT = 14
TITLE_FONT = 20

month_pos = mdates.MonthLocator()
month_fmt = mdates.DateFormatter("%b")


def pretty_plot_dates(dates, values, title, ylabel, outpath, ylimit=None, color=None):
    fig, ax = plt.subplots(figsize=(22, 6))
    if color is None:
        ax.plot(dates, values, linewidth=1.6)
    else:
        ax.plot(dates, values, linewidth=1.6, color=color)
    ax.set_title(title, fontsize=TITLE_FONT, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT, fontweight="bold")
    ax.set_xlabel("Months →", fontsize=20, fontweight="bold")
    ax.xaxis.set_major_locator(month_pos)
    ax.xaxis.set_major_formatter(month_fmt)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(alpha=0.25)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)


dates = pd.to_datetime(daily["date"]).dt.tz_convert(None)  # naive for plotting

# set y-limits for SSIM to [0,1] like your images
ssim_vals = daily["SSIM"].values
mae_vals = daily["MAE"].values
rmse_vals = daily["RMSE"].values

out_ssim = os.path.join(OUT_DIR, f"Figure4_SSIM_{TEST_YEAR_USED}.png")
out_mae = os.path.join(OUT_DIR, f"Figure5_MAE_{TEST_YEAR_USED}.png")
out_rmse = os.path.join(OUT_DIR, f"Figure6_RMSE_{TEST_YEAR_USED}.png")

pretty_plot_dates(
    dates,
    ssim_vals,
    f"Figure: Time series of SSIM for test data ({TEST_YEAR_USED}) of SwinLSTM",
    "SSIM",
    out_ssim,
    ylimit=(0.0, 1.0),
    color="#d73027",
)
pretty_plot_dates(
    dates,
    mae_vals,
    f"Figure: Time series of MAE (units) for test data ({TEST_YEAR_USED}) of SwinLSTM",
    "MAE (units)",
    out_mae,
    color="#7fc97f",
)
pretty_plot_dates(
    dates,
    rmse_vals,
    f"Figure: Time series of RMSE (units) for test data ({TEST_YEAR_USED}) of SwinLSTM",
    "RMSE (units)",
    out_rmse,
    color="#2b8cbe",
)

# copy to /mnt/data for container preview if available
mnt = "/mnt/data"
if os.path.isdir(mnt) and os.access(mnt, os.W_OK):
    try:
        copyfile(out_ssim, os.path.join(mnt, os.path.basename(out_ssim)))
        copyfile(out_mae, os.path.join(mnt, os.path.basename(out_mae)))
        copyfile(out_rmse, os.path.join(mnt, os.path.basename(out_rmse)))
        print("Also copied figures to /mnt/data/")
    except Exception:
        pass

print("Done. Figures saved to:", OUT_DIR)
