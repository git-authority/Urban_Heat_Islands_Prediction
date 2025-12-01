#!/usr/bin/env python3
"""
metrics_and_plots_2024_fixed.py

Corrected placeholder-time logic (uses sample_step spacing). Computes per-sample
metrics on 2024 test set, saves CSV, aggregates to daily means, and saves 3 plots.
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset as ncDataset, num2date
from datetime import datetime, timedelta, timezone
import pandas as pd

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

# ---------- Config ----------
folder_path = "../../../Dataset"
out_dir = "SwinLSTM_Sliding_v1"
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3

batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True

hidden_dim = 192
num_layers = 3
WINDOW_SIZE = 4
ATTN_HEADS = 4
dropout_p = 0.05

TEST_YEAR = "2024"
TEST_YEAR_INT = int(TEST_YEAR)
DPI = 200

from torch.utils.data import Dataset as TorchDataset


class SlidingMaskDataset(TorchDataset):
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
        self.frame_times = []
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)
        self.fallback_year = int(fallback_year) if fallback_year is not None else None

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
            month_files = [f for f in sorted(os.listdir(yd)) if f.endswith(".nc")]

            def month_key(fn):
                name = os.path.splitext(fn)[0].lower()
                return month_map.get(name, 999), fn

            month_files = sorted(month_files, key=month_key)
            for mf in month_files:
                files_paths.append(os.path.join(yd, mf))

        if not files_paths:
            raise ValueError(
                f"No .nc files found in {folder_path} for years {year_dirs}"
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
                        tv = times[t]
                        if not hasattr(tv, "year"):
                            try:
                                tv = (
                                    np.datetime64(tv)
                                    .astype("datetime64[s]")
                                    .astype(datetime)
                                )
                            except Exception:
                                tv = None
                        self.frame_times.append(tv)
                    else:
                        self.frame_times.append(None)
            elif arr.ndim == 2:
                self.frames.append(arr.astype(np.float32))
                if times is not None and len(times) >= 1:
                    tv = times[0]
                    if not hasattr(tv, "year"):
                        try:
                            tv = (
                                np.datetime64(tv)
                                .astype("datetime64[s]")
                                .astype(datetime)
                            )
                        except Exception:
                            tv = None
                    self.frame_times.append(tv)
                else:
                    self.frame_times.append(None)
            ds.close()

        if len(self.frames) == 0:
            raise ValueError("No frames loaded")

        # downsample
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]
            self.frame_times = self.frame_times[:: self.sample_step]

        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent frame shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        # FIXED: use sample_step spacing when creating placeholders
        if any(t is None for t in self.frame_times):
            fy = self.fallback_year
            if fy is None:
                fy = None
            if fy is None:
                try:
                    fy = int(year_dirs[0])
                except Exception:
                    fy = 2000
            total = len(self.frames)
            # IMPORTANT FIX: multiply by sample_step so placeholders reflect real spacing
            times = [
                datetime(fy, 1, 1, tzinfo=timezone.utc)
                + timedelta(hours=i * self.sample_step)
                for i in range(total)
            ]
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

        ts_dt = self.frame_times[e - 1 + self.target_offset]
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        ts_epoch = int(ts_dt.timestamp())

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
            ts_epoch,
        )


# ---------- Model (same as training) ----------
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
        assert H % ws == 0 and W % ws == 0
        x = (
            x.view(B, C, H // ws, ws, W // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
        )
        num_win = (H // ws) * (W // ws)
        return x.view(B * num_win, ws * ws, C), (H // ws, W // ws)

    def window_unpartition(self, x_w, grid_hw, H, W):
        B_numwin, N, C = x_w.shape
        ws = self.window_size
        num_h, num_w = grid_hw
        B = B_numwin // (num_h * num_w)
        x = (
            x_w.view(B, num_h, num_w, ws, ws, C)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, C, H, W)
        )
        return x

    def forward(self, x, hidden):
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.in_proj(comb)
        pad_h = (self.window_size - (H % self.window_size)) % self.window_size
        pad_w = (self.window_size - (W % self.window_size)) % self.window_size
        proj_padded = (
            F.pad(proj, (0, pad_w, 0, pad_h), mode="reflect")
            if (pad_h or pad_w)
            else proj
        )
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


# ---------- Build datasets ----------
all_years = sorted(
    [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
)
if not all_years:
    raise RuntimeError("No year directories found in folder_path")
print("Detected year folders:", all_years)

TEST_YEAR_CHOSEN = TEST_YEAR if TEST_YEAR in all_years else all_years[-1]
trainval_years = [y for y in all_years if y != TEST_YEAR_CHOSEN]
print("Using train years:", trainval_years, "test year:", TEST_YEAR_CHOSEN)

dataset_trainval = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=trainval_years,
    fallback_year=int(TEST_YEAR_CHOSEN),
)
dataset_test = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=[TEST_YEAR_CHOSEN],
    fallback_year=int(TEST_YEAR_CHOSEN),
)

print("Train+Val samples:", len(dataset_trainval), "Test samples:", len(dataset_test))

# ---------- climatology ----------
clim_path = os.path.join(out_dir, "climatology.npy")
if os.path.exists(clim_path):
    climatology = np.load(clim_path)
    print("Loaded climatology from", clim_path)
else:
    n_tv = len(dataset_trainval)
    split_idx = int(np.floor(0.8 * n_tv))
    H, W = dataset_trainval.H, dataset_trainval.W
    clim_sum = np.zeros((H, W), dtype=np.float64)
    count = 0
    for i in range(split_idx):
        _, y, _ = dataset_trainval[i]
        clim_sum += y.numpy().squeeze(0)
        count += 1
    climatology = (clim_sum / max(1, count)).astype(np.float32)
    np.save(clim_path, climatology)
    print("Saved climatology to:", clim_path)


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


n_tv = len(dataset_trainval)
split_idx = int(np.floor(0.8 * n_tv))
train_idx = np.arange(0, split_idx)
norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    print("Computing normalization (mean/std) from anomaly training set...")
    norm_mean, norm_std = compute_norm_from_anomalies(
        dataset_trainval, train_idx, climatology
    )
    print("norm mean, std:", norm_mean, norm_std)

clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)
sea_mask = dataset_trainval.sea_mask

# ---------- model & checkpoint ----------
model = ResidualSwinLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    window_size=WINDOW_SIZE,
    num_heads=ATTN_HEADS,
    dropout_p=dropout_p,
).to(device)


def find_checkpoint(name="best_model.pth"):
    candidate = os.path.join(out_dir, name)
    if os.path.exists(candidate):
        return candidate
    for root, dirs, files in os.walk("."):
        if name in files:
            return os.path.join(root, name)
    return None


ck = find_checkpoint()
if ck is None:
    raise FileNotFoundError("best_model.pth not found. Put it in out_dir or CWD.")
print("Loading checkpoint:", ck)
state = torch.load(ck, map_location=device)
loaded = False
if isinstance(state, dict):
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        try:
            model.load_state_dict(state["state_dict"])
            loaded = True
        except Exception:
            try:
                model.load_state_dict(state["state_dict"], strict=False)
                loaded = True
            except Exception:
                loaded = False
    if not loaded:
        try:
            model.load_state_dict(state)
            loaded = True
        except Exception:
            try:
                model.load_state_dict(state, strict=False)
                loaded = True
            except Exception:
                loaded = False
else:
    try:
        model.load_state_dict(state)
        loaded = True
    except Exception:
        loaded = False

if not loaded:
    raise RuntimeError("Failed to load model state dict.")
model.eval()
print("Model loaded.")

# ---------- inference ----------
test_loader = DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=0
)

mae_list = []
rmse_list = []
ssim_list = []
timestamps = []

with torch.no_grad():
    for batch in test_loader:
        if len(batch) == 3:
            Xb, yb, tsb = batch
        else:
            Xb, yb = batch
            tsb = None

        Xb = Xb.to(device)
        yb = yb.to(device)
        Xanom = Xb - clim_t
        if USE_NORMALIZATION:
            Xanom = (Xanom - norm_mean) / norm_std

        out = model(Xanom)
        if out.shape != yb.shape:
            out = F.interpolate(
                out, size=yb.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            out = out * norm_std + norm_mean
        out_abs = out + clim_t

        out_np = out_abs.cpu().numpy()
        y_np = yb.cpu().numpy()
        B = out_np.shape[0]
        for bi in range(B):
            pred = out_np[bi, 0]
            actual = y_np[bi, 0]
            mask = sea_mask
            land_pred = pred[~mask]
            land_actual = actual[~mask]
            if land_actual.size:
                mae = float(np.nanmean(np.abs(land_actual - land_pred)))
                mse = float(np.nanmean((land_actual - land_pred) ** 2))
                rmse = float(np.sqrt(mse))
            else:
                mae = float("nan")
                rmse = float("nan")
            if ssim_fn is not None:
                a = actual.astype(np.float64).copy()
                p = pred.astype(np.float64).copy()
                try:
                    p[mask] = a[mask]
                except Exception:
                    pass
                dr = float(a.max() - a.min())
                if dr == 0:
                    dr = 1e-6
                try:
                    s = float(ssim_fn(a, p, data_range=dr))
                except Exception:
                    s = float("nan")
            else:
                s = float("nan")

            mae_list.append(mae)
            rmse_list.append(rmse)
            ssim_list.append(s)

            if tsb is not None:
                try:
                    epoch = int(tsb[bi].item())
                    ts_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                except Exception:
                    ts_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
            else:
                ts_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
            timestamps.append(ts_dt)

# ---------- save CSV ----------
csv_path = os.path.join(out_dir, f"test_metrics_{TEST_YEAR_CHOSEN}.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "timestamp_utc", "MAE", "RMSE", "SSIM"])
    for i, (t, mae, rm, ss) in enumerate(
        zip(timestamps, mae_list, rmse_list, ssim_list)
    ):
        w.writerow([i, t.isoformat(), mae, rm, ss])
print("Saved per-sample CSV:", csv_path)

# ---------- daily aggregation & plotting ----------
df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df[df["timestamp_utc"].dt.year == int(TEST_YEAR_CHOSEN)].copy()
df["date"] = df["timestamp_utc"].dt.floor("D")
daily = df.groupby("date")[["MAE", "RMSE", "SSIM"]].mean().reset_index()

full_idx = pd.date_range(
    start=f"{TEST_YEAR_CHOSEN}-01-01",
    end=f"{TEST_YEAR_CHOSEN}-12-31",
    freq="D",
    tz="UTC",
)
daily_full = pd.DataFrame({"date": full_idx})
daily = (
    pd.merge(daily_full, daily, on="date", how="left")
    .sort_values("date")
    .reset_index(drop=True)
)

dates = pd.to_datetime(daily["date"])
try:
    dates_naive = dates.dt.tz_convert(None)
except Exception:
    try:
        dates_naive = dates.dt.tz_localize(None)
    except Exception:
        dates_naive = dates

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
    ax.set_xlabel("Months →", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_name)


fig4 = os.path.join(out_dir, f"Figure4_SSIM_{TEST_YEAR_CHOSEN}.png")
fig5 = os.path.join(out_dir, f"Figure5_MAE_{TEST_YEAR_CHOSEN}.png")
fig6 = os.path.join(out_dir, f"Figure6_RMSE_{TEST_YEAR_CHOSEN}.png")

save_daily_plot(dates_naive, daily["SSIM"].values, "SSIM", fig4, color="#de2d26")
save_daily_plot(dates_naive, daily["MAE"].values, "MAE (units)", fig5, color="#7fc97f")
save_daily_plot(
    dates_naive, daily["RMSE"].values, "RMSE (units)", fig6, color="#2c7fb8"
)

print("All done. CSV + Figures saved to:", out_dir)
