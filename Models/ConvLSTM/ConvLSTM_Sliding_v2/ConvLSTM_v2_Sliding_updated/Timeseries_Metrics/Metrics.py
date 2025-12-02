#!/usr/bin/env python3
"""
make_convlstm_v2_timeseries.py

Produce daily-aggregated SSIM / MAE / RMSE time-series for the ConvLSTM_v2_updated model.
Writes CSV + 3 PNGs to the model folder.
"""
import os
import csv
import numpy as np
from netCDF4 import Dataset as ncDataset
from datetime import datetime, timedelta, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# optional ssim
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

# ---------------- CONFIG ----------------
DATA_FOLDER = (
    "../../../../Dataset/2024"  # same folder you trained on (year 2024 .nc files)
)
MODEL_DIR = "ConvLSTM_v2_updated"  # your out_dir
OUT_DIR = MODEL_DIR
TEST_YEAR = 2024
SAMPLE_STEP = 3
INPUT_LEN = 8
TARGET_OFFSET = 4
USE_NORMALIZATION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
DPI = 200
# ---------------- END CONFIG ----------------

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
print("Device:", DEVICE)


# ---------------- dataset matching your training script ----------------
class SlidingMaskDatasetLocal(TorchDataset):
    def __init__(self, folder_path, input_len=8, target_offset=4, sample_step=1):
        import os
        from netCDF4 import Dataset as _nc

        self.frames = []
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if not files:
            raise ValueError(f"No .nc files found in {folder_path}")

        for fn in files:
            path = os.path.join(folder_path, fn)
            try:
                ds = _nc(path)
                if "t2m" not in ds.variables:
                    ds.close()
                    continue
                var = ds.variables["t2m"]
                arr = np.array(var[:])
                if hasattr(var, "_FillValue"):
                    arr = np.where(arr == var._FillValue, np.nan, arr)
                if hasattr(var, "missing_value"):
                    arr = np.where(arr == var.missing_value, np.nan, arr)
                if arr.ndim == 3:
                    for t in range(arr.shape[0]):
                        self.frames.append(arr[t].astype(np.float32))
                elif arr.ndim == 2:
                    self.frames.append(arr.astype(np.float32))
                ds.close()
            except Exception as e:
                print("Skipping", fn, ":", e)

        if len(self.frames) == 0:
            raise ValueError("No frames loaded")

        # downsample
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]

        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)

        starts = []
        for s in range(len(self.frames) - input_len - target_offset + 1):
            starts.append(s)
        if len(starts) == 0:
            raise ValueError("Not enough frames for chosen input_len/target_offset")
        self.starts = starts

        # create placeholder times (frame index -> timestamp)
        # start at Jan 1 TEST_YEAR 00:00 UTC
        base = datetime(TEST_YEAR, 1, 1, tzinfo=timezone.utc)
        total = len(self.frames)
        self.frame_times = [
            base + timedelta(hours=i * self.sample_step) for i in range(total)
        ]

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
        epoch = int(ts_dt.timestamp())

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
            epoch,
        )


# ---------------- model (match your ResidualConvLSTM signature) ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, dropout_p=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad
        )
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)
        self.dropout_p = dropout_p

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
        if self.training and self.dropout_p > 0:
            hnext = F.dropout2d(hnext, p=self.dropout_p)
        return hnext, cnext

    def init_hidden(self, b, spatial, device):
        H, W = spatial
        return torch.zeros(b, self.hidden_dim, H, W, device=device), torch.zeros(
            b, self.hidden_dim, H, W, device=device
        )


class ResidualConvLSTM(nn.Module):
    def __init__(
        self, in_channels, hidden_dim=128, num_layers=3, kernel_size=5, dropout_p=0.05
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                ConvLSTMCell(
                    in_dim, hidden_dim, kernel_size=kernel_size, dropout_p=dropout_p
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final = nn.Conv2d(hidden_dim, 1, kernel_size=1)

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
        out = self.final(last)
        return out


# ---------------- load dataset ----------------
dataset = SlidingMaskDatasetLocal(
    DATA_FOLDER,
    input_len=INPUT_LEN,
    target_offset=TARGET_OFFSET,
    sample_step=SAMPLE_STEP,
)
print("Test samples:", len(dataset))
H, W = dataset.H, dataset.W
sea_mask = dataset.sea_mask
land_mask = ~sea_mask

# ---------------- climatology ----------------
clim_path = os.path.join(MODEL_DIR, "climatology.npy")
if os.path.exists(clim_path):
    climatology = np.load(clim_path)
    print("Loaded climatology:", clim_path)
else:
    # fallback: compute mean over dataset targets
    print(
        "climatology.npy not found in model dir — computing fallback climatology from dataset targets"
    )
    csum = np.zeros((H, W), dtype=np.float64)
    cnt = 0
    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        csum += y.numpy().squeeze(0)
        cnt += 1
    climatology = (csum / max(1, cnt)).astype(np.float32)
    np.save(clim_path, climatology)
    print("Saved fallback climatology to:", clim_path)


# ---------------- normalization (compute anomaly mean/std) ----------------
def compute_norm_from_dataset(ds):
    s = 0.0
    ss = 0.0
    cnt = 0
    for i in range(len(ds)):
        X, y, _ = ds[i]
        Xn = X.numpy() - climatology[np.newaxis, :, :]
        yn = y.numpy().squeeze(0) - climatology
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
    print("Computing anomaly mean/std (from dataset)...")
    norm_mean, norm_std = compute_norm_from_dataset(dataset)
    print("norm mean, std:", norm_mean, norm_std)

clim_t = (
    torch.from_numpy(climatology).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
)  # 1x1xHxW

# ---------------- build model and load checkpoint ----------------
model = ResidualConvLSTM(
    in_channels=INPUT_LEN, hidden_dim=128, num_layers=3, kernel_size=5, dropout_p=0.05
).to(DEVICE)

# find .pth in model_dir or cwd
ckpt = None
for fn in os.listdir(MODEL_DIR):
    if fn.endswith(".pth"):
        ckpt = os.path.join(MODEL_DIR, fn)
        break
if ckpt is None:
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".pth"):
                ckpt = os.path.join(root, f)
                break
        if ckpt:
            break

if ckpt is None:
    raise FileNotFoundError(
        "No .pth checkpoint found in MODEL_DIR or CWD. Place your best_model.pth in MODEL_DIR."
    )
print("Loading checkpoint:", ckpt)
state = torch.load(ckpt, map_location=DEVICE)
if (
    isinstance(state, dict)
    and "state_dict" in state
    and isinstance(state["state_dict"], dict)
):
    state = state["state_dict"]
try:
    model.load_state_dict(state)
except Exception:
    model.load_state_dict(state, strict=False)
model.eval()
print("Model loaded.")

# ---------------- inference (sample-by-sample) ----------------
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

mae_list = []
rmse_list = []
ssim_list = []
timestamps = []

with torch.no_grad():
    for Xb, yb, epoch in loader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
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

        pred = out_abs.cpu().numpy()[0, 0]
        actual = yb.cpu().numpy()[0, 0]

        land_actual = actual[land_mask]
        land_pred = pred[land_mask]
        if land_actual.size:
            diff = land_actual - land_pred
            mae = float(np.nanmean(np.abs(diff)))
            rmse = float(np.sqrt(np.nanmean(diff**2)))
        else:
            mae = float("nan")
            rmse = float("nan")

        if ssim_fn is not None:
            a = actual.astype(np.float64).copy()
            p = pred.astype(np.float64).copy()
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

        # epoch is integer epoch value returned by dataset
        try:
            ts = int(epoch[0].item()) if hasattr(epoch[0], "item") else int(epoch)
            ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            ts_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        timestamps.append(ts_dt)

# ---------------- save per-sample CSV ----------------
csv_path = os.path.join(OUT_DIR, f"test_metrics_{TEST_YEAR}.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "timestamp_utc", "MAE", "RMSE", "SSIM"])
    for i, (t, mae, rm, ss) in enumerate(
        zip(timestamps, mae_list, rmse_list, ssim_list)
    ):
        w.writerow([i, t.isoformat(), mae, rm, ss])
print("Saved per-sample CSV:", csv_path)

# ---------------- daily aggregation & plotting ----------------
df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df[df["timestamp_utc"].dt.year == int(TEST_YEAR)].copy()
df["date"] = df["timestamp_utc"].dt.floor("D")
daily = df.groupby("date")[["MAE", "RMSE", "SSIM"]].mean().reset_index()

full_idx = pd.date_range(
    start=f"{TEST_YEAR}-01-01", end=f"{TEST_YEAR}-12-31", freq="D", tz="UTC"
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
    dates_naive = dates.dt.tz_localize(None)

# plot helper
plt.rcParams["font.family"] = "Times New Roman"
month_locator = mdates.MonthLocator()
month_formatter = mdates.DateFormatter("%b")


def save_daily_plot(dates, vals, metric_name, out_name, color=None, ylim=None):
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.plot_date(dates, vals, "-", lw=1.4, color=color)
    ax.xaxis.set_major_locator(month_locator)
    ax.xaxis.set_major_formatter(month_formatter)
    ax.set_xlim([mdates.date2num(dates.min()), mdates.date2num(dates.max())])
    ax.set_xlabel("Months →", fontsize=16, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=16, fontweight="bold")
    ax.grid(alpha=0.25)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_name)


fig4 = os.path.join(OUT_DIR, f"Figure4_SSIM_{TEST_YEAR}.png")
fig5 = os.path.join(OUT_DIR, f"Figure5_MAE_{TEST_YEAR}.png")
fig6 = os.path.join(OUT_DIR, f"Figure6_RMSE_{TEST_YEAR}.png")

save_daily_plot(
    dates_naive, daily["SSIM"].values, "SSIM", fig4, color="#d73027", ylim=(0.0, 1.0)
)
save_daily_plot(dates_naive, daily["MAE"].values, "MAE (units)", fig5, color="#7fc97f")
save_daily_plot(
    dates_naive, daily["RMSE"].values, "RMSE (units)", fig6, color="#2b8cbe"
)

print("All done. CSV + Figures saved to:", OUT_DIR)
