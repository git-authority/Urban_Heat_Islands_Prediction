import os
import numpy as np
import csv
from datetime import datetime, timedelta, timezone

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import pandas as pd

from netCDF4 import Dataset  # only used to get lat/lon if you later want maps

# ---------- IMPORT YOUR MODEL + DATASET DEFINITIONS FROM v4/v5 ----------
# If your file is SwinLSTM_Sliding_v5.py, change the module name below accordingly.
from SwinLSTM_Sliding_v5 import (  # <- change to _v5 if needed
    SlidingMaskDataset,
    ResidualSwinLSTMWithRefine,
    input_len,
    target_offset,
    SAMPLE_STEP,
    WINDOW_SIZE,
    ATTN_HEADS,
    hidden_dim,
    folder_path,
    out_dir,
)

# ---------- OPTIONAL: SSIM (if available) ----------
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"
DPI = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------- CONFIG YOU MAY EDIT ----------------

# Test year (we'll also "pretend" all timestamps are 2024 for plotting labels)
TEST_YEAR = "2024"

# Path to saved climatology (already generated during training)
CLIM_PATH = os.path.join(out_dir, "climatology.npy")

# <<< IMPORTANT: FILL THESE FROM YOUR TRAINING LOGS >>>
# In your training script, you saw something like:
#   norm mean, std (anomalies): 4.1647525 4.7543273
# Put the corresponding values here for THIS model.
NORM_MEAN = 0.0  # <-- REPLACE with your anomaly mean
NORM_STD = 1.0  # <-- REPLACE with your anomaly std
# ----------------------------------------------------

print("norm:", NORM_MEAN, NORM_STD)

# ---------------- LOAD CLIMATOLOGY ----------------
if not os.path.exists(CLIM_PATH):
    raise FileNotFoundError(f"Could not find climatology at {CLIM_PATH}")
climatology = np.load(CLIM_PATH).astype(np.float32)

# ---------------- BUILD TEST DATASET ONLY ----------------
# Detect available year folders and choose TEST_YEAR safely
all_years = sorted(
    [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
)
if TEST_YEAR not in all_years:
    TEST_YEAR = all_years[-1]  # fallback to last year
print("Detected years:", all_years)
print("Using TEST year:", TEST_YEAR)

dataset_test = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=[TEST_YEAR],
)

sea_mask = dataset_test.sea_mask
H, W = dataset_test.H, dataset_test.W

# Broadcast climatology tensor
clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)
sea_mask_t = torch.from_numpy(sea_mask).to(device)
land_mask_t = (~sea_mask_t).to(device).unsqueeze(0).unsqueeze(0)

test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
print("Total test samples:", len(dataset_test))

# ---------------- BUILD AND LOAD MODEL ----------------
model = ResidualSwinLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=3,
    window_size=WINDOW_SIZE,
    num_heads=ATTN_HEADS,
    dropout_p=0.05,
).to(device)

ckpt_path = os.path.join(out_dir, "best_model.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"best_model.pth not found at {ckpt_path}")

state = torch.load(ckpt_path, map_location=device)
try:
    model.load_state_dict(state)
except Exception as e:
    print("Warning: strict load failed, trying non-strict:", e)
    model.load_state_dict(state, strict=False)
model.to(device)
model.eval()


# ---------------- HELPER: gradient loss (not needed for metrics, kept in case) ----------------
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


# ---------------- HELPER: SSIM ----------------
def compute_single_ssim(actual_np, pred_np, sea_mask):
    if ssim_fn is None:
        return float("nan")
    a = actual_np.astype(np.float64).copy()
    p = pred_np.astype(np.float64).copy()
    try:
        p[sea_mask] = a[sea_mask]
    except Exception:
        pass
    dr = float(a.max() - a.min())
    if dr == 0:
        dr = 1e-6
    try:
        s = float(ssim_fn(a, p, data_range=dr))
    except Exception:
        s = float("nan")
    return s


# ---------------- INFERENCE LOOP OVER TEST SET ----------------
test_mae = []
test_rmse = []
test_ssim = []
test_timestamps = []

# We "pretend" year is 2024 by constructing synthetic timestamps
# Each sample corresponds to SAMPLE_STEP * 3 hours (since original was hourly 3h)
dt_hours = 3 * SAMPLE_STEP
base_time = datetime(int(TEST_YEAR), 1, 1, tzinfo=timezone.utc)

with torch.no_grad():
    for idx, (Xb, yb) in enumerate(tqdm(test_loader, desc="Test inference", ncols=80)):
        Xb = Xb.to(device)  # 1 x C x H x W
        yb = yb.to(device)  # 1 x 1 x H x W

        # anomalies
        Xb_anom = Xb - clim_t
        if NORM_STD > 0:
            Xb_anom = (Xb_anom - NORM_MEAN) / NORM_STD

        outb = model(Xb_anom)
        if outb.shape[2:] != yb.shape[2:]:
            outb = F.interpolate(
                outb, size=yb.shape[2:], mode="bilinear", align_corners=False
            )

        # denormalize + add climatology
        if NORM_STD > 0:
            outb = outb * NORM_STD + NORM_MEAN
        outb_abs = outb + clim_t

        pred_np = outb_abs.cpu().numpy()[0, 0]  # H,W
        actual_np = yb.cpu().numpy()[0, 0]

        mask = sea_mask
        land_pred = pred_np[~mask]
        land_actual = actual_np[~mask]

        if land_actual.size:
            mae_v = float(np.nanmean(np.abs(land_actual - land_pred)))
            mse_v = float(np.nanmean((land_actual - land_pred) ** 2))
            rmse_v = float(np.sqrt(mse_v))
        else:
            mae_v = float("nan")
            rmse_v = float("nan")

        s_v = compute_single_ssim(actual_np, pred_np, mask)

        test_mae.append(mae_v)
        test_rmse.append(rmse_v)
        test_ssim.append(s_v)

        # synthetic timestamp
        ts_dt = base_time + timedelta(hours=idx * dt_hours)
        test_timestamps.append(ts_dt)

# ---------------- SAVE PER-SAMPLE CSV ----------------
csv_path = os.path.join(out_dir, f"test_metrics_{TEST_YEAR}_inference_only.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "timestamp_utc", "MAE", "RMSE", "SSIM"])
    for i, (t, mae_v, rmse_v, s_v) in enumerate(
        zip(test_timestamps, test_mae, test_rmse, test_ssim)
    ):
        w.writerow([i, t.isoformat(), mae_v, rmse_v, s_v])

print("Saved per-sample test CSV:", csv_path)

# ---------------- DAILY AGGREGATION ----------------
df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

# Restrict to TEST_YEAR (even though we fabricated timestamps, for safety)
df = df[df["timestamp_utc"].dt.year == int(TEST_YEAR)].copy()
df["date"] = df["timestamp_utc"].dt.floor("D")

daily = df.groupby("date")[["MAE", "RMSE", "SSIM"]].mean().reset_index()

# Ensure full-year coverage Jan 1 -> Dec 31 (2024)
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
    try:
        dates_naive = dates.dt.tz_localize(None)
    except Exception:
        dates_naive = dates

# Fallback if SSIM missing
if "SSIM" not in daily.columns:
    daily["SSIM"] = np.nan

month_locator = mdates.MonthLocator()
month_formatter = mdates.DateFormatter("%b")


def save_daily_plot(dates, vals, metric_name, out_name, color=None):
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot_date(mdates.date2num(dates), vals, "-", lw=1.0, color=color)
    ax.xaxis.set_major_locator(month_locator)
    ax.xaxis.set_major_formatter(month_formatter)
    ax.set_xlim([mdates.date2num(dates.min()), mdates.date2num(dates.max())])
    ax.set_xlabel("Months →", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"Time series of daily {metric_name} for test data ({TEST_YEAR}) of SwinLSTM",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_name)


fig_ssim = os.path.join(out_dir, f"Figure1_SSIM_timeseries_{TEST_YEAR}.png")
fig_mae = os.path.join(out_dir, f"Figure2_MAE_timeseries_{TEST_YEAR}.png")
fig_rmse = os.path.join(out_dir, f"Figure3_RMSE_timeseries_{TEST_YEAR}.png")

save_daily_plot(dates_naive, daily["SSIM"].values, "SSIM", fig_ssim)
save_daily_plot(dates_naive, daily["MAE"].values, "MAE (units)", fig_mae)
save_daily_plot(dates_naive, daily["RMSE"].values, "RMSE (units)", fig_rmse)

print("Done. Time-series figures + CSV saved to:", out_dir)
