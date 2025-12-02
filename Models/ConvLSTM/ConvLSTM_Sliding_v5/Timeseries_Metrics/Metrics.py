import os
import csv
import numpy as np
from netCDF4 import Dataset as ncDataset
from datetime import datetime, timedelta, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

# ---------------- CONFIG ----------------
PARENT_DATA_FOLDER = "../../../Dataset"  # parent containing 2020,2021,... folders
OUT_DIR = "Metrics"
os.makedirs(OUT_DIR, exist_ok=True)
TRAIN_YEARS = ["2020", "2021", "2022", "2023"]
TEST_YEAR = "2024"
SAMPLE_STEP = 3
INPUT_LEN = 8
TARGET_OFFSET = 4
USE_NORMALIZATION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
DPI = 200

# model hyperparams (must match training)
HIDDEN_DIM = 192
NUM_LAYERS = 3
KERNEL_SIZE = 5
DROPOUT_P = 0.05
# ----------------------------------------

print("Device:", DEVICE)


# ---------------- Dataset (adapted) ----------------
class SlidingMaskDatasetYears(TorchDataset):
    """Collect .nc files from given parent folder but only include files belonging to the provided years list.
    If years is None -> include all years found.
    """

    def __init__(
        self, parent_folder, years=None, input_len=8, target_offset=4, sample_step=1
    ):
        import os
        from netCDF4 import Dataset as _nc

        self.frames = []
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

        files_paths = []

        # list entries in parent_folder
        entries = sorted(os.listdir(parent_folder))
        year_dirs = [
            e for e in entries if os.path.isdir(os.path.join(parent_folder, e))
        ]
        # numeric sort preferred
        try:
            year_dirs = sorted(year_dirs, key=lambda x: int(x))
        except Exception:
            year_dirs = sorted(year_dirs)

        if years is not None:
            # keep only requested years that exist
            year_dirs = [y for y in year_dirs if y in years]

        for y in year_dirs:
            yd = os.path.join(parent_folder, y)
            month_files = [f for f in sorted(os.listdir(yd)) if f.endswith(".nc")]
            for mf in month_files:
                files_paths.append(os.path.join(yd, mf))

        # fallback: allow direct .nc files under parent if no year dirs
        if not files_paths:
            files_paths = [
                os.path.join(parent_folder, f)
                for f in sorted(os.listdir(parent_folder))
                if f.endswith(".nc")
            ]

        if not files_paths:
            raise ValueError(f"No .nc files found in {parent_folder} for years={years}")

        for path in files_paths:
            fn = os.path.basename(path)
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
                print(f"Skipping {fn}: {e}")

        if len(self.frames) == 0:
            raise ValueError("No frames loaded")

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

        # build frame times assuming chronological order starting at Jan 1 of first year in years or parent
        try:
            base_year = (
                int(years[0])
                if years
                else int(
                    sorted(
                        [
                            d
                            for d in os.listdir(parent_folder)
                            if os.path.isdir(os.path.join(parent_folder, d))
                        ]
                    )[0]
                )
            )
        except Exception:
            base_year = 2020
        base = datetime(base_year, 1, 1, tzinfo=timezone.utc)
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
            torch.from_numpy(inp_filled).float(),  # C,H,W
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),  # 1,H,W
            epoch,
        )


# ---------------- model classes (same as training) ----------------
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


class ResidualConvLSTMWithRefine(nn.Module):
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


# ---------------- helper SSIM ----------------
def compute_ssim_safe(a, p, sea_mask):
    if ssim_fn is None:
        return float("nan")
    a_f = a.astype(np.float64).copy()
    p_f = p.astype(np.float64).copy()
    try:
        p_f[sea_mask] = a_f[sea_mask]
    except Exception:
        pass
    dr = float(a_f.max() - a_f.min())
    dr = dr if dr != 0 else 1e-6
    try:
        return float(ssim_fn(a_f, p_f, data_range=dr))
    except Exception:
        return float("nan")


# ---------------- Build TRAIN dataset (2020-2023) to compute climatology & norm ----------------
print(
    "Loading TRAIN years dataset (2020-2023) to compute climatology and normalization..."
)
train_ds = SlidingMaskDatasetYears(
    PARENT_DATA_FOLDER,
    years=TRAIN_YEARS,
    input_len=INPUT_LEN,
    target_offset=TARGET_OFFSET,
    sample_step=SAMPLE_STEP,
)
H, W = train_ds.H, train_ds.W
sea_mask = train_ds.sea_mask
land_mask = ~sea_mask

# compute climatology from train targets
clim_path = os.path.join(OUT_DIR, "climatology.npy")
if os.path.exists(clim_path):
    climatology = np.load(clim_path)
    print("Loaded climatology:", clim_path)
else:
    print("Computing climatology from TRAIN targets...")
    csum = np.zeros((H, W), dtype=np.float64)
    cnt = 0
    for i in range(len(train_ds)):
        _, y, _ = train_ds[i]
        csum += y.numpy().squeeze(0)
        cnt += 1
    climatology = (csum / max(1, cnt)).astype(np.float32)
    np.save(clim_path, climatology)
    print("Saved climatology to:", clim_path)

# compute normalization (anomaly mean/std) from TRAIN set


def compute_norm_from_dataset(ds, climatology):
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
    print("Computing anomaly mean/std from TRAIN dataset...")
    norm_mean, norm_std = compute_norm_from_dataset(train_ds, climatology)
    print("norm mean, std:", norm_mean, norm_std)

clim_t = torch.from_numpy(climatology).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

# ---------------- build model and load checkpoint ----------------
model = ResidualConvLSTMWithRefine(
    in_channels=INPUT_LEN,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    kernel_size=KERNEL_SIZE,
    dropout_p=DROPOUT_P,
).to(DEVICE)

ckpt = None
# search for best_model.pth in training out_dir first
cand = os.path.join("ConvLSTM_Sliding_v5", "best_model.pth")
if os.path.exists(cand):
    ckpt = cand
else:
    # fallback: any .pth in current workdir or OUT_DIR
    for fn in os.listdir("."):
        if fn.endswith(".pth"):
            ckpt = os.path.abspath(fn)
            break
    if ckpt is None:
        for fn in os.listdir(OUT_DIR):
            if fn.endswith(".pth"):
                ckpt = os.path.join(OUT_DIR, fn)
                break

if ckpt is None:
    raise FileNotFoundError(
        "No .pth checkpoint found. Place best_model.pth in ConvLSTM_Sliding_v5 or OUT_DIR."
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

# ---------------- Build TEST dataset for year 2024 ----------------
print(f"Loading TEST dataset for year {TEST_YEAR} ...")
test_ds = SlidingMaskDatasetYears(
    PARENT_DATA_FOLDER,
    years=[TEST_YEAR],
    input_len=INPUT_LEN,
    target_offset=TARGET_OFFSET,
    sample_step=SAMPLE_STEP,
)
print("Test samples:", len(test_ds))

# ---------------- inference sample-by-sample ----------------
loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

mae_list = []
rmse_list = []
ssim_list = []
timestamps = []

with torch.no_grad():
    for Xb, yb, epoch in loader:
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
            s = compute_ssim_safe(actual, pred, sea_mask)
        else:
            s = float("nan")

        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(s)

        try:
            ts = int(epoch[0].item()) if hasattr(epoch[0], "item") else int(epoch)
            ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            ts_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        timestamps.append(ts_dt)

# ---------------- write CSV ----------------
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

plt.rcParams["font.family"] = "Times New Roman"
month_locator = mdates.MonthLocator()
month_formatter = mdates.DateFormatter("%b")


def save_daily_plot(dates, vals, metric_name, out_name, ylim=None):
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.plot(dates, vals, "-", lw=1.6)
    ax.set_title(f"{metric_name} — {TEST_YEAR}", fontsize=18, fontweight="bold")
    ax.xaxis.set_major_locator(month_locator)
    ax.xaxis.set_major_formatter(month_formatter)
    ax.set_xlabel("Months →", fontsize=16, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=16, fontweight="bold")
    ax.grid(alpha=0.25)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_name)


fig4 = os.path.join(OUT_DIR, f"SSIM_{TEST_YEAR}.png")
fig5 = os.path.join(OUT_DIR, f"MAE_{TEST_YEAR}.png")
fig6 = os.path.join(OUT_DIR, f"RMSE_{TEST_YEAR}.png")

save_daily_plot(dates_naive, daily["SSIM"].values, "SSIM", fig4, ylim=(0.0, 1.0))
save_daily_plot(dates_naive, daily["MAE"].values, "MAE (K)", fig5)
save_daily_plot(dates_naive, daily["RMSE"].values, "RMSE (K)", fig6)

print("Done. CSV + Figures saved to:", OUT_DIR)
