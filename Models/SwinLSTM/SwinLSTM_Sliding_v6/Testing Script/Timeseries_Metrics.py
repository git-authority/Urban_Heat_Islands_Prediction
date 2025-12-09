import os
import math
import numpy as np
from netCDF4 import Dataset
import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
FOLDER_PATH = "../../../Dataset"  # top-level dataset folder (contains 2024/)
TEST_YEAR = "2024"  # test year folder name
YEAR_DIR = os.path.join(FOLDER_PATH, TEST_YEAR)

OUT_DIR = "./Timeseries_Metrics_2024"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "./best_model.pth"
CLIM_PATH = "./climatology.npy"

INPUT_LEN = 8
TARGET_OFFSET = 4
SAMPLE_STEP = 3  # must match training
USE_NORMALIZATION = True

HIDDEN_DIM = 160
NUM_LAYERS = 2
WINDOW_SIZE = 4
ATTN_HEADS = 4
DROPOUT_P = 0.08

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
print("SSIM available:", ssim_fn is not None)


# ---------------------------------------------------------
# SwinLSTM model (same as training)
# ---------------------------------------------------------
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
    def __init__(self, in_ch, hidden_ch, window_size=4, num_heads=4, dropout_p=0.05):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.window_size = window_size
        self.dropout_p = dropout_p

        self.input_proj = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size=1)
        self.attn = LocalWindowAttention(dim=hidden_ch, num_heads=num_heads)
        self.gates = nn.Conv2d(hidden_ch, 4 * hidden_ch, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, hidden_ch), num_channels=hidden_ch)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
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
        Hp, Wp, pad_h, pad_w, nh, nw = grid_info
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
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.input_proj(comb)

        x_win, grid_info = self.window_partition(proj)
        attn_out = self.attn(x_win)
        attn_map = self.window_unpartition(attn_out, grid_info)

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
        self.layers = nn.ModuleList(
            [
                SwinLSTMCell(
                    1 if i == 0 else hidden_dim,
                    hidden_dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    dropout_p=dropout_p,
                )
                for i in range(num_layers)
            ]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        hiddens = [layer.init_hidden(B, (H, W), device) for layer in self.layers]
        last = None
        for t in range(C):
            frame = x[:, t : t + 1]
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + inp
                hiddens[li] = (hnext, cnext)
                inp = hnext
            last = inp
        return self.refine(last)


# ---------------------------------------------------------
# Utilities: load frames + timestamps from 2024
# ---------------------------------------------------------
def times_from_ncvar(var):
    """Convert a NetCDF time variable to list[datetime] if possible."""
    try:
        units = getattr(var, "units", None)
        calendar = getattr(var, "calendar", "standard")
        vals = np.array(var[:])
        from netCDF4 import num2date

        dates = num2date(vals, units, calendar=calendar)
        out = []
        for d in dates:
            if isinstance(d, np.datetime64):
                out.append(
                    np.datetime64(d).astype("datetime64[ms]").astype(dt.datetime)
                )
            elif isinstance(d, dt.datetime):
                out.append(d.replace(tzinfo=None))
            else:
                try:
                    out.append(
                        dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
                    )
                except Exception:
                    out.append(None)
        return out
    except Exception:
        return None


print("Loading 2024 dataset from:", YEAR_DIR)
frames = []
times = []

# sort files by month-ish name if possible
files = [f for f in os.listdir(YEAR_DIR) if f.endswith(".nc")]
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


def file_key(fn):
    base = os.path.splitext(fn)[0].lower()
    return (month_order.get(base, 999), fn)


files = sorted(files, key=file_key)

for fn in files:
    path = os.path.join(YEAR_DIR, fn)
    ds = Dataset(path)
    if "t2m" not in ds.variables:
        ds.close()
        continue
    var = ds.variables["t2m"]
    arr = np.array(var[:], dtype=np.float32)

    # fill values -> NaN
    if hasattr(var, "_FillValue"):
        arr = np.where(arr == var._FillValue, np.nan, arr)
    if hasattr(var, "missing_value"):
        arr = np.where(arr == var.missing_value, np.nan, arr)

    # time variable
    times_for_file = None
    possible_time_vars = [v for v in ds.variables if "time" in v.lower()]
    for tname in possible_time_vars:
        try:
            times_for_file = times_from_ncvar(ds.variables[tname])
            if times_for_file is not None:
                break
        except Exception:
            times_for_file = None

    if arr.ndim == 3:
        T = arr.shape[0]
        for t_idx in range(T):
            frames.append(arr[t_idx])
            if times_for_file is not None and len(times_for_file) == T:
                times.append(times_for_file[t_idx])
            else:
                times.append(None)
    elif arr.ndim == 2:
        frames.append(arr)
        times.append(
            times_for_file[0] if (times_for_file and len(times_for_file)) else None
        )

    ds.close()

if len(frames) == 0:
    raise RuntimeError("No frames loaded from 2024 dataset.")

frames = frames[::SAMPLE_STEP]
times = times[::SAMPLE_STEP]

# Filter strictly to TEST_YEAR if timestamps are available
filtered_frames = []
filtered_times = []
year_int = int(TEST_YEAR)
for f, t in zip(frames, times):
    if t is not None:
        if t.year == year_int:
            filtered_frames.append(f)
            filtered_times.append(t)
    else:
        # if timestamp missing, keep it (we assume it's in 2024 since it's in 2024 folder)
        filtered_frames.append(f)
        filtered_times.append(t)

frames = np.array(filtered_frames, dtype=np.float32)
times = filtered_times
print("Total frames after downsample+year filter:", len(frames))

H, W = frames.shape[1], frames.shape[2]
sea_mask = np.isnan(frames).all(axis=0)

# ---------------------------------------------------------
# Load climatology and compute normalization stats
# ---------------------------------------------------------
clim = np.load(CLIM_PATH).astype(np.float32)
if clim.shape != (H, W):
    raise RuntimeError(f"Climatology shape {clim.shape} != frame shape {(H, W)}")

clim_t = (
    torch.from_numpy(clim).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
)  # [1,1,H,W]

print("Computing normalization stats from 2024 anomalies...")
land_idx = ~sea_mask
vals = []
for i in range(len(frames) - INPUT_LEN - TARGET_OFFSET):
    X = frames[i : i + INPUT_LEN]
    y = frames[i + INPUT_LEN - 1 + TARGET_OFFSET]
    vals.append((X - clim)[..., land_idx])
    vals.append((y - clim)[land_idx])

vals = np.concatenate([v.flatten() for v in vals])
norm_mean = float(vals.mean())
norm_std = float(vals.std() + 1e-6)
print("norm_mean:", norm_mean, "norm_std:", norm_std)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = ResidualSwinLSTM(
    in_channels=INPUT_LEN,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    window_size=WINDOW_SIZE,
    num_heads=ATTN_HEADS,
    dropout_p=DROPOUT_P,
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()
print("Loaded model from:", MODEL_PATH)

# ---------------------------------------------------------
# Inference loop over 2024 frames
# ---------------------------------------------------------
ssim_list, mae_list, rmse_list, t_list = [], [], [], []

print("Running inference over 2024...")
for i in tqdm(range(len(frames) - INPUT_LEN - TARGET_OFFSET)):
    X = frames[i : i + INPUT_LEN]
    y = frames[i + INPUT_LEN - 1 + TARGET_OFFSET]
    ts = times[i + INPUT_LEN - 1 + TARGET_OFFSET]

    # if timestamp exists and not in TEST_YEAR, skip (safety)
    if ts is not None and ts.year != year_int:
        continue

    # fill NaNs in inputs
    land_vals = X[:, ~sea_mask]
    fill_vals = np.nanmean(land_vals, axis=1)
    X_f = X.copy()
    for t_idx in range(INPUT_LEN):
        X_f[t_idx][sea_mask] = fill_vals[t_idx]

    # fill NaNs in target
    tgt_f = y.copy()
    tgt_f[sea_mask] = np.nanmean(y[~sea_mask])

    X_t = torch.from_numpy(X_f).unsqueeze(0).float().to(DEVICE)  # [1,T,H,W]
    y_t = (
        torch.from_numpy(tgt_f).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    )  # [1,1,H,W]

    X_anom = X_t - clim_t
    if USE_NORMALIZATION:
        X_anom = (X_anom - norm_mean) / norm_std

    with torch.no_grad():
        pred = model(X_anom)

    if USE_NORMALIZATION:
        pred = pred * norm_std + norm_mean
    pred = pred + clim_t

    p_np = pred[0, 0].cpu().numpy()
    a_np = tgt_f

    p_land = p_np[~sea_mask]
    a_land = a_np[~sea_mask]

    mae = float(np.mean(np.abs(a_land - p_land)))
    rmse = float(np.sqrt(np.mean((a_land - p_land) ** 2)))

    # SSIM
    if ssim_fn is not None:
        p_for = p_np.copy()
        p_for[sea_mask] = a_np[sea_mask]
        dr = float(a_np.max() - a_np.min())
        if dr <= 0:
            dr = 1e-6
        ssim_val = float(ssim_fn(a_np, p_for, data_range=dr))
    else:
        ssim_val = float("nan")

    mae_list.append(mae)
    rmse_list.append(rmse)
    ssim_list.append(ssim_val)

    # timestamp for plotting
    if ts is None:
        # fallback synthetic time if missing
        ts = dt.datetime(year_int, 1, 1) + dt.timedelta(hours=3 * SAMPLE_STEP * i)
    t_list.append(ts)

print("Total plotted points:", len(t_list))


# ---------------------------------------------------------
# Plotting helper (stem-style, like your figures)
# ---------------------------------------------------------
def plot_stem(times, vals, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(24, 6))
    ax.plot(times, vals, "o", markersize=3)
    ax.vlines(times, ymin=0, ymax=vals, colors="C0", alpha=0.30, linewidth=0.6)

    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Time (2024)", fontsize=14)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=35)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


plot_stem(
    t_list,
    ssim_list,
    f"Figure 1: Time series of SSIM for test data ({TEST_YEAR}) of SwinLSTM",
    "SSIM",
    os.path.join(OUT_DIR, f"Figure1_SSIM_timeseries_{TEST_YEAR}.png"),
)

plot_stem(
    t_list,
    mae_list,
    f"Figure 2: Time series of MAE for test data ({TEST_YEAR}) of SwinLSTM",
    "MAE (normalized)",
    os.path.join(OUT_DIR, f"Figure2_MAE_timeseries_{TEST_YEAR}.png"),
)

plot_stem(
    t_list,
    rmse_list,
    f"Figure 3: Time series of RMSE for test data ({TEST_YEAR}) of SwinLSTM",
    "RMSE (normalized)",
    os.path.join(OUT_DIR, f"Figure3_RMSE_timeseries_{TEST_YEAR}.png"),
)

print("DONE — all 3 plots saved (2024 only).")
