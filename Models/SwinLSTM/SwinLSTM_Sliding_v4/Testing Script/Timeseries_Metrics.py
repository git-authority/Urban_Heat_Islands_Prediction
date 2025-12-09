import os
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim_fn
except:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
folder_path = "../../../Dataset/2024"
model_path = "./best_model.pth"
clim_path = "./climatology.npy"
out_dir = "./Timeseries_Metrics"
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3
hidden_dim = 192
num_layers = 3
kernel_size = 5
dropout_p = 0.05
USE_NORMALIZATION = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------
# Minimal ConvLSTM model (same architecture as training)
# ---------------------------------------------------------
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
            frame = x[:, t : t + 1]
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                h, c = layer(inp, (h, c))
                if li > 0:
                    h = h + inp
                hiddens[li] = (h, c)
                inp = h
            last = inp
        return self.refine(last)


# ---------------------------------------------------------
# Load dataset frames (2024 folder) and build synthetic 2024 timestamps
# ---------------------------------------------------------
print("Loading 2024 dataset...")

# 1) Load all frames from .nc files
frames_list = []
files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
for fn in files:
    ds = Dataset(os.path.join(folder_path, fn))
    arr = np.array(ds["t2m"][:], dtype=np.float32)
    ds.close()
    if arr.ndim == 3:
        for t in range(arr.shape[0]):
            frames_list.append(arr[t])
    else:
        frames_list.append(arr)

# 2) Downsample in time (same as training)
frames = frames_list[::SAMPLE_STEP]
frames = np.array(frames)

# 3) Build synthetic timestamps and clip strictly to 2024
start = datetime(2024, 1, 1)
dt = timedelta(hours=3 * SAMPLE_STEP)  # 3-hourly data * SAMPLE_STEP

times_full = [start + i * dt for i in range(len(frames))]
keep_idx = [i for i, t in enumerate(times_full) if t.year == 2024]

frames = frames[keep_idx]
times = [times_full[i] for i in keep_idx]

print(f"Total frames in 2024 after downsampling & clipping: {len(frames)}")

H, W = frames.shape[1], frames.shape[2]
sea_mask = np.isnan(frames).all(axis=0)

# ---------------------------------------------------------
# Climatology
# ---------------------------------------------------------
clim = np.load(clim_path).astype(np.float32)
clim_t = (
    torch.from_numpy(clim).float().to(device).unsqueeze(0).unsqueeze(0)
)  # [1,1,H,W]


# ---------------------------------------------------------
# Normalization mean/std
# ---------------------------------------------------------
print("Computing normalization stats...")
land_idx = ~sea_mask
vals = []

for i in range(len(frames) - input_len - target_offset):
    X = frames[i : i + input_len]
    y = frames[i + input_len - 1 + target_offset]
    vals.append((X - clim)[..., land_idx])
    vals.append((y - clim)[land_idx])

vals = np.concatenate([v.flatten() for v in vals])
norm_mean = float(vals.mean())
norm_std = float(vals.std() + 1e-6)
print("norm_mean,", norm_mean, "norm_std,", norm_std)


# ---------------------------------------------------------
# Load model weights
# ---------------------------------------------------------
print("Loading model...")

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


# ---------------------------------------------------------
# Inference loop
# ---------------------------------------------------------
ssim_list, mae_list, rmse_list, t_list = [], [], [], []

print("Running inference on 2024…")
for i in tqdm(range(len(frames) - input_len - target_offset)):
    X = frames[i : i + input_len]
    y = frames[i + input_len - 1 + target_offset]

    # fill NaNs in inputs
    land_vals = X[:, ~sea_mask]
    fill_vals = np.nanmean(land_vals, axis=1)
    X_f = X.copy()
    for t in range(input_len):
        X_f[t][sea_mask] = fill_vals[t]

    # fill NaNs in target
    tgt_f = y.copy()
    tgt_f[sea_mask] = np.nanmean(y[~sea_mask])

    # tensors
    X_t = torch.from_numpy(X_f).unsqueeze(0).float().to(device)  # [1,T,H,W]
    y_t = (
        torch.from_numpy(tgt_f).unsqueeze(0).unsqueeze(0).float().to(device)
    )  # [1,1,H,W]

    # anomalies & normalization
    X_anom = X_t - clim_t
    if USE_NORMALIZATION:
        X_anom = (X_anom - norm_mean) / norm_std

    # prediction
    pred = model(X_anom)

    # de-normalize & add climatology
    if USE_NORMALIZATION:
        pred = pred * norm_std + norm_mean
    pred = pred + clim_t

    p_np = pred[0, 0].detach().cpu().numpy()
    a_np = tgt_f

    # land-only errors
    p_land = p_np[~sea_mask]
    a_land = a_np[~sea_mask]

    mae = np.mean(np.abs(a_land - p_land))
    rmse = np.sqrt(np.mean((a_land - p_land) ** 2))

    # SSIM
    if ssim_fn is not None:
        p_for = p_np.copy()
        p_for[sea_mask] = a_np[sea_mask]
        dr = a_np.max() - a_np.min()
        dr = dr if dr > 0 else 1e-6
        ssim_val = ssim_fn(a_np, p_for, data_range=dr)
    else:
        ssim_val = np.nan

    mae_list.append(mae)
    rmse_list.append(rmse)
    ssim_list.append(ssim_val)

    # timestamp of target frame (all are within 2024)
    t_list.append(times[i + input_len - 1 + target_offset])


# ---------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------
def plot_stem(times, vals, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(times, vals, "o", markersize=3)
    for t, v in zip(times, vals):
        ax.vlines(t, ymin=0, ymax=v, colors="C0", alpha=0.35, linewidth=0.6)

    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (2024)")

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
    "Figure 1: Time series of SSIM for test data (2024) of SwinLSTM",
    "SSIM",
    os.path.join(out_dir, "Figure1_SSIM_timeseries_2024.png"),
)

plot_stem(
    t_list,
    mae_list,
    "Figure 2: Time series of MAE for test data (2024) of SwinLSTM",
    "MAE (normalized)",
    os.path.join(out_dir, "Figure2_MAE_timeseries_2024.png"),
)

plot_stem(
    t_list,
    rmse_list,
    "Figure 3: Time series of RMSE for test data (2024) of SwinLSTM",
    "RMSE (normalized)",
    os.path.join(out_dir, "Figure3_RMSE_timeseries_2024.png"),
)

print("DONE — all 3 plots saved (only 2024).")
