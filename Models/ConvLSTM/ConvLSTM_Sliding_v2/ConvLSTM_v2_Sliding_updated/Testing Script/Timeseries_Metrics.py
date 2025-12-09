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

# Optional imports
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"

# ---------------- CONFIG ----------------
folder_path = "../../../../Dataset/2024"  # folder containing .nc files with 't2m'
base_out_dir = "./"  # same as training script
model_path = os.path.join(base_out_dir, "best_model.pth")
clim_path = os.path.join(base_out_dir, "climatology.npy")
ts_out_dir = os.path.join(base_out_dir, "Timeseries_Metrics")
os.makedirs(ts_out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3  # frames are every 3 hours; we downsample by 3
val_split = 0.18
seed = 42

USE_NORMALIZATION = True
hidden_dim = 128
num_layers = 3
kernel_size = 5
dropout_p = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# ---------------- Model (same as ConvLSTM_v2 training) ----------------
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
        if self.training and self.dropout_p > 0:
            hnext = F.dropout2d(hnext, p=self.dropout_p)
        return hnext, cnext

    def init_hidden(self, b, spatial, device):
        H, W = spatial
        return (
            torch.zeros(b, self.hidden_dim, H, W, device=device),
            torch.zeros(b, self.hidden_dim, H, W, device=device),
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
        hiddens = [l.init_hidden(B, (H, W), x.device) for l in self.layers]
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


# ---------------- Eval dataset with timestamps ----------------
class EvalDataset(torch.utils.data.Dataset):
    """
    Sliding windows over 2024 with the same downsampling & filling logic
    as SlidingMaskDataset, plus synthetic timestamps.

    IMPORTANT: timestamps are artificial but are **forced to lie within 2024**.
    """

    def __init__(self, folder, input_len, target_offset, sample_step):
        self.frames = []
        files = sorted([f for f in os.listdir(folder) if f.endswith(".nc")])
        if not files:
            raise RuntimeError(f"No .nc files in {folder}")

        for fn in files:
            path = os.path.join(folder, fn)
            try:
                ds = Dataset(path)
                if "t2m" not in ds.variables:
                    ds.close()
                    continue
                arr = np.array(ds["t2m"][:], dtype=np.float32)
                if arr.ndim == 3:
                    for t in range(arr.shape[0]):
                        self.frames.append(arr[t])
                elif arr.ndim == 2:
                    self.frames.append(arr)
                ds.close()
            except Exception as e:
                print(f"Skipping {fn}: {e}")

        if len(self.frames) == 0:
            raise RuntimeError("No frames loaded")

        # temporal downsampling
        self.frames = self.frames[::sample_step]

        self.input_len = input_len
        self.target_offset = target_offset

        # land/sea mask
        stack = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stack).all(axis=0)
        self.H, self.W = self.frames[0].shape

        # ---------- Synthetic timestamps strictly inside 2024 ----------
        # We linearly map all frame indices into [2024-01-01, 2024-12-31 23:59].
        self.times = []
        N = len(self.frames)
        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 12, 31, 23, 59)
        total_seconds = (end_time - start_time).total_seconds()

        if N == 1:
            # single point -> center of year
            self.times.append(start_time + timedelta(seconds=total_seconds / 2.0))
        else:
            for i in range(N):
                alpha = i / (N - 1)  # 0 → 1
                sec_offset = alpha * total_seconds
                timestamp = start_time + timedelta(seconds=sec_offset)
                # safety: clamp if any numerical drift
                if timestamp.year < 2024:
                    timestamp = start_time
                elif timestamp.year > 2024:
                    timestamp = end_time
                self.times.append(timestamp)
        # ---------------------------------------------------------------

        # sliding windows
        self.starts = []
        for s in range(len(self.frames) - input_len - target_offset + 1):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len

        inp = np.stack(self.frames[s:e], axis=0).astype(np.float32)
        tgt = self.frames[e - 1 + self.target_offset].astype(np.float32)

        mask = self.sea_mask

        # fill NaNs with land mean
        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        land_vals_tgt = tgt[~mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        tstamp = self.times[e - 1 + self.target_offset]

        return (
            torch.from_numpy(inp_filled).float(),  # C,H,W
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),  # 1,H,W
            tstamp,
        )


# ---------------- Normalization helper ----------------
def compute_norm_from_anomalies(dataset, train_idx, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    clim = climatology.astype(np.float64)
    for i in train_idx:
        X, y, _ = dataset[i]
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


# ---------------- Load dataset, climatology, norm ----------------
print("📦 Loading evaluation dataset...")
dataset = EvalDataset(folder_path, input_len, target_offset, SAMPLE_STEP)
print(f"Total samples (windows): {len(dataset)}")

sea_mask = dataset.sea_mask
H, W = dataset.H, dataset.W
mask_flat = (~sea_mask).ravel()

if not os.path.exists(clim_path):
    raise FileNotFoundError(f"climatology.npy not found at {clim_path}")
climatology = np.load(clim_path).astype(np.float32)
clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)

# same random split logic as training to compute norm stats
np.random.seed(seed)
indices = np.arange(len(dataset))
np.random.shuffle(indices)
split = int(np.floor(val_split * len(dataset)))
val_idx = indices[:split]
train_idx = indices[split:]

print("📏 Computing normalization mean/std from training subset (anomalies)...")
norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    norm_mean, norm_std = compute_norm_from_anomalies(dataset, train_idx, climatology)
print(f"norm_mean = {norm_mean:.6f}, norm_std = {norm_std:.6f}")


# ---------------- Load model ----------------
print("🧠 Loading trained model weights...")
model = ResidualConvLSTM(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout_p=dropout_p,
).to(device)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"best_model.pth not found at {model_path}")
state = torch.load(model_path, map_location=device)
try:
    model.load_state_dict(state)
except Exception as e:
    print("Warning: strict load failed:", e)
    model.load_state_dict(state, strict=False)
model.to(device)
model.eval()


# ---------------- Inference + SSIM/MAE/RMSE ----------------
times = []
ssim_list = []
mae_list = []
rmse_list = []

print("🚀 Running inference over all samples...")
with torch.no_grad():
    for idx in tqdm(range(len(dataset)), ncols=80):
        X, y, tstamp = dataset[idx]

        X = X.unsqueeze(0).to(device)  # 1,C,H,W
        y = y.unsqueeze(0).to(device)  # 1,1,H,W

        X_anom = X - clim_t
        if USE_NORMALIZATION:
            X_anom = (X_anom - norm_mean) / norm_std

        pred = model(X_anom)
        if pred.shape[2:] != y.shape[2:]:
            pred = F.interpolate(
                pred, size=y.shape[2:], mode="bilinear", align_corners=False
            )

        if USE_NORMALIZATION:
            pred = pred * norm_std + norm_mean
        pred_abs = pred + clim_t  # 1,1,H,W

        p2d = pred_abs[0, 0].detach().cpu().numpy()
        a2d = y[0, 0].detach().cpu().numpy()

        # land-only error
        p_land = p2d[~sea_mask]
        a_land = a2d[~sea_mask]
        diff = a_land - p_land

        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(mse))

        # SSIM (sea forced to actual)
        if ssim_fn is not None:
            p_for_ssim = p2d.copy()
            p_for_ssim[sea_mask] = a2d[sea_mask]
            dr = float(a2d.max() - a2d.min())
            if dr == 0.0:
                dr = 1e-6
            try:
                ssim_val = float(ssim_fn(a2d, p_for_ssim, data_range=dr))
            except Exception:
                ssim_val = float("nan")
        else:
            ssim_val = float("nan")

        times.append(tstamp)
        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(ssim_val)

print("✅ Finished inference & metrics.")


# ---------------- Plotting helpers ----------------
def plot_stem_timeseries(times, values, title, ylabel, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(20, 6))

    # markers
    ax.plot(times, values, "o", markersize=3)

    # vertical lines from y=0
    for t, v in zip(times, values):
        ax.vlines(t, ymin=0.0, ymax=v, colors="C0", alpha=0.35, linewidth=0.7)

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Time (2024)", fontsize=14)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=35)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


# ---------------- Make ONLY SSIM, MAE, RMSE plots ----------------
# SSIM
plot_stem_timeseries(
    times,
    ssim_list,
    "Figure 1: Time series of SSIM for test data (2024) of ConvLSTM",
    "SSIM",
    os.path.join(ts_out_dir, "Figure1_SSIM_timeseries_2024.png"),
)

# MAE
plot_stem_timeseries(
    times,
    mae_list,
    "Figure 2: Time series of MAE for test data (2024) of ConvLSTM",
    "MAE (normalized)",
    os.path.join(ts_out_dir, "Figure2_MAE_timeseries_2024.png"),
)

# RMSE
plot_stem_timeseries(
    times,
    rmse_list,
    "Figure 3: Time series of RMSE for test data (2024) of ConvLSTM",
    "RMSE (normalized)",
    os.path.join(ts_out_dir, "Figure3_RMSE_timeseries_2024.png"),
)

print("🎉 Done. All SSIM/MAE/RMSE plots saved in", ts_out_dir)
