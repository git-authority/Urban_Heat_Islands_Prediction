import os
from datetime import datetime, timedelta
from pathlib import Path

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
folder_path = os.path.join("..", "..", "..", "Dataset", "2024")
out_dir = "./"
model_path = os.path.join(out_dir, "best_model.pth")
clim_path = os.path.join(out_dir, "climatology.npy")

ts_out_dir = os.path.join(out_dir, "Timeseries_Metrics")
os.makedirs(ts_out_dir, exist_ok=True)

# must match training
input_len = 8
target_offset = 4
SAMPLE_STEP = 3
val_split = 0.18
seed = 42

hidden_dim = 256
num_layers = 3
WINDOW_SIZE = 8
ATTN_HEADS = 8
dropout_p = 0.05
USE_NORMALIZATION = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# ---------------- SwinLSTM definition (MUST match training) ----------------
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
        # x: B, N, C where N = window_size*window_size
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each: B,N,heads,hd
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,heads,N,N
        attn = attn.softmax(dim=-1)
        out = attn @ v  # B,heads,N,hd
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class SwinLSTMCell(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, window_size=8, num_heads=8, dropout_p=0.05
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        # projection to hidden channels
        self.in_proj = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)
        # local window attention
        self.attn = LocalWindowAttention(
            dim=hidden_dim, window_size=window_size, num_heads=num_heads
        )
        # gates conv (1x1)
        self.gates_conv = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)

    def window_partition(self, x):
        # x: B, C, H, W -> windows: (B * num_windows, ws*ws, C)
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
        # x_w: (B * num_win, ws*ws, C)
        B_numwin, N, C = x_w.shape
        ws = self.window_size
        num_h, num_w = grid_hw
        B = B_numwin // (num_h * num_w)
        x = x_w.view(B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, num_h, ws, num_w, ws
        x = x.view(B, C, H, W)
        return x

    def forward(self, x, hidden):
        # x: B x input_dim x H x W
        # hidden: (h, c) each B x hidden_dim x H x W
        h, c = hidden
        B, _, H, W = x.shape

        comb = torch.cat([x, h], dim=1)  # B, input+hidden, H, W
        proj = self.in_proj(comb)  # B, hidden, H, W

        # pad if necessary so H,W divisible by window_size
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

        conv = self.gates_conv(attn_map)  # B, 4*hidden, H, W
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

        # stronger refinement head with residual shortcut
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.shortcut = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x: B x C x H x W (C = time)
        B, C, H, W = x.size()
        device = x.device
        hiddens = [l.init_hidden(B, (H, W), device) for l in self.layers]
        last = None
        for t in range(C):
            frame = x[:, t : t + 1, :, :]  # B x 1 x H x W
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    # residual connection across layers
                    hnext = hnext + inp
                hiddens[li] = (hnext, cnext)
                inp = hnext
            last = inp
        out = self.refine(last)
        out = out + self.shortcut(last)  # residual shortcut
        return out


# ---------------- Dataset for evaluation (2024 only) ----------------
class EvalDataset(torch.utils.data.Dataset):
    """
    Same filling & downsampling as SlidingMaskDataset,
    but restricted to 2024 and with timestamps.
    """

    def __init__(self, folder_path, input_len, target_offset, sample_step):
        self.frames = []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if not files:
            raise RuntimeError(f"No .nc files in {folder_path}")

        for fn in files:
            path = os.path.join(folder_path, fn)
            try:
                ds = Dataset(path)
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
            raise RuntimeError("No frames loaded")

        # temporal downsampling
        self.frames = self.frames[::sample_step]

        self.input_len = input_len
        self.target_offset = target_offset

        stack = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stack).all(axis=0)
        self.H, self.W = self.frames[0].shape

        # timestamps (pretend 3-hourly, then × sample_step)
        self.times = []
        start_time = datetime(2024, 1, 1, 0, 0)
        dt = timedelta(hours=3 * sample_step)
        for i in range(len(self.frames)):
            self.times.append(start_time + i * dt)

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


def force_to_2024(tlist):
    """Keep month/day/hour, but force year to 2024 in labels."""
    new_times = []
    for t in tlist:
        new_times.append(datetime(2024, t.month, t.day, t.hour, t.minute))
    return new_times


# ---------------- Load dataset + climatology + norm ----------------
print("📦 Loading 2024 evaluation dataset...")
dataset = EvalDataset(folder_path, input_len, target_offset, SAMPLE_STEP)
print(f"Total samples (windows): {len(dataset)}")

sea_mask = dataset.sea_mask
H, W = dataset.H, dataset.W
mask_flat = (~sea_mask).ravel()

if not os.path.exists(clim_path):
    raise FileNotFoundError(f"climatology.npy not found at {clim_path}")
climatology = np.load(clim_path).astype(np.float32)
clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)

# same idea: compute norm stats from a subset of this dataset
np.random.seed(seed)
indices = np.arange(len(dataset))
np.random.shuffle(indices)
split = int(np.floor(val_split * len(dataset)))
val_idx = indices[:split]
train_idx = indices[split:]

print("📏 Computing normalization mean/std from training subset (2024 anomalies)...")
norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    norm_mean, norm_std = compute_norm_from_anomalies(dataset, train_idx, climatology)
print(f"norm_mean = {norm_mean:.6f}, norm_std = {norm_std:.6f}")


# ---------------- Load model ----------------
print("🧠 Loading trained model weights...")
model = ResidualSwinLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    window_size=WINDOW_SIZE,
    num_heads=ATTN_HEADS,
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

print("🚀 Running inference over all 2024 samples...")
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

        # SSIM (force sea = actual)
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

print("✅ Finished inference & metrics (SSIM / MAE / RMSE).")


# ---------------- Plotting helper (stem-style) ----------------
def plot_stem_timeseries(times, values, title, ylabel, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(times, values, "o", markersize=3)
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


# "Pretend" all frames are in 2024 for labels
times_2024 = force_to_2024(times)

# SSIM
plot_stem_timeseries(
    times_2024,
    ssim_list,
    "Figure 1: Time series of SSIM for test data (2024) of SwinLSTM v3",
    "SSIM",
    os.path.join(ts_out_dir, "Figure1_SSIM_timeseries_2024.png"),
)

# MAE
plot_stem_timeseries(
    times_2024,
    mae_list,
    "Figure 2: Time series of MAE for test data (2024) of SwinLSTM v3",
    "MAE (normalized)",
    os.path.join(ts_out_dir, "Figure2_MAE_timeseries_2024.png"),
)

# RMSE
plot_stem_timeseries(
    times_2024,
    rmse_list,
    "Figure 3: Time series of RMSE for test data (2024) of SwinLSTM v3",
    "RMSE (normalized)",
    os.path.join(ts_out_dir, "Figure3_RMSE_timeseries_2024.png"),
)

print("🎉 Done. SSIM / MAE / RMSE plots saved in:", ts_out_dir)
