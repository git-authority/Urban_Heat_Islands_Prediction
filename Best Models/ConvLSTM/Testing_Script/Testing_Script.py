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
folder_path = "../Dataset/2024"  # 2024 data with t2m
out_dir = "../ConvLSTM_Sliding_New"        # trained model folder

model_path = os.path.join(out_dir, "best_model.pth")
clim_path = os.path.join(out_dir, "climatology.npy")

ts_out_dir = os.path.join(out_dir, "Timeseries_Metrics")
os.makedirs(ts_out_dir, exist_ok=True)

# must match training (ConvLSTM_Sliding_v5)
input_len = 8
target_offset = 4
SAMPLE_STEP = 3
val_split = 0.20  # only used for norm stats split
seed = 42

hidden_dim = 192
num_layers = 3
kernel_size = 5
dropout_p = 0.05
USE_NORMALIZATION = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# ---------------- Model definition (same as training) ----------------
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
        out = self.refine(last)
        return out


# ---------------- Dataset for evaluation (2024) ----------------
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, input_len, target_offset, sample_step):
        self.frames = []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if not files:
            raise RuntimeError(f"No .nc files in {folder_path}")

        for fn in files:
            path = os.path.join(folder_path, fn)
            ds = Dataset(path)
            arr = np.array(ds["t2m"][:], dtype=np.float32)
            if arr.ndim == 3:
                for t in range(arr.shape[0]):
                    self.frames.append(arr[t])
            else:
                self.frames.append(arr)
            ds.close()

        self.frames = self.frames[::sample_step]

        self.input_len = input_len
        self.target_offset = target_offset

        stack = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stack).all(axis=0)
        self.H, self.W = self.frames[0].shape

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

        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            fill = float(np.nanmean(frame[~mask]))
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        fill_t = float(np.nanmean(tgt[~mask]))
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        tstamp = self.times[e - 1 + self.target_offset]

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
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


# ---------------- Load dataset + climatology ----------------
print("📦 Loading evaluation dataset (2024)...")
dataset = EvalDataset(folder_path, input_len, target_offset, SAMPLE_STEP)

sea_mask = dataset.sea_mask
H, W = dataset.H, dataset.W

climatology = np.load(clim_path).astype(np.float32)
clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)

np.random.seed(seed)
indices = np.arange(len(dataset))
np.random.shuffle(indices)
split = int(np.floor(val_split * len(dataset)))
val_idx = indices[:split]
train_idx = indices[split:]

norm_mean, norm_std = compute_norm_from_anomalies(dataset, train_idx, climatology)


# ---------------- Load model ----------------
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


# ---------------- Inference + metrics ----------------
times, ssim_list, mae_list, rmse_list = [], [], [], []

with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        X, y, ts = dataset[idx]
        X = X.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        X_anom = X - clim_t
        X_anom = (X_anom - norm_mean) / norm_std

        pred = model(X_anom)
        pred = pred * norm_std + norm_mean
        pred = pred + clim_t

        p = pred[0, 0].cpu().numpy()
        a = y[0, 0].cpu().numpy()

        diff = a[~sea_mask] - p[~sea_mask]
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))

        if ssim_fn is not None:
            p2 = p.copy()
            p2[sea_mask] = a[sea_mask]
            dr = float(a.max() - a.min())
            if dr == 0:
                dr = 1e-6
            ssim_val = ssim_fn(a, p2, data_range=dr)
        else:
            ssim_val = np.nan

        times.append(ts)
        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(ssim_val)


# ---------------- Plotting ----------------
def plot_stem_timeseries(times, values, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(times, values, "o", markersize=3)
    for t, v in zip(times, values):
        ax.vlines(t, 0, v, alpha=0.35)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Time (2024)", fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=35)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


plot_stem_timeseries(
    times, ssim_list,
    "Figure 1: Time series of SSIM (2024)",
    "SSIM",
    os.path.join(ts_out_dir, "Figure1_SSIM_timeseries_2024.png"),
)

plot_stem_timeseries(
    times, mae_list,
    "Figure 2: Time series of MAE (2024)",
    "MAE",
    os.path.join(ts_out_dir, "Figure2_MAE_timeseries_2024.png"),
)

plot_stem_timeseries(
    times, rmse_list,
    "Figure 3: Time series of RMSE (2024)",
    "RMSE",
    os.path.join(ts_out_dir, "Figure3_RMSE_timeseries_2024.png"),
)

print("🎉 Done. SSIM / MAE / RMSE plots saved in:", ts_out_dir)