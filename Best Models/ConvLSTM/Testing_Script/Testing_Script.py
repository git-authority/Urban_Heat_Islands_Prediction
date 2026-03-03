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

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"

# ---------------- CONFIG ----------------
folder_path = "../../../Dataset/2024"
model_out_dir = "../Model"
out_dir = "../Normalization"

model_path = os.path.join(model_out_dir, "best_model.pth")
clim_path = os.path.join(out_dir, "climatology.npy")

ts_out_dir = os.path.join(out_dir, "Timeseries_Metrics")
os.makedirs(ts_out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3
val_split = 0.20
seed = 42

hidden_dim = 192
num_layers = 3
kernel_size = 5
dropout_p = 0.05
USE_NORMALIZATION = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------------- Model Definition ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, dropout_p=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad
        )
        self.dropout_p = dropout_p
        self.gn = nn.GroupNorm(min(8, hidden_dim), hidden_dim)

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
    def __init__(self, in_channels, hidden_dim=128, num_layers=3, kernel_size=5, dropout_p=0.05):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvLSTMCell(1 if i == 0 else hidden_dim, hidden_dim, kernel_size, dropout_p)
            for i in range(num_layers)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        hiddens = [l.init_hidden(B, (H, W), x.device) for l in self.layers]
        last = None
        for t in range(C):
            frame = x[:, t:t+1]
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

# ---------------- Dataset ----------------
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, input_len, target_offset, sample_step):
        self.frames = []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])

        for fn in files:
            ds = Dataset(os.path.join(folder_path, fn))
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

        # ---- FIXED TIMESTAMP GENERATION ----
        original_timestep_hours = 1  # raw data is hourly
        start_time = datetime(2024, 1, 1, 0, 0)
        dt = timedelta(hours=original_timestep_hours * sample_step)

        self.times = [
            start_time + i * dt for i in range(len(self.frames))
        ]

        self.starts = list(range(len(self.frames) - input_len - target_offset + 1))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len

        inp = np.stack(self.frames[s:e]).astype(np.float32)
        tgt = self.frames[e - 1 + self.target_offset].astype(np.float32)

        mask = self.sea_mask

        inp_filled = np.empty_like(inp)
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

# ---------------- Load Dataset ----------------
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
train_idx = indices[split:]

def compute_norm_from_anomalies(dataset, train_idx, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    for i in train_idx:
        X, y, _ = dataset[i]
        Xn = X.numpy() - climatology[np.newaxis]
        yn = y.numpy().squeeze(0) - climatology
        arr = np.concatenate([Xn.ravel(), yn.ravel()])
        s += arr.sum()
        ss += (arr**2).sum()
        cnt += arr.size
    mean = s / cnt
    std = np.sqrt(max(ss / cnt - mean**2, 1e-12))
    return mean, std

norm_mean, norm_std = compute_norm_from_anomalies(dataset, train_idx, climatology)

# ---------------- Load Model ----------------
model = ResidualConvLSTMWithRefine(
    input_len, hidden_dim, num_layers, kernel_size, dropout_p
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# ---------------- Inference ----------------
times, ssim_list, mae_list, rmse_list = [], [], [], []

with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        X, y, ts = dataset[idx]
        X = X.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        X_anom = (X - clim_t - norm_mean) / norm_std
        pred = model(X_anom)

        # ---- ENSURE DENORMALIZATION ----
        if USE_NORMALIZATION:
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
            dr = float(a.max() - a.min()) or 1e-6
            ssim_val = ssim_fn(a, p2, data_range=dr)
        else:
            ssim_val = np.nan

        times.append(ts)
        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(ssim_val)

# ---------------- Plotting ----------------
def plot_stem_timeseries(times, values, title, ylabel, save_path):
    filtered = [(t, v) for t, v in zip(times, values) if t.year == 2024]
    if filtered:
        times, values = zip(*filtered)

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(times, values, "o", markersize=3)
    for t, v in zip(times, values):
        ax.vlines(t, 0, v, alpha=0.35)

    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Time (2024)", fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(datetime(2024, 1, 1), datetime(2024, 12, 31, 23, 59))
    plt.xticks(rotation=35)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_stem_timeseries(times, ssim_list,
    "Figure 1: Time series of SSIM (2024)",
    "SSIM",
    os.path.join(ts_out_dir, "Figure1_SSIM_timeseries_2024.png"),
)

plot_stem_timeseries(times, mae_list,
    "Figure 2: Time series of MAE (2024)",
    "MAE",
    os.path.join(ts_out_dir, "Figure2_MAE_timeseries_2024.png"),
)

plot_stem_timeseries(times, rmse_list,
    "Figure 3: Time series of RMSE (2024)",
    "RMSE",
    os.path.join(ts_out_dir, "Figure3_RMSE_timeseries_2024.png"),
)

print("🎉 Done. Fixed 2024-only SSIM / MAE / RMSE plots saved in:", ts_out_dir)