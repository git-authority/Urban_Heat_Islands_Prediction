import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"   # reduce fragmentation

import re
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.amp import GradScaler, autocast
from skimage.metrics import structural_similarity as ssim_metric
import xarray as xr

# ==========================================
# 1. CONFIGURATION
# ==========================================
base_dir    = "/kaggle/input/datasets/legion2022/uhi-dataset/Dataset"
script_dir  = "/kaggle/working"

heatmap_folder  = os.path.join(script_dir, "Heatmaps_t2m_gray_11x11")
model_save_path = os.path.join(script_dir, "ConvLSTM_t2m_best.pth")
eval_out        = os.path.join(script_dir, "Eval_ConvLSTM_2024")
stats_path      = os.path.join(script_dir, "dataset_stats.json")
mask_path       = os.path.join(script_dir, "land_mask.npy")
losses_path     = os.path.join(script_dir, "train_val_losses_convlstm.json")
checkpoint_path = os.path.join(script_dir, "training_checkpoint_convlstm.pth")
# ADDED: path to save grid lat/lon coordinates for axis tick labels
coords_path     = os.path.join(script_dir, "grid_coords.json")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

input_len     = 8
target_offset = 4

img_size    = 11
IN_CHANNELS = 3
batch_size  = 16
num_epochs  = 50
lr          = 1e-4
HIDDEN_DIMS = (32, 64, 64)
kernel_size = 3
dropout_p   = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
print(f"✅ Dataset Path: {base_dir}")
print(f"✅ input_len={input_len} | IN_CHANNELS={IN_CHANNELS} | "
      f"target_offset={target_offset} → 12 h-ahead prediction")


# ==========================================
# ADDED: PLOT HELPER FUNCTIONS
# ==========================================
def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for integer n (1→'st', 2→'nd', 3→'rd', else 'th')."""
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _fmt_date(dt) -> str:
    """Format a datetime-like object as '1st January, 2024'."""
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    day = dt.day
    return f"{day}{_ordinal_suffix(day)} {dt.strftime('%B')}, {dt.year}"


def _fmt_time(dt) -> str:
    """Format a datetime-like object as '12am', '9pm', '3pm', etc."""
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    h = dt.hour
    if h == 0:
        return "12am"
    elif h < 12:
        return f"{h}am"
    elif h == 12:
        return "12pm"
    else:
        return f"{h - 12}pm"


def _build_suptitle(seq_timestamps, tgt_timestamp) -> str:
    """
    Build the figure super-title in format:
    'Input: 1st January, 2024 | 12am - 9pm  |  Output: 2nd January, 2024 | 9am'

    seq_timestamps : list of Python datetime objects (input sequence)
    tgt_timestamp  : np.datetime64 or Python datetime (target frame)
    """
    t_start = seq_timestamps[0]
    t_end   = seq_timestamps[-1]
    inp_date = _fmt_date(t_start)
    inp_t0   = _fmt_time(t_start)
    inp_t1   = _fmt_time(t_end)
    out_date = _fmt_date(tgt_timestamp)
    out_time = _fmt_time(tgt_timestamp)
    return (f"Input: {inp_date} | {inp_t0} - {inp_t1}"
            f"  |  Output: {out_date} | {out_time}")


# ==========================================
# 2. DATA GENERATION & STATS
# ==========================================
def ensure_data_ready():
    """
    Ensures PNGs + land mask exist and global min/max stats are available.
    ADDED: saves grid lat/lon to coords_path for plot axis ticks.
    """
    if not os.path.exists(stats_path):
        print("⚠️ Stats file missing. Scanning NetCDF files...")
        years = ['2020', '2021', '2022', '2023', '2024']
        nc_files = []
        for y in years:
            year_dir = os.path.join(base_dir, y)
            if os.path.exists(year_dir):
                for root, _, files in os.walk(year_dir):
                    for f in files:
                        if f.endswith('.nc'):
                            nc_files.append(os.path.join(root, f))

        if not nc_files:
            raise RuntimeError(f"❌ No NetCDF files found in {base_dir}")

        all_vals = []
        for path in tqdm(sorted(nc_files), desc="Scanning Stats"):
            try:
                ds = xr.open_dataset(path)
                da = ds['t2m']
                all_vals.append(da.values.flatten())
            except Exception:
                pass

        full_arr = np.concatenate(all_vals)
        g_min    = float(np.nanmin(full_arr))
        g_max    = float(np.nanmax(full_arr))

        with open(stats_path, 'w') as f:
            json.dump({"min": g_min, "max": g_max}, f)
        print(f"✅ Stats saved: Min={g_min:.2f}K, Max={g_max:.2f}K")

    with open(stats_path, 'r') as f:
        stats = json.load(f)
        g_min, g_max = stats['min'], stats['max']

    pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    if len(pngs) > 1000:
        print(f"✅ Found {len(pngs)} existing images. Skipping generation.")
        return g_min, g_max

    print("⚠️ Images missing. Generating 11×11 PNGs...")

    years    = ['2020', '2021', '2022', '2023', '2024']
    nc_files = []
    for y in years:
        year_dir = os.path.join(base_dir, y)
        if os.path.exists(year_dir):
            for root, _, files in os.walk(year_dir):
                for f in files:
                    if f.endswith('.nc'):
                        nc_files.append(os.path.join(root, f))

    all_t2m = []
    for path in tqdm(sorted(nc_files), desc="Reading NetCDF"):
        try:
            ds = xr.open_dataset(path)
            da = ds['t2m']
            if 'valid_time' in da.dims:
                da = da.rename({'valid_time': 'time'})
            if 'time' in da.dims and 'step' in da.dims:
                if 'valid_time' in ds:
                    new_time = ds['valid_time'].values.reshape(-1)
                else:
                    new_time = np.arange(da.sizes['time'] * da.sizes['step'])
                da = da.stack(new_time=('time', 'step'))
                da = da.assign_coords(new_time=('new_time', new_time))
                da = da.rename({'new_time': 'time'})
                da = da.drop_vars(['step'], errors='ignore')
            if 'latitude'  in da.coords: da = da.sortby('latitude')
            if 'longitude' in da.coords: da = da.sortby('longitude')
            all_t2m.append(da)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    t2m_all = xr.concat(all_t2m, dim='time')
    t2m_arr = t2m_all.values
    times   = t2m_all.time.values

    if t2m_arr.shape[1:] != (11, 11):
        raise ValueError(
            f"Expected native grid shape (11,11), got {t2m_arr.shape[1:]}")

    print(f"✅ Native spatial shape: {t2m_arr.shape[1]}×{t2m_arr.shape[2]}")

    # ADDED: save lat/lon coordinates so evaluation plots have real axis ticks
    if not os.path.exists(coords_path):
        try:
            raw_lats = t2m_all.latitude.values   # ascending from sortby
            raw_lons = t2m_all.longitude.values
            # flipud matches the vertical flip applied to every PNG
            lats_flipped = np.flipud(raw_lats).tolist()
            lons_list    = raw_lons.tolist()
            with open(coords_path, 'w') as _cf:
                json.dump({'lats': lats_flipped, 'lons': lons_list}, _cf)
            print(f"✅ Grid coords saved: "
                  f"lat {lats_flipped[0]:.2f}→{lats_flipped[-1]:.2f}, "
                  f"lon {lons_list[0]:.2f}→{lons_list[-1]:.2f}")
        except Exception as _e:
            print(f"⚠️  Could not save grid coords: {_e}")

    land_mask_raw = None
    for i in range(len(t2m_arr)):
        if not np.all(np.isnan(t2m_arr[i])):
            land_mask_raw = np.flipud(~np.isnan(t2m_arr[i]))
            break
    if land_mask_raw is not None:
        np.save(mask_path, land_mask_raw)
        print(f"✅ Land mask saved ({land_mask_raw.shape}): "
              f"{land_mask_raw.sum()} land / {land_mask_raw.size} total pixels")

    for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
        if pd.Timestamp(times[i]).hour % 3 != 0:
            continue
        temp = np.flipud(t2m_arr[i])
        norm = np.clip((temp - g_min) / (g_max - g_min), 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)
        img  = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        ts_str = np.datetime_as_string(times[i], unit='h').replace(":", "-")
        img.save(os.path.join(heatmap_folder, f"t2m_{i:05d}_{ts_str}.png"))

    return g_min, g_max


# ==========================================
# 3. DATASET CLASS
# ==========================================
class HeatmapSeqFromPaths(Dataset):
    """
    Returns:
        seq : (input_len, 3, 11, 11)  — heatmap + sin(hour) + cos(hour)
        tgt : (1, 11, 11)             — absolute normalised target [0,1]
    """
    def __init__(self, tuples, transform=None):
        self.tuples    = tuples
        self.transform = transform

    def __len__(self):
        return len(self.tuples)

    def _load_img(self, p):
        img = Image.open(p).convert("L")
        if self.transform:
            return self.transform(img)
        return transforms.ToTensor()(img)

    @staticmethod
    def _time_encoding(ts) -> tuple:
        hour  = ts.hour + getattr(ts, 'minute', 0) / 60.0
        angle = 2.0 * math.pi * hour / 24.0
        return math.sin(angle), math.cos(angle)

    def __getitem__(self, idx):
        seq_paths, tgt_path, seq_timestamps = self.tuples[idx]

        frames = [self._load_img(p) for p in seq_paths]
        tgt    = self._load_img(tgt_path)

        seq_3ch = []
        for frame, ts in zip(frames, seq_timestamps):
            sin_val, cos_val = self._time_encoding(ts)
            sin_map = torch.full_like(frame, sin_val)
            cos_map = torch.full_like(frame, cos_val)
            seq_3ch.append(torch.cat([frame, sin_map, cos_map], dim=0))

        seq = torch.stack(seq_3ch, dim=0)
        return seq, tgt


# ==========================================
# 4. MODEL DEFINITION
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout_p=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad
        )
        self.dropout_p = dropout_p
        num_groups = max(g for g in range(1, 9) if hidden_dim % g == 0)
        self.gn = nn.GroupNorm(num_groups, hidden_dim)

    def forward(self, x, hidden):
        h, c   = hidden
        comb   = torch.cat([x, h], dim=1)
        conv   = self.conv(comb)
        ci, cf, co, cg = torch.split(conv, self.hidden_dim, dim=1)
        i      = torch.sigmoid(ci)
        f      = torch.sigmoid(cf)
        o      = torch.sigmoid(co)
        g      = torch.tanh(cg)
        cnext  = f * c + i * g
        hnext  = o * torch.tanh(cnext)
        hnext  = self.gn(hnext)
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
    def __init__(self,
                 in_channels: int  = 3,
                 hidden_dims       = (32, 64, 64),
                 kernel_size: int  = 3,
                 dropout_p: float  = 0.05):
        super().__init__()

        in_dims = [in_channels] + list(hidden_dims[:-1])

        self.layers = nn.ModuleList([
            ConvLSTMCell(in_dims[i], hidden_dims[i], kernel_size, dropout_p)
            for i in range(len(hidden_dims))
        ])

        self.res_projs = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            in_d  = in_dims[i]
            hid_d = hidden_dims[i]
            self.res_projs.append(
                nn.Conv2d(in_d, hid_d, 1) if in_d != hid_d else nn.Identity()
            )

        final_h = hidden_dims[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_h, final_h, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_h, final_h, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_h, 1, 1),
        )

    def forward(self, x):
        B, T, Ch, H, W = x.size()
        hiddens = [l.init_hidden(B, (H, W), x.device) for l in self.layers]

        for t in range(T):
            inp = x[:, t]
            for li, layer in enumerate(self.layers):
                h, c         = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + self.res_projs[li - 1](inp)
                hiddens[li] = (hnext, cnext)
                inp = hnext

        return torch.sigmoid(self.refine(inp))


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    # ---- 1. Data & Stats ----
    g_min, g_max = ensure_data_ready()
    temp_range   = g_max - g_min
    print(f"📉 Denormalisation Key: Min={g_min:.2f}K, Range={temp_range:.2f}K")

    # ---- 2. Gather Files ----
    pngs  = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    dt_re = re.compile(r'.*?_(\d{4}-\d{2}-\d{2}T\d{2})')

    file_times = []
    for p in pngs:
        m = dt_re.search(os.path.basename(p))
        if m:
            ts = np.datetime64(m.group(1)).astype('datetime64[m]').astype(object)
            file_times.append((p, ts))

    file_times.sort(key=lambda x: x[1])
    paths      = [x[0] for x in file_times]
    years      = [x[1].year for x in file_times]
    timestamps = [x[1] for x in file_times]

    # ---- 3. Build sequences ----
    EXPECTED_INTERVAL = timedelta(hours=3)
    samples_by_year   = {}

    total_window = input_len + target_offset
    for i in range(len(paths) - total_window + 1):
        all_idxs = list(range(i, i + total_window))
        if any(
            timestamps[all_idxs[k + 1]] - timestamps[all_idxs[k]] != EXPECTED_INTERVAL
            for k in range(len(all_idxs) - 1)
        ):
            continue
        y_set = {years[j] for j in all_idxs}
        if len(y_set) != 1:
            continue
        seq_idxs = all_idxs[:input_len]
        targ_idx = all_idxs[input_len + target_offset - 1]
        yr       = years[targ_idx]
        samples_by_year.setdefault(yr, []).append((seq_idxs, targ_idx))

    train_years = [2020, 2021, 2022, 2023]
    test_year   = 2024

    train_samples = []
    for y in train_years:
        train_samples.extend(samples_by_year.get(y, []))

    test_tuples = [
        (
            [paths[idx]      for idx in s],
            paths[t],
            [timestamps[idx] for idx in s],
        )
        for s, t in samples_by_year.get(test_year, [])
    ]
    train_tuples = [
        (
            [paths[idx]      for idx in s],
            paths[t],
            [timestamps[idx] for idx in s],
        )
        for s, t in train_samples
    ]

    print(f"Train Samples: {len(train_tuples)} | Test Samples: {len(test_tuples)}")

    # ---- 4. Datasets & Loaders ----
    transform = transforms.Compose([transforms.ToTensor()])

    full_train_ds = HeatmapSeqFromPaths(train_tuples, transform=transform)
    test_ds       = HeatmapSeqFromPaths(test_tuples,  transform=transform)

    n_total = len(full_train_ds)
    n_train = int(0.8 * n_total)

    train_ds = Subset(full_train_ds, list(range(0, n_train)))
    val_ds   = Subset(full_train_ds, list(range(n_train, n_total)))

    print(f"Sequential Split: {len(train_ds)} Training / {len(val_ds)} Validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---- 5. Model ----
    model = ResidualConvLSTMWithRefine(
        in_channels = IN_CHANNELS,
        hidden_dims = HIDDEN_DIMS,
        kernel_size = kernel_size,
        dropout_p   = dropout_p,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model parameters: {total_params:,}")

    if os.path.exists(losses_path):
        with open(losses_path, 'r') as _f:
            _d = json.load(_f)
            train_losses = _d.get('train', [])
            val_losses   = _d.get('val',   [])
        print(f"📂 Loaded {len(train_losses)} epochs of loss history")
    else:
        train_losses = []
        val_losses   = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler           = GradScaler('cuda')
    best_val         = float('inf')
    patience         = 8
    patience_counter = 0
    ema_val          = None
    ema_alpha        = 0.3
    start_epoch      = 1
    training_done    = False
    criterion        = nn.L1Loss()

    if os.path.exists(checkpoint_path):
        print("🔄 Attempting to resume from checkpoint...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch      = ckpt['epoch'] + 1
            best_val         = ckpt['best_val']
            patience_counter = ckpt['patience_counter']
            ema_val          = ckpt['ema_val']
            print(f"✅ Resumed from epoch {ckpt['epoch']} | Best val: {best_val:.6f}")
            if start_epoch > num_epochs:
                print("✅ Training already complete.")
                training_done = True
        except (RuntimeError, KeyError) as e:
            print(f"⚠️  Checkpoint incompatible ({e}). Starting fresh.")
            os.remove(checkpoint_path)
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            train_losses = []
            val_losses   = []

    if not training_done and os.path.exists(model_save_path) and start_epoch == 1:
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"⚡ Loaded saved model. Skipping training.")
            training_done = True
        except RuntimeError as e:
            print(f"⚠️  Saved model incompatible ({e}). Starting fresh.")
            os.remove(model_save_path)

    # ---- 6. Training ----
    if not training_done:
        print(f"🚀 {'Starting' if start_epoch == 1 else 'Resuming'} "
              f"Training from epoch {start_epoch}...")

        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            train_loss = 0.0

            for seq, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
                seq, tgt = seq.to(device), tgt.to(device)
                optimizer.zero_grad()
                with autocast('cuda'):
                    pred = model(seq)
                    loss = criterion(pred, tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, tgt in val_loader:
                    seq, tgt = seq.to(device), tgt.to(device)
                    with autocast('cuda'):
                        pred     = model(seq)
                        val_loss += criterion(pred, tgt).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            ema_val = (avg_val if ema_val is None
                       else ema_alpha * avg_val + (1 - ema_alpha) * ema_val)
            scheduler.step(ema_val)

            if ema_val < best_val:
                best_val         = ema_val
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print("🔥 Saved Best Model")
            else:
                patience_counter += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f} | "
                f"LR {current_lr:.2e} | Patience {patience_counter}/{patience}"
            )

            torch.save({
                'epoch':            epoch,
                'model':            model.state_dict(),
                'optimizer':        optimizer.state_dict(),
                'scheduler':        scheduler.state_dict(),
                'scaler':           scaler.state_dict(),
                'best_val':         best_val,
                'patience_counter': patience_counter,
                'ema_val':          ema_val,
            }, checkpoint_path)

            with open(losses_path, 'w') as _f:
                json.dump({'train': train_losses, 'val': val_losses}, _f)

            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                model.load_state_dict(
                    torch.load(model_save_path, map_location=device))
                break

        if os.path.exists(model_save_path):
            model.load_state_dict(
                torch.load(model_save_path, map_location=device))

    if train_losses:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses,
                 label='Train Loss', color='tab:blue', marker='o', markersize=4)
        plt.plot(range(1, len(val_losses) + 1), val_losses,
                 label='Val Loss', color='tab:orange', marker='o', markersize=4)
        plt.title("Training vs Validation Loss (L1)")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_out, "train_val_loss.png"), dpi=200)
        plt.close()
        print("📈 Saved: train_val_loss.png")

    # ---- 7. Evaluation on 2024 ----
    print("📊 Evaluating on 2024 test set...")

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("✅ Best model weights loaded for evaluation.")
    model.eval()

    # Land mask
    if os.path.exists(mask_path):
        land_mask_raw = np.load(mask_path)
        if land_mask_raw.shape != (img_size, img_size):
            _mimg     = Image.fromarray(
                land_mask_raw.astype(np.uint8) * 255).convert("L")
            _mimg     = _mimg.resize((img_size, img_size), Image.NEAREST)
            land_mask = np.array(_mimg) > 0
        else:
            land_mask = land_mask_raw
        print(f"✅ Land mask loaded: {land_mask.sum()} / {land_mask.size} land pixels")
    else:
        land_mask = None
        print("⚠️ No land mask found — metrics over all pixels")

    # ADDED: load grid coordinates for axis ticks (fallback to pixel indices)
    if os.path.exists(coords_path):
        with open(coords_path) as _cf:
            _coords  = json.load(_cf)
            grid_lats = np.array(_coords['lats'])
            grid_lons = np.array(_coords['lons'])
        print(f"✅ Grid coords loaded: lat {grid_lats[0]:.2f}→{grid_lats[-1]:.2f}, "
              f"lon {grid_lons[0]:.2f}→{grid_lons[-1]:.2f}")
    else:
        grid_lats = None
        grid_lons = None
        print("⚠️ No grid coords file — axis ticks will be pixel indices")

    mae_scores, rmse_scores, ssim_scores = [], [], []
    dates = []

    def denorm(arr):
        return arr * temp_range + g_min

    with torch.no_grad():
        for i, (seq, tgt) in enumerate(
                tqdm(test_loader, desc="Evaluating")):

            with autocast('cuda'):
                pred_t = model(seq.to(device)).cpu().float()

            pred_np = pred_t.numpy().squeeze()
            tgt_np  = tgt.numpy().squeeze()

            pred_np = np.nan_to_num(pred_np, nan=0.0)
            tgt_np  = np.nan_to_num(tgt_np,  nan=0.0)

            pred_abs = denorm(pred_np)
            tgt_abs  = denorm(tgt_np)

            if land_mask is not None:
                pred_land = pred_abs[land_mask]
                tgt_land  = tgt_abs[land_mask]
            else:
                pred_land = pred_abs.flatten()
                tgt_land  = tgt_abs.flatten()

            diff = pred_land - tgt_land
            mae  = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))

            pred_ssim = pred_np.copy()
            tgt_ssim  = tgt_np.copy()
            if land_mask is not None:
                pred_ssim[~land_mask] = 0.0
                tgt_ssim[~land_mask]  = 0.0
            try:
                s = float(ssim_metric(tgt_ssim, pred_ssim,
                                      data_range=1.0, win_size=7))
            except Exception:
                s = 0.0

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            ssim_scores.append(s)

            fn = os.path.basename(test_tuples[i][1])
            m  = dt_re.search(fn)
            dates.append(np.datetime64(m.group(1)) if m else np.datetime64('NaT'))

            # ============================================================
            # CHANGED: first sample comparison plot — publication style
            # ============================================================
            if i == 0:
                # --- Metrics for the metrics box ---
                mse_val  = float(np.mean(diff ** 2))

                # --- Timestamps for the super-title ---
                # seq_timestamps for sample 0 are in test_tuples[0][2]
                seq_ts_0 = test_tuples[0][2]          # list of Python datetimes
                tgt_ts_0 = dates[0]                   # np.datetime64

                suptitle_str = _build_suptitle(seq_ts_0, tgt_ts_0)

                # --- Shared colour scale for Actual / Predicted ---
                # Use combined min/max so both panels are on the same scale
                vmin_shared = min(tgt_abs.min(), pred_abs.min())
                vmax_shared = max(tgt_abs.max(), pred_abs.max())

                # --- Error map (symmetric around 0) ---
                error_abs = tgt_abs - pred_abs
                err_absmax = max(abs(error_abs.min()), abs(error_abs.max()))
                err_absmax = err_absmax if err_absmax > 0 else 1.0   # guard

                # --- Axis extent for imshow (uses real lat/lon if available) ---
                # extent = [left, right, bottom, top]  (lon_min, lon_max, lat_min, lat_max)
                if grid_lats is not None and grid_lons is not None:
                    half_dlon = abs(grid_lons[1] - grid_lons[0]) / 2.0
                    half_dlat = abs(grid_lats[1] - grid_lats[0]) / 2.0
                    img_extent = [
                        grid_lons[0]  - half_dlon,
                        grid_lons[-1] + half_dlon,
                        grid_lats[-1] - half_dlat,   # bottom (southernmost after flipud)
                        grid_lats[0]  + half_dlat,   # top    (northernmost after flipud)
                    ]
                    lon_ticks = np.round(np.linspace(
                        grid_lons[0], grid_lons[-1], min(5, len(grid_lons))), 1)
                    lat_ticks = np.round(np.linspace(
                        grid_lats[-1], grid_lats[0], min(5, len(grid_lats))), 1)
                else:
                    img_extent = None
                    lon_ticks  = None
                    lat_ticks  = None

                # --- Build figure ---
                fig, axes = plt.subplots(
                    1, 3,
                    figsize      = (14, 5),
                    constrained_layout = False,
                )
                fig.subplots_adjust(
                    left=0.06, right=0.97,
                    top=0.82,  bottom=0.18,
                    wspace=0.38,
                )

                # ---- Panel helper ----
                def _add_panel(ax, data, title, cmap,
                               vmin, vmax, cbar_label):
                    """Draw one heatmap panel with geo axes and colorbar."""
                    kwargs = dict(
                        cmap          = cmap,
                        vmin          = vmin,
                        vmax          = vmax,
                        interpolation = 'nearest',
                        aspect        = 'auto',
                    )
                    if img_extent is not None:
                        kwargs['extent'] = img_extent
                        kwargs['origin'] = 'upper'
                    im = ax.imshow(data, **kwargs)

                    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

                    # Axis labels
                    ax.set_xlabel("Longitude →", fontsize=9)
                    ax.set_ylabel("Latitude ↑",  fontsize=9)

                    # Tick labels
                    if lon_ticks is not None:
                        ax.set_xticks(lon_ticks)
                        ax.set_xticklabels(
                            [f"{v:.1f}" for v in lon_ticks], fontsize=7.5)
                    if lat_ticks is not None:
                        ax.set_yticks(lat_ticks)
                        ax.set_yticklabels(
                            [f"{v:.1f}" for v in lat_ticks], fontsize=7.5)

                    ax.tick_params(axis='both', which='both',
                                   length=3, width=0.6)

                    # Colorbar
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(cbar_label, fontsize=8)
                    cbar.ax.tick_params(labelsize=7)

                # ---- Draw three panels ----
                _add_panel(
                    axes[0], tgt_abs,
                    title      = "Actual",
                    cmap       = 'gray',
                    vmin       = vmin_shared,
                    vmax       = vmax_shared,
                    cbar_label = "2m Temperature (K)",
                )
                _add_panel(
                    axes[1], pred_abs,
                    title      = "Predicted",
                    cmap       = 'gray',
                    vmin       = vmin_shared,
                    vmax       = vmax_shared,
                    cbar_label = "2m Temperature (K)",
                )
                _add_panel(
                    axes[2], error_abs,
                    title      = "Error = Actual - Predicted",
                    cmap       = 'gray',
                    vmin       = -err_absmax,
                    vmax       =  err_absmax,
                    cbar_label = "Error (K)",
                )

                # ---- Super-title ----
                fig.suptitle(
                    suptitle_str,
                    fontsize   = 11,
                    fontweight = 'bold',
                    y          = 0.96,
                )

                # ---- Metrics box below panels ----
                metrics_str = (
                    f"MSE: {mse_val:.4f}    "
                    f"MAE: {mae:.4f}    "
                    f"RMSE: {rmse:.4f}    "
                    f"SSIM: {s:.4f}"
                )
                fig.text(
                    0.5, 0.04,
                    metrics_str,
                    ha        = 'center',
                    va        = 'center',
                    fontsize  = 9,
                    fontfamily= 'monospace',
                    bbox      = dict(
                        boxstyle   = 'round,pad=0.5',
                        facecolor  = 'white',
                        edgecolor  = '#444444',
                        linewidth  = 1.0,
                        alpha      = 0.9,
                    ),
                )

                # ---- Save ----
                out_path = os.path.join(eval_out, "first_sample_comparison.png")
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"🖼️  Saved: first_sample_comparison.png")
            # ============================================================
            # END of first sample plot block
            # ============================================================

    # ---- 8. Save Metrics CSV ----
    df = pd.DataFrame({
        'time':     dates,
        'mae_abs':  mae_scores,
        'rmse_abs': rmse_scores,
        'ssim':     ssim_scores,
    })
    csv_path = os.path.join(eval_out, "absolute_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: absolute_results.csv  ({len(df)} rows)")

    # ---- 9. Metric Trend Plots ----
    def plot_metric(values, ylabel, title, color, filename):
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(dates, values, marker='.', linewidth=0.6,
                markersize=2.5, color=color, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (2024)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_out, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {filename}")

    plot_metric(ssim_scores,  "SSIM",         "SSIM 2024",
                "tab:blue",   "SSIM_2024.png")
    plot_metric(mae_scores,   "MAE (Kelvin)",  "Absolute MAE 2024",
                "tab:orange", "Absolute_MAE_2024.png")
    plot_metric(rmse_scores,  "RMSE (Kelvin)", "Absolute RMSE 2024",
                "tab:red",    "Absolute_RMSE_2024.png")

    # ---- 10. Summary Stats ----
    print("\n" + "=" * 55)
    print(f"  2024 Test-Set Summary ({len(mae_scores)} samples)")
    print("=" * 55)
    print(f"  Mean SSIM          : {np.mean(ssim_scores):.4f}")
    print(f"  Mean MAE  (Kelvin) : {np.mean(mae_scores):.4f}")
    print(f"  Mean RMSE (Kelvin) : {np.mean(rmse_scores):.4f}")
    print("=" * 55)
    print(f"\n🎉 All outputs saved in: {eval_out}")


if __name__ == '__main__':
    main()