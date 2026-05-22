import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
from pathlib import Path

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_metric
import xarray as xr

# ==========================================
# 1. CONFIGURATION
# ==========================================
base_dir    = "/kaggle/input/datasets/legion2022/uhi-dataset/Dataset"
script_dir  = "/kaggle/working"

heatmap_folder  = os.path.join(script_dir, "Heatmaps_t2m_gray_fast")
model_save_path = os.path.join(script_dir, "UNetConvLSTM_t2m_best.pth")
eval_out        = os.path.join(script_dir, "Eval_UNetConvLSTM_2024")
stats_path      = os.path.join(script_dir, "dataset_stats.json")
mask_path       = os.path.join(script_dir, "land_mask.npy")
losses_path     = os.path.join(script_dir, "train_val_losses.json")
checkpoint_path = os.path.join(script_dir, "training_checkpoint.pth")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

# Hyperparameters
input_len  = 8
img_size   = 11
batch_size = 8
num_epochs = 60
lr         = 5e-5
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Device: {device}")
print(f"✅ Dataset Path: {base_dir}")

# ==========================================
# 2. DATA GENERATION & STATS
# ==========================================
def ensure_data_ready():
    """
    Ensures PNGs + land mask exist and global min/max stats are available.
    """
    # --- Stats ---
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
                da = ds['t2m'].isel(latitude=slice(0, 11), longitude=slice(0, 11))
                all_vals.append(da.values.flatten())
            except: pass

        full_arr = np.concatenate(all_vals)
        g_min = float(np.nanmin(full_arr))
        g_max = float(np.nanmax(full_arr))

        with open(stats_path, 'w') as f:
            json.dump({"min": g_min, "max": g_max}, f)
        print(f"✅ Stats saved: Min={g_min:.2f}K, Max={g_max:.2f}K")

    with open(stats_path, 'r') as f:
        stats = json.load(f)
        g_min, g_max = stats['min'], stats['max']

    # --- PNGs ---
    # Force fresh regeneration so no stale wrongly-oriented files remain
    import shutil
    if os.path.exists(heatmap_folder):
        shutil.rmtree(heatmap_folder)
    os.makedirs(heatmap_folder, exist_ok=True)
    if os.path.exists(mask_path):
        os.remove(mask_path)

    print("⚠️ Generating PNGs (fresh)...")

    years = ['2020', '2021', '2022', '2023', '2024']
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
            if 'latitude' in da.coords: da = da.sortby('latitude')
            if 'longitude' in da.coords: da = da.sortby('longitude')
            all_t2m.append(da)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    t2m_all = xr.concat(all_t2m, dim='time')
    t2m_sub = t2m_all.isel(latitude=slice(0, 11), longitude=slice(0, 11))
    t2m_arr = t2m_sub.values
    times   = t2m_sub.time.values

    # --- Land mask: True = land (non-NaN), False = sea ---
    land_mask_raw = None
    for i in range(len(t2m_arr)):
        if not np.all(np.isnan(t2m_arr[i])):
            land_mask_raw = ~np.isnan(t2m_arr[i])   # shape: (H, W)
            break
    if land_mask_raw is not None:
        # Match PNG orientation: vertical flip then horizontal flip
        land_mask_raw = np.flipud(land_mask_raw)
        np.save(mask_path, land_mask_raw)

    # --- Save PNGs (3-hourly only) ---
    for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
        if pd.Timestamp(times[i]).hour % 3 != 0:
            continue
        temp = t2m_arr[i]
        norm = (temp - g_min) / (g_max - g_min)
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)   # sea pixels → 0 (black)
        # Vertical flip then horizontal flip for correct final orientation
        norm = np.flipud(norm)
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        ts_str = np.datetime_as_string(times[i], unit='h').replace(":", "-")
        img.save(os.path.join(heatmap_folder, f"t2m_{i:05d}_{ts_str}.png"))

    return g_min, g_max


# ==========================================
# 3. DATASET CLASS
# ==========================================
class HeatmapSeqFromPaths(Dataset):
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

    def _load(self, p):
        tensor = self._load_img(p)
        # Time-of-day encoding: sin/cos of hour as extra channels
        hour = 0
        m = re.search(r'T(\d{2})', os.path.basename(p))
        if m:
            hour = int(m.group(1))
        sin_ch = torch.full_like(tensor, np.sin(2 * np.pi * hour / 24))
        cos_ch = torch.full_like(tensor, np.cos(2 * np.pi * hour / 24))

        # Seasonal encoding: extract date from filename
        day_of_year = 1
        dm = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(p))
        if dm:
            day_of_year = datetime.strptime(dm.group(1), '%Y-%m-%d').timetuple().tm_yday
        sin_doy = torch.full_like(tensor, np.sin(2 * np.pi * day_of_year / 365))
        cos_doy = torch.full_like(tensor, np.cos(2 * np.pi * day_of_year / 365))

        return torch.cat([tensor, sin_ch, cos_ch, sin_doy, cos_doy], dim=0)

    def __getitem__(self, idx):
        seq_paths, tgt_path = self.tuples[idx]
        seq = torch.stack([self._load(p) for p in seq_paths], dim=0)
        tgt = self._load_img(tgt_path)

        # Extract target frame's time encoding
        hour = 0
        m = re.search(r'T(\d{2})', os.path.basename(tgt_path))
        if m:
            hour = int(m.group(1))
        day_of_year = 1
        dm = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(tgt_path))
        if dm:
            day_of_year = datetime.strptime(dm.group(1), '%Y-%m-%d').timetuple().tm_yday
        tgt_time = torch.tensor([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365)
        ], dtype=torch.float32)

        return seq, tgt, tgt_time



# ==========================================
# 4. MODEL DEFINITION (True U-Net + ConvLSTM)
# ==========================================

# Number of GroupNorm groups used throughout.
# Must divide every channel width: 32 ✓ 64 ✓ 128 ✓  (all multiples of 8)
_GN_GROUPS = 8


def _gn(num_ch: int) -> nn.GroupNorm:
    """
    Convenience factory: GroupNorm(8, num_ch).
    Keeps layer definitions concise and the group count in one place.
    """
    return nn.GroupNorm(_GN_GROUPS, num_ch)


class DoubleConv(nn.Module):
    """
    Two consecutive  Conv2d(3×3, pad=1) → GroupNorm → ReLU  blocks.
    Spatial dimensions are preserved (padding=1 keeps H×W constant).

    GroupNorm replaces BatchNorm2d:
      • At the 2×2 bottleneck with B=8, BatchNorm sees only 32 spatial
        positions per channel — too few for reliable running-mean estimates.
      • GroupNorm(8, ch) normalises over (ch//8) channels × H × W within
        each sample, producing stable statistics at any spatial size.
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),  # (B, mid_ch, H, W)
            _gn(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),  # (B, out_ch, H, W)
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_op(x)


class Down(nn.Module):
    """
    MaxPool2d(2) then DoubleConv.  Halves both spatial dimensions.

    11×11 → MaxPool → 5×5  (floor division)
     5×5  → MaxPool → 2×2
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),             # (B, in_ch, H//2, W//2)
            DoubleConv(in_ch, out_ch),   # (B, out_ch, H//2, W//2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Bilinear upsample (×2) → pad to match skip → concat → DoubleConv.

    align_corners=False (changed from True):
      Aligns pixel centres rather than corner pixels.  For odd→odd upsampling
      (2→5, 5→11) the True mode places samples slightly outside the valid grid
      on one side, creating asymmetric boundary values that raise SSIM variance
      near edges.  False produces symmetric, uniform boundary sampling.

    F.pad correction is still needed because floor(2×2)=4 ≠ 5 and
    floor(2×5)=10 ≠ 11 — the padding restores the 1-pixel shortfall on each
    odd dimension, keeping the skip concatenation shape-safe.

    in_ch = upsampled_channels + skip_channels  (caller's responsibility).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: (B, C1, H1, W1)  — tensor to upsample
        # x2: (B, C2, H2, W2)  — encoder skip connection  (H2 ≥ H1*2)
        x1 = self.up(x1)                                          # (B, C1, H1*2, W1*2)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])          # (B, C1, H2, W2)
        return self.conv(torch.cat([x2, x1], dim=1))               # (B, out_ch, H2, W2)


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.  All four gates are fused into one Conv2d for
    efficiency.  No normalisation inside the cell — applying norm to the
    hidden state between timesteps would corrupt the temporal memory by
    re-centring the state distribution after every step.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,   # concat(x_t, h_{t-1})
            4 * hidden_dim,           # i, f, o, g gates
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # x: (B, input_dim, H, W)
        # h: (B, hidden_dim, H, W)
        # c: (B, hidden_dim, H, W)
        gates = self.conv(torch.cat([x, h], dim=1))        # (B, 4*hidden_dim, H, W)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next                              # each (B, hidden_dim, H, W)


class ConvLSTM(nn.Module):
    """
    Unrolls ConvLSTMCell over the full input sequence.
    Applied only at the bottleneck (2×2) — cheap yet captures all T=8
    timesteps of temporal dynamics before decoding.
    Returns only the final hidden state h_T.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell       = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq: (T, B, C, H, W)
        T, B, C, H, W = input_seq.shape
        # Initialise states in the same dtype as input (fp16 under AMP)
        h = torch.zeros(B, self.hidden_dim, H, W,
                        device=input_seq.device, dtype=input_seq.dtype)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(input_seq[t], h, c)
        return h  # (B, hidden_dim, H, W)


class UNetConvLSTM(nn.Module):
    """
    True U-Net encoder-decoder with ConvLSTM at the bottleneck.

    Spatial flow for 11×11 inputs
    ─────────────────────────────
    Encoder (per timestep t):
      inc   : (B,  5, 11, 11) → (B,  32, 11, 11)   ← skip1
      down1 : (B, 32, 11, 11) → (B,  64,  5,  5)   ← skip2
      down2 : (B, 64,  5,  5) → (B, 128,  2,  2)   ← bottleneck

    Temporal (all T steps):
      ConvLSTM on bottleneck  → (B, 128,  2,  2)
      + time conditioning     → (B, 128,  2,  2)   (broadcast add)

    Skip aggregation (temporal ConvLSTM):
      skip_lstm1 on (T,B,32,11,11) → final h: (B, 32, 11, 11)   skip1_temporal
      skip_lstm2 on (T,B,64, 5, 5) → final h: (B, 64,  5,  5)   skip2_temporal

    Decoder:
      up1 : upsample(128) → cat(skip2, 64) = 192 → (B,  64,  5,  5)
      up2 : upsample(64)  → cat(skip1, 32) =  96 → (B,  32, 11, 11)
      outc: (B, 1, 11, 11)

    Residual:
      return out + last input frame temperature channel
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.inc   = DoubleConv(in_channels, 32)   # (B, 32, 11, 11)
        self.down1 = Down(32,  64)                  # (B, 64,  5,  5)
        self.down2 = Down(64, 128)                  # (B,128,  2,  2)

        # ── Bottleneck temporal module ────────────────────────────────────────
        self.convlstm = ConvLSTM(input_dim=128, hidden_dim=128, kernel_size=3)

        # ── Target-time conditioning ──────────────────────────────────────────
        # Linear(4→128) is correct here because the spatial extent (2×2) varies
        # across model versions; broadcasting a (B,128,1,1) vector avoids tying
        # the projection to a fixed spatial size.
        self.time_proj = nn.Linear(4, 128)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up1 = Up(in_ch=128 + 64, out_ch=64)   # 2→5,  concat skip2
        self.up2 = Up(in_ch=64  + 32, out_ch=32)   # 5→11, concat skip1

        # ── Output head ──────────────────────────────────────────────────────
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

        # ── NEW: Temporal ConvLSTM modules for skip connections ───────────────────
        #
        # Why two separate ConvLSTMs instead of one shared one?
        #   skip1 operates at full resolution (11×11, 32 ch) — fine spatial detail
        #   skip2 operates at half resolution ( 5× 5, 64 ch) — mid-level structure
        #   Their temporal dynamics differ: fine-scale urban/water boundaries evolve
        #   faster than broad structural temperature gradients.  Separate LSTMs let
        #   each resolution level learn its own temporal integration strategy.
        #
        # Why kernel_size=3 for 11×11 but also for 5×5?
        #   At 5×5 a kernel_size=3 still covers a receptive field of 60% of the grid,
        #   which is sufficient.  Dropping to kernel_size=1 would lose spatial context;
        #   going larger than 3 at 5×5 brings no gain and increases parameter count.
        #
        # hidden_dim == input_dim keeps skip tensor shapes identical to the old
        #   skip1_avg / skip2_avg shapes, so up1 and up2 receive exactly the same
        #   channel counts as before — no decoder changes needed.
        #
        self.skip_lstm1 = ConvLSTM(input_dim=32, hidden_dim=32, kernel_size=3)
        # input : (T, B,  32, 11, 11)   — one frame per encoder timestep
        # output: (   B,  32, 11, 11)   — final hidden state h_T
        # param count: ConvLSTMCell Conv2d: (32+32) × 4×32 × 3×3 = 73,728

        self.skip_lstm2 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3)
        # input : (T, B,  64,  5,  5)
        # output: (   B,  64,  5,  5)
        # param count: ConvLSTMCell Conv2d: (64+64) × 4×64 × 3×3 = 294,912
        #
        # Total new parameters: ~369K — lightweight relative to the full model.
        # ── END of new __init__ additions ─────────────────────────────────────────

    def forward(self, x: torch.Tensor, target_time: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, 5, 11, 11)
        B, T, C, H, W = x.shape

        skip1_list, skip2_list, bot_list = [], [], []

        for t in range(T):
            frame = x[:, t]                          # (B,  5, 11, 11)
            s1    = self.inc(frame)                  # (B, 32, 11, 11)
            s2    = self.down1(s1)                   # (B, 64,  5,  5)
            b     = self.down2(s2)                   # (B,128,  2,  2)
            skip1_list.append(s1)
            skip2_list.append(s2)
            bot_list.append(b)

        # ── NEW: Temporal ConvLSTM skip fusion ────────────────────────────────
        #
        # Why this is better than mean() for UHI forecasting:
        #   mean() collapses the 8-frame sequence into a single flat average.
        #   It erases: (a) diurnal heating/cooling trajectories at coastlines,
        #   (b) the order in which urban hotspots intensify across the day,
        #   (c) delayed thermal release from concrete after peak solar load.
        #   A ConvLSTM instead reads frames causally — earlier frames set
        #   the hidden state that biases how later frames are interpreted,
        #   so the decoder receives a spatially-resolved summary of how each
        #   pixel evolved over T=8 steps rather than a memoryless average.
        #
        # Step 1 — stack lists into a (T, B, C, H, W) sequence tensor
        skip1_seq = torch.stack(skip1_list, dim=0)   # (T, B,  32, 11, 11)
        skip2_seq = torch.stack(skip2_list, dim=0)   # (T, B,  64,  5,  5)

        # Step 2 — run each skip sequence through its ConvLSTM
        #   self.skip_lstm1 / skip_lstm2 are the two new modules added in __init__
        #   .forward() internally loops over T, updates (h, c) each step,
        #   and returns only the final hidden state h_T — same as bottleneck LSTM.
        skip1_temporal = self.skip_lstm1(skip1_seq)   # (B,  32, 11, 11)
        skip2_temporal = self.skip_lstm2(skip2_seq)   # (B,  64,  5,  5)
        #
        # Shapes are identical to the old skip1_avg / skip2_avg, so up1 and up2
        # receive exactly (B,64,5,5) and (B,32,11,11) as before — no decoder edits.
        # ── END of new skip fusion block ──────────────────────────────────────

        # ConvLSTM on bottleneck sequence — all temporal dynamics condensed here
        h = self.convlstm(torch.stack(bot_list, 0))      # (B,128,  2,  2)

        # Inject target-time encoding (sin/cos of hour + day-of-year)
        if target_time is not None:
            time_feat = self.time_proj(target_time)       # (B, 128)
            h = h + time_feat.view(B, 128, 1, 1)          # (B,128,  2,  2)

        d   = self.up1(h,          skip2_temporal)        # (B, 64,  5,  5)
        d   = self.up2(d,          skip1_temporal)        # (B, 32, 11, 11)
        out = self.outc(d)                                # (B,  1, 11, 11)

        # Residual: predict the delta from the last observed temperature frame
        last_frame = x[:, -1, 0:1, :, :]                 # (B,  1, 11, 11)
        return out + last_frame                           # (B,  1, 11, 11)


# ==========================================
# LOSS FUNCTIONS
# ==========================================

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss safe for small spatial grids (window_size=3).

    No changes to the SSIM formula itself.  Structure-sensitivity is improved
    by running on lightly smoothed (single AvgPool pass) rather than heavily
    smoothed inputs — handled in CombinedLoss.
    """
    def __init__(self, window_size: int = 3):
        super().__init__()
        self.ws = window_size
        self.C1 = 0.01 ** 2   # luminance stabiliser
        self.C2 = 0.03 ** 2   # contrast/structure stabiliser

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p   = self.ws
        pad = p // 2
        mu1     = F.avg_pool2d(pred,   p, stride=1, padding=pad)
        mu2     = F.avg_pool2d(target, p, stride=1, padding=pad)
        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        s1  = F.avg_pool2d(pred   * pred,   p, stride=1, padding=pad) - mu1_sq
        s2  = F.avg_pool2d(target * target, p, stride=1, padding=pad) - mu2_sq
        s12 = F.avg_pool2d(pred   * target, p, stride=1, padding=pad) - mu1_mu2
        num = (2 * mu1_mu2 + self.C1) * (2 * s12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2)
        return 1.0 - (num / den).mean()


class CombinedLoss(nn.Module):
    """
    Combined reconstruction loss.

    Formula
    ───────
      Loss = 0.60 × Huber  +  0.25 × SSIM  +  0.15 × GradientL1

    Why each component
    ──────────────────
    Huber (0.60):
      Robust pixel-level reconstruction.  Quadratic near zero (like MSE)
      but linear for large errors, so outlier temperature spikes don't
      dominate.  Kept as the dominant term for training stability.

    SSIM (0.25):
      Structural similarity on a single lightly-blurred copy of each map.
      Single AvgPool pass (kernel=3) is the minimum smoothing needed to
      avoid NaN from near-zero local variance on 11×11 grids.  One pass
      (vs. a second heavier blur) preserves boundary contrast in the SSIM
      gradient, making the loss more sensitive to structural errors and
      reducing dips on difficult samples with sharp UHI edges.
      Weight reduced from 0.30 to 0.25 to make room for the gradient term.

    GradientL1 (0.15):
      L1 between finite-difference gradients (horizontal dx, vertical dy)
      of prediction and target.  Penalises blurry predictions that achieve
      low pixel error while washing out local contrast — the primary cause
      of SSIM variance on high-gradient samples (coastline, urban heat
      boundaries).  Weight 0.15 is deliberately modest: gradient magnitudes
      on normalised 11×11 maps are small, so the term usefully shapes the
      loss landscape without competing with Huber for dominant influence.

    The existing TV regularisation term in the training loop
    (0.02 × mean_abs_horizontal_diff) is complementary — it penalises
    unnecessary oscillations in the prediction while the gradient loss here
    specifically penalises *missing* structure in the target.
    """

    def __init__(self, grad_weight: float = 0.15):
        super().__init__()
        self.huber       = nn.HuberLoss()
        self.ssim        = SSIMLoss(window_size=3)
        # Single light-blur pass for SSIM stability; not removed entirely
        # because direct SSIM on 11×11 grids can produce near-zero local
        # variance in flat sea regions, destabilising the denominator.
        self.blur        = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.grad_weight = grad_weight

    @staticmethod
    def _image_gradients(t: torch.Tensor):
        """
        Finite differences along W (horizontal) and H (vertical).

        t:  (B, 1, H, W)
        dx: (B, 1, H, W-1)  — difference between adjacent columns
        dy: (B, 1, H-1, W)  — difference between adjacent rows

        No padding: L1 is computed only where both neighbours exist,
        avoiding artefacts from zero-padded boundary derivatives.
        """
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]    # (B, 1, H, W-1)
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]    # (B, 1, H-1, W)
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 1, H, W)

        # ── Pixel reconstruction ──────────────────────────────────────────────
        huber_loss = self.huber(pred, target)

        # ── Structure similarity ──────────────────────────────────────────────
        # Single light blur: stabilises SSIM on flat sea pixels while keeping
        # boundary contrast present in the SSIM gradient signal.
        pred_s    = self.blur(pred)
        target_s  = self.blur(target)
        ssim_loss = self.ssim(pred_s, target_s)

        # ── Edge-aware gradient loss ──────────────────────────────────────────
        pred_dx,   pred_dy   = self._image_gradients(pred)
        target_dx, target_dy = self._image_gradients(target)
        grad_loss = (
            F.l1_loss(pred_dx, target_dx) +   # horizontal edge agreement
            F.l1_loss(pred_dy, target_dy)      # vertical   edge agreement
        )

        return (
            0.60 * huber_loss
          + 0.25 * ssim_loss
          + self.grad_weight * grad_loss
        )


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    # 1. Data & Stats
    g_min, g_max = ensure_data_ready()
    temp_range   = g_max - g_min
    print(f"📉 Denormalization Key: Min={g_min:.2f}K, Range={temp_range:.2f}K")

    # 2. Gather Files
    pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    if len(pngs) == 0:
        raise RuntimeError(f"❌ No PNG files found in '{heatmap_folder}'. Regenerate heatmaps.")
    print(f"✅ Found {len(pngs)} PNG files. Sample: {os.path.basename(pngs[0])}")

    dt_re = re.compile(r'.*?_(\d{4}-\d{2}-\d{2}T\d{2})')

    file_times = []
    for p in pngs:
        m = dt_re.search(os.path.basename(p))
        if m:
            ts = np.datetime64(m.group(1)).astype('datetime64[m]').astype(object)
            file_times.append((p, ts))

    if len(file_times) == 0:
        raise RuntimeError(
            f"❌ Regex matched 0 files. Sample filename: {os.path.basename(pngs[0])}"
        )
    print(f"✅ Parsed timestamps for {len(file_times)} files.")

    file_times.sort(key=lambda x: x[1])
    paths      = [x[0] for x in file_times]
    years      = [x[1].year for x in file_times]
    timestamps = [x[1] for x in file_times]

    from collections import Counter
    print(f"📅 Year distribution: {dict(sorted(Counter(years).items()))}")

    # 3. Build sequences — only contiguous 3-hour blocks
    EXPECTED_INTERVAL = timedelta(hours=3)
    samples_by_year   = {}
    for i in range(len(paths) - input_len - 3):
        seq_idxs = list(range(i, i + input_len))
        targ_idx = i + input_len + 3
        if any(
            timestamps[j+1] - timestamps[j] != EXPECTED_INTERVAL
            for j in range(i, targ_idx)
        ):
            continue
        y_seq = {years[j] for j in range(i, targ_idx + 1)}
        if len(y_seq) == 1:
            yr = years[targ_idx]
            samples_by_year.setdefault(yr, []).append((seq_idxs, targ_idx))

    print(f"📦 Samples per year: { {k: len(v) for k, v in samples_by_year.items()} }")

    # 4. Split — 2020-2023 train/val, 2024 test
    train_years = [2020, 2021, 2022, 2023]
    test_year   = 2024

    train_samples = []
    for y in train_years:
        train_samples.extend(samples_by_year.get(y, []))

    if len(train_samples) == 0:
        raise RuntimeError(f"❌ No training samples for years {train_years}. Keys found: {list(samples_by_year.keys())}")

    train_tuples = [([paths[idx] for idx in s], paths[t]) for s, t in train_samples]
    test_tuples  = [([paths[idx] for idx in s], paths[t]) for s, t in samples_by_year.get(test_year, [])]

    print(f"Train Samples: {len(train_tuples)} | Test Samples: {len(test_tuples)}")

    # 5. Datasets & Loaders
    full_train_ds = HeatmapSeqFromPaths(train_tuples, transform=None)
    test_ds       = HeatmapSeqFromPaths(test_tuples,  transform=None)

    n_total = len(full_train_ds)
    n_train = int(0.8 * n_total)

    train_ds = Subset(full_train_ds, list(range(0, n_train)))
    val_ds   = Subset(full_train_ds, list(range(n_train, n_total)))

    print(f"Sequential Split: {len(train_ds)} Training / {len(val_ds)} Validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=2, pin_memory=True)

    # 6. Model
    model = UNetConvLSTM(in_channels=5).to(device)

    if os.path.exists(losses_path):
        import json as _json
        with open(losses_path, 'r') as _f:
            _d = _json.load(_f)
            train_losses   = _d.get('train',   [])
            val_losses     = _d.get('val',     [])
            ema_val_losses = _d.get('ema_val', [])
        print(f"📂 Loaded {len(train_losses)} epochs of loss history")
    else:
        train_losses   = []
        val_losses     = []
        ema_val_losses = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion        = CombinedLoss()
    scaler           = GradScaler()
    best_val         = float('inf')
    patience         = 8
    patience_counter = 0
    ema_val          = None
    ema_alpha        = 0.3
    start_epoch      = 1
    training_done    = False

    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch      = ckpt['epoch'] + 1
        best_val         = ckpt['best_val']
        patience_counter = ckpt['patience_counter']
        ema_val          = ckpt['ema_val']
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f"✅ Resumed from epoch {ckpt['epoch']} | Best val: {best_val:.6f}")
        if start_epoch > num_epochs:
            print("✅ Training already complete.")
            training_done = True
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
    elif os.path.exists(model_save_path):
        print(f"⚡ Loading model from {model_save_path} (no checkpoint, skipping training)")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        training_done = True

    if not training_done:
        print(f"🚀 {'Starting' if start_epoch == 1 else 'Resuming'} Training from epoch {start_epoch}...")
        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            train_loss = 0.0
            for seq, tgt, tgt_time in tqdm(train_loader, desc=f"Epoch {epoch}"):
                seq, tgt = seq.to(device), tgt.to(device)
                tgt_time  = tgt_time.to(device)
                optimizer.zero_grad()
                with autocast():
                    pred = model(seq, tgt_time)
                    loss = criterion(pred, tgt)
                    loss = loss + 0.02 * torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, tgt, tgt_time in val_loader:
                    with autocast():
                        seq_d    = seq.to(device)
                        tgt_time = tgt_time.to(device)
                        val_loss += criterion(model(seq_d, tgt_time), tgt.to(device)).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            ema_val = avg_val if ema_val is None else ema_alpha * avg_val + (1 - ema_alpha) * ema_val
            ema_val_losses.append(ema_val)

            if ema_val < best_val:
                best_val         = ema_val
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print("🔥 Saved Best Model")
            else:
                patience_counter += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f} | LR {current_lr:.2e} | Patience {patience_counter}/{patience}")

            torch.save({
                'epoch':            epoch,
                'model':            model.state_dict(),
                'optimizer':        optimizer.state_dict(),
                'scheduler':        scheduler.state_dict(),
                'scaler':           scaler.state_dict(),
                'best_val':         best_val,
                'patience_counter': patience_counter,
                'ema_val':          ema_val,
                'ema_val_losses':   ema_val_losses,
            }, checkpoint_path)

            import json as _json
            with open(losses_path, 'w') as _f:
                _json.dump({'train': train_losses, 'val': val_losses, 'ema_val': ema_val_losses}, _f)

            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                break

        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))

    if train_losses:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_losses)    + 1), train_losses,    label='Train Loss',     color='tab:blue',   marker='o')
        plt.plot(range(1, len(val_losses)      + 1), val_losses,      label='Val Loss (Raw)', color='tab:orange', marker='o', alpha=0.5)
        if ema_val_losses:
            plt.plot(range(1, len(ema_val_losses) + 1), ema_val_losses, label='Val Loss (EMA)', color='tab:green', linestyle='--', linewidth=2)
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Combined Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_out, "train_val_loss.png"))
        plt.close()
        print("📈 Train/Val loss plot saved.")

    # 7. Evaluation
    print("📊 Evaluating...")
    model.eval()
    mse_scores, mae_scores, rmse_scores, ssim_scores = [], [], [], []
    dates = []
    first_sample_data = {}   # will hold actual/pred/error for sample 0

    if os.path.exists(mask_path):
        land_mask = np.load(mask_path)
        print(f"✅ Land mask loaded: {land_mask.sum()} land pixels / {land_mask.size} total")
    else:
        land_mask = None
        print("⚠️ No land mask found — metrics over all pixels")

    with torch.no_grad():
        for i, (seq, tgt, tgt_time) in enumerate(tqdm(test_loader)):
            with autocast():
                seq_d    = seq.to(device)
                tgt_time = tgt_time.to(device)
                pred = model(seq_d, tgt_time).float().cpu().numpy().squeeze()
            tgt_np = tgt.numpy().squeeze()

            pred   = np.nan_to_num(pred,   nan=0.0)
            tgt_np = np.nan_to_num(tgt_np, nan=0.0)

            pred_abs = pred   * temp_range + g_min
            tgt_abs  = tgt_np * temp_range + g_min

            if land_mask is not None:
                pred_land = pred_abs[land_mask]
                tgt_land  = tgt_abs[land_mask]
            else:
                pred_land = pred_abs.flatten()
                tgt_land  = tgt_abs.flatten()

            diff = pred_land - tgt_land
            mae  = np.mean(np.abs(diff))
            mse  = np.mean(diff ** 2)
            rmse = np.sqrt(mse)

            pred_ssim = pred.copy()
            tgt_ssim  = tgt_np.copy()
            if land_mask is not None:
                pred_ssim[~land_mask] = 0.0
                tgt_ssim[~land_mask]  = 0.0
            pred_ssim_abs = pred_ssim * temp_range + g_min
            tgt_ssim_abs  = tgt_ssim  * temp_range + g_min
            try:
                s = ssim_metric(tgt_ssim_abs, pred_ssim_abs, data_range=temp_range)
            except:
                s = 0.0

            mse_scores.append(mse)
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            ssim_scores.append(s)

            # ---- capture first test sample for comparison plot ----
            if i == 0:
                first_sample_data['tgt_abs']  = tgt_abs
                first_sample_data['pred_abs'] = pred_abs
                first_sample_data['error_abs'] = tgt_abs - pred_abs
                first_sample_data['mse']  = mse
                first_sample_data['mae']  = mae
                first_sample_data['rmse'] = rmse
                first_sample_data['ssim'] = s
                # resolve lats/lons for axis ticks
                fn0 = os.path.basename(test_tuples[0][1])
                first_sample_data['filename'] = fn0

            if i in [50, 150, 300, 600, 900]:
                print(f"Sample {i}: MAE={mae:.3f}, RMSE={rmse:.3f}, SSIM={s:.3f}")

            fn = os.path.basename(test_tuples[i][1])
            m  = dt_re.search(fn)
            if m: dates.append(np.datetime64(m.group(1)))
            else: dates.append(np.datetime64('NaT'))

    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"


    # 8. First test sample comparison plot
    if first_sample_data:
        tgt_abs_plt  = np.flipud(first_sample_data['tgt_abs'])
        pred_abs_plt = np.flipud(first_sample_data['pred_abs'])
        err_abs_plt  = np.flipud(first_sample_data['error_abs'])


        # Reconstruct correct title timestamps
        fn0 = first_sample_data['filename']
        m_dt = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2})', fn0)

        try:
            out_dt = datetime.strptime(m_dt.group(1), '%Y-%m-%dT%H')

            in_start = out_dt - timedelta(hours=33)   # prev day 12 AM
            in_end   = out_dt - timedelta(hours=12)   # prev day 9 PM

            in_date  = f"{ordinal(in_start.day)} {in_start.strftime('%B, %Y')}"
            out_date = f"{ordinal(out_dt.day)} {out_dt.strftime('%B, %Y')}"

            title_str = (
                f"Input: {in_date} | 12am - 9pm   |   "
                f"Output: {out_date} | 9am"
            )

        except:
            title_str = "First Test Sample Comparison"

        # Build approximate lat/lon tick labels for the 11×11 grid (Mumbai region)
        lons = np.round(np.linspace(72.71, 73.71, 11), 2)
        lats = np.round(np.linspace(18.6,  19.6,  11), 2)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(title_str, fontsize=12, fontweight='bold')

        vmin_t = min(tgt_abs_plt.min(), pred_abs_plt.min())
        vmax_t = max(tgt_abs_plt.max(), pred_abs_plt.max())
        tick_vals = np.round(np.linspace(vmin_t, vmax_t, 7), 2)

        err_abs_val = np.abs(err_abs_plt).max()

        # ---- Actual ----
        ax = axes[0]
        im0 = ax.imshow(tgt_abs_plt, cmap='gray', aspect='auto',
                        vmin=vmin_t, vmax=vmax_t,
                        extent=[lons[0], lons[-1], lats[0], lats[-1]],
                        origin='lower')
        ax.set_title('Actual')
        ax.set_xlabel('Longitude →')
        ax.set_ylabel('Latitude ↑')
        cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        cbar0.set_label('2m Temperature (units)')
        cbar0.set_ticks(tick_vals)

        # ---- Predicted ----
        ax = axes[1]
        im1 = ax.imshow(pred_abs_plt, cmap='gray', aspect='auto',
                        vmin=vmin_t, vmax=vmax_t,
                        extent=[lons[0], lons[-1], lats[0], lats[-1]],
                        origin='lower')
        ax.set_title('Predicted')
        ax.set_xlabel('Longitude →')
        ax.set_ylabel('Latitude ↑')
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('2m Temperature (units)')
        cbar1.set_ticks(tick_vals)

        # ---- Error ----
        ax = axes[2]
        im2 = ax.imshow(err_abs_plt, cmap='gray', aspect='auto',
                        vmin=-err_abs_val, vmax=err_abs_val,
                        extent=[lons[0], lons[-1], lats[0], lats[-1]],
                        origin='lower')
        ax.set_title('Error = Actual - Predicted')
        ax.set_xlabel('Longitude →')
        ax.set_ylabel('Latitude ↑')
        cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label('Error (units)')

        # ---- Metrics text box ----
        metrics_str = (
            f"MSE: {first_sample_data['mse']:.6f}   "
            f"MAE: {first_sample_data['mae']:.6f}   "
            f"RMSE: {first_sample_data['rmse']:.6f}   "
            f"SSIM: {first_sample_data['ssim']:.6f}"
        )
        fig.text(0.5, 0.01, metrics_str, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        comp_path = os.path.join(eval_out, "first_test_sample_comparison.png")
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 First test sample comparison plot saved → {comp_path}")

    # 9. Save results & plots
    df = pd.DataFrame({
        'time':     dates,
        'mae_abs':  mae_scores,
        'rmse_abs': rmse_scores,
        'ssim':     ssim_scores
    })
    df.to_csv(os.path.join(eval_out, "absolute_results.csv"), index=False)


    # ==========================
    # Summary statistics
    # ==========================
    summary_stats = {
        "Mean MAE": np.mean(mae_scores),
        "Median MAE": np.median(mae_scores),
        "95th Percentile MAE": np.percentile(mae_scores, 95),
        "Mean RMSE": np.mean(rmse_scores),
        "Median RMSE": np.median(rmse_scores),
        "Mean SSIM": np.mean(ssim_scores),
        "Median SSIM": np.median(ssim_scores)
    }

    print("\n📊 YEARLY SUMMARY:")
    for k, v in summary_stats.items():
        print(f"{k}: {v:.4f}")

    def plot_metric(metric_name, values, color, title):
        plt.figure(figsize=(12, 4))
        plt.plot(dates, values, marker='.', linewidth=0.5, markersize=2, color=color)
        plt.title(title)
        plt.xlabel("Time (2024)")
        plt.ylabel(metric_name)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_out, f"{title.replace(' ', '_')}.png"))
        plt.close()

    plot_metric("SSIM",          ssim_scores, "tab:blue",   "SSIM 2024")
    plot_metric("MAE (Kelvin)",  mae_scores,  "tab:orange", "Absolute MAE 2024")
    plot_metric("RMSE (Kelvin)", rmse_scores, "tab:red",    "Absolute RMSE 2024")

    print(f"🎉 Done! Results saved in {eval_out}")


if __name__ == '__main__':
    main()