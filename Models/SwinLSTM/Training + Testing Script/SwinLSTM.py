import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
model_save_path = os.path.join(script_dir, "SwinLSTM_t2m_best.pth")
eval_out        = os.path.join(script_dir, "Eval_SwinLSTM_2024")
stats_path      = os.path.join(script_dir, "dataset_stats.json")
mask_path       = os.path.join(script_dir, "land_mask.npy")
losses_path     = os.path.join(script_dir, "train_val_losses_swinlstm.json")
checkpoint_path = os.path.join(script_dir, "training_checkpoint_swinlstm.pth")
coords_path     = os.path.join(script_dir, "grid_coords.json")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

input_len     = 8     # 8 frames × 3 h = 24 h context window
target_offset = 4     # 4 frames × 3 h = 12 h forecast horizon

img_size    = 11
IN_CHANNELS = 3       # temperature + sin(hour) + cos(hour)
batch_size  = 16
num_epochs  = 50
lr          = 1e-4

# ── SwinLSTM architecture hyper-parameters ──────────────────────────────────
EMBED_DIM   = 96    # token embedding dimension (= hidden dim)
NUM_HEADS   = 6     # attention heads; EMBED_DIM must be divisible
WINDOW_SIZE = 3     # spatial attention window (3×3); 11×11 padded to 12×12
NUM_BLOCKS  = 4     # Swin blocks per cell (alternating W-MSA / SW-MSA pairs)
MLP_RATIO   = 4.0   # FFN expansion ratio inside Swin blocks
DROP_RATE   = 0.0   # dropout (keep 0 for small grids)
HUBER_DELTA = 1.0   # Huber loss δ

# ── Multi-horizon loss weights: 3h | 6h | 9h | 12h ──────────────────────────
# Total = 0.05·L3h + 0.10·L6h + 0.20·L9h + 1.00·L12h + 0.05·GradLoss(12h)
MH_WEIGHTS       = [0.05, 0.10, 0.20, 1.00]
GRAD_LOSS_WEIGHT = 0.05   # weight of spatial-gradient sharpness term on 12h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
print(f"✅ Dataset Path: {base_dir}")
print(f"✅ Architecture  : SwinLSTM")
print(f"   embed_dim={EMBED_DIM} | num_heads={NUM_HEADS} | "
      f"window_size={WINDOW_SIZE} | swin_blocks/cell={NUM_BLOCKS}")
print(f"✅ input_len={input_len} | IN_CHANNELS={IN_CHANNELS} | "
      f"target_offset={target_offset} → 12 h-ahead prediction")


# ==========================================
# 2. PLOT HELPER FUNCTIONS
#    (kept identical to reference)
# ==========================================
def _ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _fmt_date(dt) -> str:
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    day = dt.day
    return f"{day}{_ordinal_suffix(day)} {dt.strftime('%B')}, {dt.year}"


def _fmt_time(dt) -> str:
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    h = dt.hour
    if h == 0:   return "12am"
    elif h < 12: return f"{h}am"
    elif h == 12: return "12pm"
    else:         return f"{h - 12}pm"


def _build_suptitle(seq_timestamps, tgt_timestamp) -> str:
    t_start  = seq_timestamps[0]
    t_end    = seq_timestamps[-1]
    inp_date = _fmt_date(t_start)
    inp_t0   = _fmt_time(t_start)
    inp_t1   = _fmt_time(t_end)
    out_date = _fmt_date(tgt_timestamp)
    out_time = _fmt_time(tgt_timestamp)
    return (f"Input: {inp_date} | {inp_t0} - {inp_t1}"
            f"  |  Output: {out_date} | {out_time}")


# ==========================================
# 3. DATA GENERATION & STATS
#    (kept identical to reference)
# ==========================================
def ensure_data_ready():
    """
    Ensures PNGs + land mask exist; computes/loads global min-max stats.
    Saves grid lat/lon coordinates for geo-referenced plot axis ticks.
    """
    # ---- Stats ----
    if not os.path.exists(stats_path):
        print("⚠️ Stats file missing. Scanning NetCDF files...")
        years    = ['2020', '2021', '2022', '2023', '2024']
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

    # ---- PNGs ----
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

    # Save lat/lon coordinates for geo-referenced axis ticks
    if not os.path.exists(coords_path):
        try:
            raw_lats     = t2m_all.latitude.values
            raw_lons     = t2m_all.longitude.values
            lats_flipped = np.flipud(raw_lats).tolist()
            lons_list    = raw_lons.tolist()
            with open(coords_path, 'w') as _cf:
                json.dump({'lats': lats_flipped, 'lons': lons_list}, _cf)
            print(f"✅ Grid coords saved: "
                  f"lat {lats_flipped[0]:.2f}→{lats_flipped[-1]:.2f}, "
                  f"lon {lons_list[0]:.2f}→{lons_list[-1]:.2f}")
        except Exception as _e:
            print(f"⚠️  Could not save grid coords: {_e}")

    # Land mask from NaN pattern
    land_mask_raw = None
    for i in range(len(t2m_arr)):
        if not np.all(np.isnan(t2m_arr[i])):
            land_mask_raw = np.flipud(~np.isnan(t2m_arr[i]))
            break
    if land_mask_raw is not None:
        np.save(mask_path, land_mask_raw)
        print(f"✅ Land mask saved ({land_mask_raw.shape}): "
              f"{land_mask_raw.sum()} land / {land_mask_raw.size} total pixels")

    # Save grayscale PNGs (3-hourly only, vertically flipped)
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
# 4. DATASET CLASS
#    (kept identical to reference)
# ==========================================
class HeatmapSeqFromPaths(Dataset):
    """
    Returns:
        seq : (input_len, 3, 11, 11)  — temperature + sin(hour) + cos(hour)
        tgt : (1, 11, 11)             — absolute normalised temperature [0,1]
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

        seq = torch.stack(seq_3ch, dim=0)   # (T, 3, 11, 11)
        return seq, tgt


# ==========================================
# 4b. MULTI-HORIZON DATASET
#     Extends base dataset to return all 4 target frames for
#     multi-horizon loss (3h, 6h, 9h, 12h).
#     HeatmapSeqFromPaths is NOT modified.
# ==========================================
class MultiHorizonDataset(HeatmapSeqFromPaths):
    """
    Extends HeatmapSeqFromPaths to return all 4 forecast-horizon targets.

    Tuple format expected: (seq_paths, [tgt_3h, tgt_6h, tgt_9h, tgt_12h], seq_ts)

    Returns:
        seq  : (T, 3, 11, 11)   — same as HeatmapSeqFromPaths
        tgts : (4, 1, 11, 11)   — stacked targets at +3h, +6h, +9h, +12h
    """
    def __getitem__(self, idx):
        seq_paths, tgt_paths_list, seq_timestamps = self.tuples[idx]

        frames = [self._load_img(p) for p in seq_paths]
        tgts   = [self._load_img(p) for p in tgt_paths_list]   # list of 4

        seq_3ch = []
        for frame, ts in zip(frames, seq_timestamps):
            sin_val, cos_val = self._time_encoding(ts)
            sin_map = torch.full_like(frame, sin_val)
            cos_map = torch.full_like(frame, cos_val)
            seq_3ch.append(torch.cat([frame, sin_map, cos_map], dim=0))

        seq    = torch.stack(seq_3ch, dim=0)    # (T, 3, 11, 11)
        tgts_t = torch.stack(tgts,    dim=0)    # (4, 1, 11, 11)
        return seq, tgts_t


# ==========================================
# 5. SWINLSTM MODEL
# ==========================================

# ---- 5a. Window utility functions ----

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition a spatial feature map into non-overlapping windows.

    Args:
        x           : (B, H, W, C)
        window_size : scalar window size (square)

    Returns: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor,
                   window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window_partition.

    Args:
        windows     : (num_windows * B, window_size, window_size, C)
        window_size : scalar window size
        H, W        : target spatial dimensions

    Returns: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---- 5b. Window-based Multi-Head Self-Attention ----

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA / SW-MSA) with
    relative position bias.

    Args:
        dim         : token feature dimension
        window_size : (Wh, Ww) — square assumed
        num_heads   : number of attention heads
        qkv_bias    : learnable QKV bias
        attn_drop   : attention weight dropout
        proj_drop   : output projection dropout
    """

    def __init__(self, dim: int, window_size: tuple, num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size          # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias: (2*Wh-1) × (2*Ww-1) × num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position index (N×N) where N = Wh*Ww
        coords_h    = torch.arange(window_size[0])
        coords_w    = torch.arange(window_size[1])
        coords      = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing='ij'))   # (2, Wh, Ww)
        coords_flat = torch.flatten(coords, 1)                      # (2, N)

        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()           # (N, N, 2)
        rel_coords[:, :, 0] += window_size[0] - 1
        rel_coords[:, :, 1] += window_size[1] - 1
        rel_coords[:, :, 0] *= 2 * window_size[1] - 1
        rel_pos_idx = rel_coords.sum(-1)    # (N, N)
        self.register_buffer('relative_position_index', rel_pos_idx)

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x    : (B_, N, C)  where B_ = num_windows × batch
            mask : (num_windows, N, N) shift mask, or None
        """
        B_, N, C = x.shape

        qkv = (self.qkv(x)
                   .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)            # each: (B_, heads, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)   # (B_, heads, N, N)

        # Add relative position bias
        N_win = self.window_size[0] * self.window_size[1]
        rel_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(N_win, N_win, -1)
            .permute(2, 0, 1)
            .contiguous()
        )                                                # (heads, N, N)
        attn = attn + rel_bias.unsqueeze(0)

        # Apply SW-MSA mask
        if mask is not None:
            nW   = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---- 5c. Swin Transformer Block ----

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block (STB) — the spatial reasoning primitive inside
    the SwinLSTM cell.

    Block layout (following the original Swin Transformer):
        x → LN → W-MSA (or SW-MSA) → residual
          → LN → MLP               → residual

    Even-indexed blocks within a cell use W-MSA  (shift_size = 0).
    Odd-indexed  blocks within a cell use SW-MSA (shift_size > 0).

    The spatial grid (11×11) is padded to the next multiple of window_size
    before partitioning and un-padded after; this is transparent to callers.

    Args:
        dim              : token feature dimension
        input_resolution : (H, W) of the spatial grid (before padding)
        num_heads        : attention heads
        window_size      : local window size (clamped to min(H,W))
        shift_size       : cyclic shift for SW-MSA (0 → W-MSA)
        mlp_ratio        : FFN hidden-dim expansion factor
        drop             : dropout on MLP and projection outputs
        attn_drop        : dropout on attention weights
    """

    def __init__(self, dim: int, input_resolution: tuple,
                 num_heads: int, window_size: int = 3,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads        = num_heads

        # Clamp window to grid size; disable shift for pure global attention
        self.window_size = min(window_size, min(input_resolution))
        self.shift_size  = shift_size if self.window_size < min(input_resolution) else 0

        H, W = input_resolution
        self.pad_h = (self.window_size - H % self.window_size) % self.window_size
        self.pad_w = (self.window_size - W % self.window_size) % self.window_size
        self.H_pad = H + self.pad_h
        self.W_pad = W + self.pad_w

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2   = nn.LayerNorm(dim)
        mlp_hidden   = int(dim * mlp_ratio)
        self.mlp     = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Build attention mask for SW-MSA (cyclic-shift regions must not attend)
        if self.shift_size > 0:
            img_mask = torch.zeros(1, self.H_pad, self.W_pad, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask    = (attn_mask
                            .masked_fill(attn_mask != 0, -100.0)
                            .masked_fill(attn_mask == 0,   0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, H*W, C)
        Returns: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Token count {L} ≠ H×W={H*W}"

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiple of window_size
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, 0, 0, self.pad_w, 0, self.pad_h))

        # Cyclic shift (SW-MSA only)
        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        # Window partition → attention → merge
        wins = window_partition(x, self.window_size)            # (nW*B, ws, ws, C)
        wins = wins.view(-1, self.window_size ** 2, C)
        wins = self.attn(wins, mask=self.attn_mask)             # (nW*B, ws^2, C)
        wins = wins.view(-1, self.window_size, self.window_size, C)
        x    = window_reverse(wins, self.window_size,
                              self.H_pad, self.W_pad)           # (B, H_pad, W_pad, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))

        # Remove padding and flatten
        if self.pad_h > 0 or self.pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x.view(B, H * W, C)     # residual 1
        x = x + self.mlp(self.norm2(x))         # residual 2 (MLP)
        return x


# ---- 5d. SwinLSTM Cell ----

class SwinLSTMCell(nn.Module):
    """
    SwinLSTM Recurrent Cell — the core contribution of the SwinLSTM paper.

    The ConvLSTM convolution is replaced by Swin Transformer Blocks, giving
    the model global(-ish) spatial reasoning within each recurrent step.

    Computation graph:
        Z  = Linear([Xt ‖ Ht-1])            # merge input & prior hidden
        Z  = STB_0(Z) → STB_1(Z) → …       # Swin blocks (W-MSA + SW-MSA)
        Gi = GateLinear(Z)  → 4 × dim splits
        i  = σ(Gi_i),  f = σ(Gi_f)
        o  = σ(Gi_o),  g = tanh(Gi_g)
        Ct = f⊙Ct-1 + i⊙g                  # cell update
        Ht = o⊙tanh(Ct)                     # hidden update
        Ht = LayerNorm(Ht)

    Args:
        input_dim        : feature dimension of the input token sequence
        hidden_dim       : feature dimension of hidden / cell states
        input_resolution : (H, W) spatial grid shape
        num_heads        : attention heads in each Swin block
        window_size      : local attention window size
        num_swin_blocks  : number of alternating W-MSA / SW-MSA blocks
        mlp_ratio        : FFN expansion ratio in Swin blocks
        drop             : dropout rate
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 input_resolution: tuple,
                 num_heads: int = 6, window_size: int = 3,
                 num_swin_blocks: int = 4,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.hidden_dim       = hidden_dim
        self.input_resolution = input_resolution

        # Project concatenated [Xt, Ht-1] into hidden space
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)

        # Alternating W-MSA (even idx) / SW-MSA (odd idx) blocks
        self.swin_blocks = nn.ModuleList()
        for i in range(num_swin_blocks):
            shift = (window_size // 2) if (i % 2 == 1) else 0
            self.swin_blocks.append(
                SwinTransformerBlock(
                    dim=hidden_dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                )
            )

        # Project to 4 LSTM gates (i, f, o, g)
        self.gate_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)

        # Post-update layer norm on the hidden state
        self.norm_h = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: torch.Tensor,
                h: torch.Tensor,
                c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, N, input_dim)   — embedded current input
            h : (B, N, hidden_dim)  — previous hidden state
            c : (B, N, hidden_dim)  — previous cell state

        Returns:
            h_new : (B, N, hidden_dim)
            c_new : (B, N, hidden_dim)
        """
        # Merge + project
        z = self.input_proj(torch.cat([x, h], dim=-1))   # (B, N, hidden_dim)

        # Spatial reasoning via alternating Swin blocks
        for blk in self.swin_blocks:
            z = blk(z)

        # LSTM gate computation
        gates = self.gate_proj(z)                          # (B, N, 4·hidden_dim)
        i_g, f_g, o_g, g_g = torch.chunk(gates, 4, dim=-1)

        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        o_g = torch.sigmoid(o_g)
        g_g = torch.tanh(g_g)

        c_new = f_g * c + i_g * g_g
        h_new = self.norm_h(o_g * torch.tanh(c_new))

        return h_new, c_new

    def init_hidden(self, B: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.input_resolution[0] * self.input_resolution[1]
        h = torch.zeros(B, N, self.hidden_dim, device=device)
        c = torch.zeros(B, N, self.hidden_dim, device=device)
        return h, c


# ---- 5e. Full SwinLSTM Model ----

class SwinLSTMModel(nn.Module):
    """
    End-to-end SwinLSTM model for spatiotemporal 2-metre temperature
    forecasting.

    Architecture upgrades vs. baseline
    -----------------------------------
    • embed_dim  : 96  (↑ from 64)
    • num_heads  : 6   (↑ from 4)
    • num_blocks : 4   (↑ from 2)  alternating W-MSA / SW-MSA per cell
    • Decoder    : Conv3×3(96→64) → Conv3×3_dilated(64→64) →
                   Conv3×3(64→32) → Conv1×1(32→1) → Sigmoid
    • forward()  : returns all 4 horizon predictions [3h,6h,9h,12h]
                   during training, or only the 12h prediction during eval.

    Pipeline
    --------
    1. Frame Embedding
       Each (B, 3, 11, 11) frame → (B, 121, 3) tokens → (B, 121, embed_dim).

    2. Warm-up Recurrence  [T = 8 input frames]
       Process the observed sequence through the SwinLSTM cell.

    3. Autoregressive Rollout  [target_offset = 4 steps × 3 h = 12 h]
       Each step records its prediction; these are the multi-horizon outputs.

    4. Decoder  [dilated convolution head]
       (B, embed_dim, 11, 11) → (B, 1, 11, 11) ∈ [0, 1]

    Args:
        in_channels     : input channels per frame (3: temp + sin + cos)
        embed_dim       : token embedding / hidden dimension
        img_size        : spatial grid side length (must be square)
        num_heads       : attention heads (embed_dim divisible by num_heads)
        window_size     : local attention window (padded if needed)
        num_swin_blocks : Swin blocks per cell (alternating W/SW-MSA)
        mlp_ratio       : FFN expansion ratio in Swin blocks
        drop            : dropout rate
        target_offset   : forecast steps (× 3 h each = forecast horizon)
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 96,
                 img_size: int = 11, num_heads: int = 6,
                 window_size: int = 3, num_swin_blocks: int = 4,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 target_offset: int = 4):
        super().__init__()
        self.embed_dim        = embed_dim
        self.img_size         = img_size
        self.target_offset    = target_offset
        self.input_resolution = (img_size, img_size)

        # ── Frame embedding (pixel-level; patch_size = 1, no downsampling) ──
        self.frame_embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Single SwinLSTM cell (shared across warm-up AND rollout) ──
        self.cell = SwinLSTMCell(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            input_resolution=self.input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            num_swin_blocks=num_swin_blocks,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

        # ── Reconstruction head: token map → single-channel heatmap ─────────
        # Conv3×3(D→64) → Conv3×3_d2(64→64) → Conv3×3(64→32) → Conv1×1(32→1)
        # Sigmoid produces normalised output ∈ [0, 1].
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),   # dilated
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1,  kernel_size=1),
            nn.Sigmoid(),
        )

        # ── Pre-computed rotation constants for one 3-h time step ──
        # sin(θ + Δ) = sinθ·cosΔ + cosθ·sinΔ
        # cos(θ + Δ) = cosθ·cosΔ − sinθ·sinΔ
        angle_step = 2.0 * math.pi * 3.0 / 24.0
        self.register_buffer('_cos_d', torch.tensor(math.cos(angle_step)))
        self.register_buffer('_sin_d', torch.tensor(math.sin(angle_step)))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _embed_frame(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H*W, embed_dim)"""
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, N, C)
        return self.frame_embed(tokens)                          # (B, N, embed_dim)

    def _to_spatial(self, h: torch.Tensor) -> torch.Tensor:
        """(B, N, C) → (B, C, H, W)"""
        B, N, C = h.shape
        H = W = self.img_size
        return h.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden state tokens to a normalised temperature map.
        (B, N, embed_dim) → (B, 1, H, W)  ∈ [0, 1]
        Sigmoid is applied inside self.head.
        """
        feat = self._to_spatial(h)      # (B, embed_dim, H, W)
        return self.head(feat)

    def _advance_time(self,
                      sin_map: torch.Tensor,
                      cos_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate 2-D sin/cos time maps forward by one 3-h interval."""
        new_sin = sin_map * self._cos_d + cos_map * self._sin_d
        new_cos = cos_map * self._cos_d - sin_map * self._sin_d
        return new_sin, new_cos

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, seq: torch.Tensor,
                return_all: bool | None = None) -> torch.Tensor | list:
        """
        Args:
            seq        : (B, T=8, 3, 11, 11)  — normalised input sequence
            return_all : if True  → return list of 4 tensors [3h,6h,9h,12h];
                         if False → return single tensor (12h) for eval/inference;
                         if None  → defaults to self.training flag.

        Returns (training):   list of 4 × (B, 1, 11, 11)
        Returns (eval/infer): (B, 1, 11, 11)  — 12h prediction only
        """
        if return_all is None:
            return_all = self.training

        B, T, C, H, W = seq.shape

        # ── Step 1: initialise hidden/cell states ─────────────────────────
        h, c = self.cell.init_hidden(B, seq.device)

        # ── Step 2: warm-up recurrence (process all T observed frames) ────
        for t in range(T):
            x_emb = self._embed_frame(seq[:, t])    # (B, N, embed_dim)
            h, c  = self.cell(x_emb, h, c)

        # ── Step 3: autoregressive rollout — collect all horizon outputs ──
        # Seed: reconstruction from the warm-up hidden state.
        prev_pred = self._reconstruct(h)            # (B, 1, H, W)

        # Time encoding at the last observed frame (t7)
        last_sin = seq[:, -1, 1:2].clone()          # (B, 1, H, W)
        last_cos = seq[:, -1, 2:3].clone()          # (B, 1, H, W)

        predictions: list[torch.Tensor] = []
        for _ in range(self.target_offset):
            # Advance time by one 3-h step
            last_sin, last_cos = self._advance_time(last_sin, last_cos)

            # Compose 3-channel input: [temp_pred, sin(t_next), cos(t_next)]
            x_next = torch.cat([prev_pred, last_sin, last_cos], dim=1)  # (B, 3, H, W)

            # Step cell and reconstruct
            h, c      = self.cell(self._embed_frame(x_next), h, c)
            prev_pred = self._reconstruct(h)
            predictions.append(prev_pred)   # collect +3h, +6h, +9h, +12h

        # Training  → [pred_3h, pred_6h, pred_9h, pred_12h]
        # Eval/infer → pred_12h  (preserves existing eval/inference API)
        if return_all:
            return predictions
        return predictions[-1]  # (B, 1, 11, 11) — 12h only


# ==========================================
# 6. LOSS FUNCTION
# ==========================================

class MultiHorizonMaskedLoss(nn.Module):
    """
    Multi-horizon training loss with masked Huber per horizon + gradient term.

    Formula
    -------
        L_mh    = Σ_k  w_k × MaskedHuber(pred_k, tgt_k)
                    k ∈ {3h:0.05, 6h:0.10, 9h:0.20, 12h:1.00}

        L_grad  = L1( ∇pred_12h , ∇tgt_12h )   [horiz. + vert. finite diffs]

        L_total = L_mh  +  grad_weight × L_grad

    Args:
        land_mask       : (H, W) boolean numpy array; None → no masking
        delta           : Huber δ (smooth L1 transition point)
        horizon_weights : list of 4 scalars [w_3h, w_6h, w_9h, w_12h]
        grad_weight     : relative weight of the 12h gradient term
    """

    def __init__(self,
                 land_mask=None,
                 delta: float = 1.0,
                 horizon_weights: list | None = None,
                 grad_weight: float = 0.05):
        super().__init__()
        self.delta           = delta
        self.grad_weight     = grad_weight
        self.horizon_weights = horizon_weights if horizon_weights is not None \
                               else [0.05, 0.10, 0.20, 1.00]

        if land_mask is not None:
            # Shape: (1, 1, H, W) for broadcasting over (B, 1, H, W)
            mask_t = (torch.tensor(land_mask, dtype=torch.float32)
                      .unsqueeze(0).unsqueeze(0))
            self.register_buffer('mask', mask_t)
        else:
            self.mask = None

    def _masked_huber(self, pred: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """Huber loss on land pixels only (sea zeroed out by mask)."""
        if self.mask is not None:
            mask = self.mask.to(pred.device).expand_as(pred).bool()
            return F.huber_loss(pred[mask], target[mask],
                                delta=self.delta, reduction='mean')
        return F.huber_loss(pred, target, delta=self.delta, reduction='mean')

    def _gradient_loss(self, pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """Isotropic finite-difference sharpness loss (horizontal + vertical).
        Preserves sharp UHI gradients at coastlines / urban boundaries."""
        dx_p = pred  [:, :, :, 1:] - pred  [:, :, :, :-1]
        dy_p = pred  [:, :, 1:, :] - pred  [:, :, :-1, :]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

    def forward(self,
                preds: list,
                targets: list) -> torch.Tensor:
        """
        Args:
            preds   : list of 4 (B, 1, H, W) — model predictions [3h,6h,9h,12h]
            targets : list of 4 (B, 1, H, W) — ground-truth     [3h,6h,9h,12h]
        Returns:
            scalar loss tensor
        """
        # Multi-horizon weighted masked-Huber sum
        mh_loss = sum(
            w * self._masked_huber(p, t)
            for w, p, t in zip(self.horizon_weights, preds, targets)
        )
        # Spatial gradient sharpness term on the 12h target only
        grad_loss = self._gradient_loss(preds[-1], targets[-1])
        return mh_loss + self.grad_weight * grad_loss


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():

    # ── 1. Data preparation & normalisation stats ─────────────────────────
    g_min, g_max = ensure_data_ready()
    temp_range   = g_max - g_min
    print(f"📉 Denormalisation Key: Min={g_min:.2f}K, Range={temp_range:.2f}K")

    # ── 2. Gather & sort PNG files ─────────────────────────────────────────
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

    # ── 3. Build sliding-window samples (3-h exact intervals, no cross-year) ─
    EXPECTED_INTERVAL = timedelta(hours=3)
    total_window      = input_len + target_offset

    # Two parallel dicts:
    #   samples_by_year    : (seq_idxs, targ_idx_12h)          for test set
    #   samples_by_year_mh : (seq_idxs, [tgt_3h,..,tgt_12h])   for train/val MH loss
    samples_by_year    = {}
    samples_by_year_mh = {}

    for i in range(len(paths) - total_window + 1):
        all_idxs = list(range(i, i + total_window))
        # Reject any window with irregular intervals
        if any(
            timestamps[all_idxs[k + 1]] - timestamps[all_idxs[k]] != EXPECTED_INTERVAL
            for k in range(len(all_idxs) - 1)
        ):
            continue
        # Reject cross-year samples
        y_set = {years[j] for j in all_idxs}
        if len(y_set) != 1:
            continue

        seq_idxs = all_idxs[:input_len]
        targ_idx = all_idxs[input_len + target_offset - 1]     # 12h only (test)
        tgt_idxs = [all_idxs[input_len + k]                    # +3h,+6h,+9h,+12h
                    for k in range(target_offset)]
        yr       = years[targ_idx]

        samples_by_year.setdefault(yr,    []).append((seq_idxs, targ_idx))
        samples_by_year_mh.setdefault(yr, []).append((seq_idxs, tgt_idxs))

    # ── 4. Train/val (2020-2023) and test (2024) splits ───────────────────
    train_years = [2020, 2021, 2022, 2023]
    test_year   = 2024

    # Multi-horizon tuples for training (4 targets each)
    train_samples_mh = []
    for y in train_years:
        train_samples_mh.extend(samples_by_year_mh.get(y, []))

    train_tuples_mh = [
        (
            [paths[idx] for idx in s],          # seq paths  (8)
            [paths[t]   for t   in ts],          # tgt paths  (4: 3h,6h,9h,12h)
            [timestamps[idx] for idx in s],      # seq timestamps
        )
        for s, ts in train_samples_mh
    ]

    # Standard single-target tuples for test evaluation (unchanged structure)
    test_tuples = [
        ([paths[idx] for idx in s], paths[t], [timestamps[idx] for idx in s])
        for s, t in samples_by_year.get(test_year, [])
    ]

    print(f"✅ Train Samples: {len(train_tuples_mh)} | Test Samples: {len(test_tuples)}")

    # ── 5. Datasets & data loaders ────────────────────────────────────────
    transform = transforms.Compose([transforms.ToTensor()])

    # MultiHorizonDataset for train / val  (returns 4 targets per sample)
    full_train_ds = MultiHorizonDataset(train_tuples_mh, transform=transform)
    # Standard dataset for test  (single 12h target — evaluation unchanged)
    test_ds       = HeatmapSeqFromPaths(test_tuples, transform=transform)

    n_total = len(full_train_ds)
    n_train = int(0.8 * n_total)

    train_ds = Subset(full_train_ds, list(range(0, n_train)))
    val_ds   = Subset(full_train_ds, list(range(n_train, n_total)))

    print(f"✅ Sequential Split: {len(train_ds)} Training / {len(val_ds)} Validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── 6. Load land mask (used for both loss masking & evaluation) ───────
    if os.path.exists(mask_path):
        land_mask_np = np.load(mask_path)
        if land_mask_np.shape != (img_size, img_size):
            _mimg      = Image.fromarray(
                land_mask_np.astype(np.uint8) * 255).convert("L")
            _mimg      = _mimg.resize((img_size, img_size), Image.NEAREST)
            land_mask_np = np.array(_mimg) > 0
        print(f"✅ Land mask: {land_mask_np.sum()} / {land_mask_np.size} land pixels")
    else:
        land_mask_np = None
        print("⚠️  No land mask found — training loss without masking")

    # ── 7. Model, loss, optimiser, scheduler ──────────────────────────────
    model = SwinLSTMModel(
        in_channels     = IN_CHANNELS,
        embed_dim       = EMBED_DIM,
        img_size        = img_size,
        num_heads       = NUM_HEADS,
        window_size     = WINDOW_SIZE,
        num_swin_blocks = NUM_BLOCKS,
        mlp_ratio       = MLP_RATIO,
        drop            = DROP_RATE,
        target_offset   = target_offset,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ SwinLSTM parameters: {total_params:,}")

    # Multi-horizon loss: 0.05·L3h + 0.10·L6h + 0.20·L9h + 1.00·L12h
    #                     + 0.05 × gradient_sharpness_loss(12h)
    criterion = MultiHorizonMaskedLoss(
        land_mask       = land_mask_np,
        delta           = HUBER_DELTA,
        horizon_weights = MH_WEIGHTS,
        grad_weight     = GRAD_LOSS_WEIGHT,
    ).to(device)

    # AdamW with cosine learning-rate annealing
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    scaler           = GradScaler('cuda')
    best_val         = float('inf')
    patience         = 8
    patience_counter = 0
    ema_val          = None
    ema_alpha        = 0.3
    start_epoch      = 1
    training_done    = False

    # ── 8. Load saved losses / checkpoint ────────────────────────────────
    if os.path.exists(losses_path):
        with open(losses_path, 'r') as _f:
            _d = json.load(_f)
            train_losses = _d.get('train', [])
            val_losses   = _d.get('val',   [])
        print(f"📂 Loaded {len(train_losses)} epochs of prior loss history")
    else:
        train_losses = []
        val_losses   = []

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
            print(f"✅ Resumed from epoch {ckpt['epoch']} | "
                  f"Best val: {best_val:.6f}")
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
            model.load_state_dict(
                torch.load(model_save_path, map_location=device))
            print("⚡ Loaded saved model. Skipping training.")
            training_done = True
        except RuntimeError as e:
            print(f"⚠️  Saved model incompatible ({e}). Starting fresh.")
            os.remove(model_save_path)

    # ── 9. Training loop ──────────────────────────────────────────────────
    if not training_done:
        print(f"\n🚀 {'Starting' if start_epoch == 1 else 'Resuming'} "
              f"SwinLSTM training from epoch {start_epoch}...\n")

        for epoch in range(start_epoch, num_epochs + 1):

            # ---- Training (multi-horizon) --------------------------------
            model.train()
            train_loss = 0.0

            for seq, tgts in tqdm(train_loader, desc=f"Epoch {epoch:02d} [train]"):
                # seq  : (B, T, 3, 11, 11)
                # tgts : (B, 4, 1, 11, 11) — targets at [+3h,+6h,+9h,+12h]
                seq  = seq.to(device)
                tgts = tgts.to(device)
                tgt_list = [tgts[:, k] for k in range(target_offset)]

                optimizer.zero_grad()
                with autocast('cuda'):
                    preds = model(seq)              # list of 4 (training=True)
                    loss  = criterion(preds, tgt_list)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # ---- Validation (12h only) ------------------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, tgts in val_loader:
                    seq  = seq.to(device)
                    tgts = tgts.to(device)

                    with autocast('cuda'):
                        # 🔹 Only final prediction (12h)
                        pred_12h = model(seq, return_all=False)

                        # 🔹 Only final target (12h)
                        tgt_12h = tgts[:, -1]

                        # 🔹 Validation loss only on 12h
                        loss = F.huber_loss(pred_12h, tgt_12h)

                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            # EMA-smoothed validation for early stopping
            ema_val = (avg_val if ema_val is None
                       else ema_alpha * avg_val + (1 - ema_alpha) * ema_val)

            # Cosine LR step (epoch-level, no metric required)
            scheduler.step()

            if ema_val < best_val:
                best_val         = ema_val
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print("🔥 Saved Best Model")
            else:
                patience_counter += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:02d}: "
                f"Train {avg_train:.6f} | "
                f"Val {avg_val:.6f} | "
                f"LR {current_lr:.2e} | "
                f"Patience {patience_counter}/{patience}"
            )

            # Checkpoint
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

    # ── 10. Save train/val loss plot ──────────────────────────────────────
    if train_losses:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses,
                 label='Train Loss', color='tab:blue',
                 marker='o', markersize=4)
        plt.plot(range(1, len(val_losses) + 1), val_losses,
                 label='Val Loss', color='tab:orange',
                 marker='o', markersize=4)
        plt.title("SwinLSTM — Training vs Validation Loss "
                  "(Masked Huber + Gradient Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_out, "train_val_loss.png"), dpi=200)
        plt.close()
        print("📈 Saved: train_val_loss.png")

    # ── 11. Evaluation on 2024 test set ───────────────────────────────────
    print("\n📊 Evaluating SwinLSTM on 2024 test set...")

    if os.path.exists(model_save_path):
        model.load_state_dict(
            torch.load(model_save_path, map_location=device))
        print("✅ Best model weights loaded for evaluation.")
    model.eval()

    # Land mask for evaluation metrics
    if land_mask_np is not None:
        land_mask = land_mask_np
        print(f"✅ Land mask: {land_mask.sum()} / {land_mask.size} land pixels")
    else:
        land_mask = None
        print("⚠️ No land mask — metrics over all pixels")

    # Grid coordinates for geo-referenced plots
    if os.path.exists(coords_path):
        with open(coords_path) as _cf:
            _coords   = json.load(_cf)
            grid_lats = np.array(_coords['lats'])
            grid_lons = np.array(_coords['lons'])
        print(f"✅ Grid coords: lat {grid_lats[0]:.2f}→{grid_lats[-1]:.2f}, "
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
        for i, (seq, tgt) in enumerate(tqdm(test_loader, desc="Evaluating")):

            with autocast('cuda'):
                pred_t = model(seq.to(device)).cpu().float()

            pred_np = pred_t.numpy().squeeze()
            tgt_np  = tgt.numpy().squeeze()

            pred_np = np.nan_to_num(pred_np, nan=0.0)
            tgt_np  = np.nan_to_num(tgt_np,  nan=0.0)

            pred_abs = denorm(pred_np)
            tgt_abs  = denorm(tgt_np)

            # MAE / RMSE — land pixels only
            if land_mask is not None:
                pred_land = pred_abs[land_mask]
                tgt_land  = tgt_abs[land_mask]
            else:
                pred_land = pred_abs.flatten()
                tgt_land  = tgt_abs.flatten()

            diff = pred_land - tgt_land
            mae  = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))

            # SSIM — sea pixels zeroed out
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

            # ── First-sample comparison plot (publication style) ──────────
            if i == 0:
                mse_val  = float(np.mean(diff ** 2))

                seq_ts_0     = test_tuples[0][2]
                tgt_ts_0     = dates[0]
                suptitle_str = _build_suptitle(seq_ts_0, tgt_ts_0)

                vmin_shared = min(tgt_abs.min(), pred_abs.min())
                vmax_shared = max(tgt_abs.max(), pred_abs.max())

                error_abs = tgt_abs - pred_abs
                err_max   = max(abs(error_abs.min()), abs(error_abs.max()))
                err_max   = err_max if err_max > 0 else 1.0

                # Axis extent from real lat/lon
                if grid_lats is not None and grid_lons is not None:
                    half_dlon  = abs(grid_lons[1] - grid_lons[0]) / 2.0
                    half_dlat  = abs(grid_lats[1] - grid_lats[0]) / 2.0
                    img_extent = [
                        grid_lons[0]  - half_dlon,
                        grid_lons[-1] + half_dlon,
                        grid_lats[-1] - half_dlat,
                        grid_lats[0]  + half_dlat,
                    ]
                    lon_ticks = np.round(np.linspace(
                        grid_lons[0], grid_lons[-1], min(5, len(grid_lons))), 1)
                    lat_ticks = np.round(np.linspace(
                        grid_lats[-1], grid_lats[0], min(5, len(grid_lats))), 1)
                else:
                    img_extent = None
                    lon_ticks  = None
                    lat_ticks  = None

                fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                                         constrained_layout=False)
                fig.subplots_adjust(
                    left=0.06, right=0.97,
                    top=0.82,  bottom=0.18,
                    wspace=0.38,
                )

                def _add_panel(ax, data, title, cmap, vmin, vmax, cbar_label):
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax,
                                  interpolation='nearest', aspect='auto')
                    if img_extent is not None:
                        kwargs['extent'] = img_extent
                        kwargs['origin'] = 'upper'
                    im = ax.imshow(data, **kwargs)
                    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
                    ax.set_xlabel("Longitude →", fontsize=9)
                    ax.set_ylabel("Latitude ↑",  fontsize=9)
                    if lon_ticks is not None:
                        ax.set_xticks(lon_ticks)
                        ax.set_xticklabels(
                            [f"{v:.1f}" for v in lon_ticks], fontsize=7.5)
                    if lat_ticks is not None:
                        ax.set_yticks(lat_ticks)
                        ax.set_yticklabels(
                            [f"{v:.1f}" for v in lat_ticks], fontsize=7.5)
                    ax.tick_params(axis='both', which='both', length=3, width=0.6)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(cbar_label, fontsize=8)
                    cbar.ax.tick_params(labelsize=7)

                _add_panel(axes[0], tgt_abs,   "Actual",
                           'gray', vmin_shared, vmax_shared, "2m Temperature (K)")
                _add_panel(axes[1], pred_abs,  "Predicted (SwinLSTM)",
                           'gray', vmin_shared, vmax_shared, "2m Temperature (K)")
                _add_panel(axes[2], error_abs, "Error = Actual − Predicted",
                           'gray', -err_max, err_max, "Error (K)")

                fig.suptitle(suptitle_str, fontsize=11,
                             fontweight='bold', y=0.96)

                metrics_str = (
                    f"MSE: {mse_val:.4f}    "
                    f"MAE: {mae:.4f}    "
                    f"RMSE: {rmse:.4f}    "
                    f"SSIM: {s:.4f}"
                )
                fig.text(
                    0.5, 0.04, metrics_str,
                    ha='center', va='center', fontsize=9,
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='white', edgecolor='#444444',
                              linewidth=1.0, alpha=0.9),
                )
                out_path = os.path.join(eval_out, "first_sample_comparison.png")
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close()
                print("🖼️  Saved: first_sample_comparison.png")

    # ── 12. Save metrics CSV ──────────────────────────────────────────────
    df = pd.DataFrame({
        'time':     dates,
        'mae_abs':  mae_scores,
        'rmse_abs': rmse_scores,
        'ssim':     ssim_scores,
    })
    csv_path = os.path.join(eval_out, "absolute_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: absolute_results.csv  ({len(df)} rows)")

    # ── 13. Metric trend plots ────────────────────────────────────────────
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

    plot_metric(ssim_scores,  "SSIM",         "SwinLSTM — SSIM 2024",
                "tab:blue",   "SSIM_2024.png")
    plot_metric(mae_scores,   "MAE (Kelvin)",  "SwinLSTM — Absolute MAE 2024",
                "tab:orange", "Absolute_MAE_2024.png")
    plot_metric(rmse_scores,  "RMSE (Kelvin)", "SwinLSTM — Absolute RMSE 2024",
                "tab:red",    "Absolute_RMSE_2024.png")

    # ── 14. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  SwinLSTM 2024 Test-Set Summary ({len(mae_scores)} samples)")
    print("=" * 55)
    print(f"  Mean SSIM          : {np.mean(ssim_scores):.4f}")
    print(f"  Mean MAE  (Kelvin) : {np.mean(mae_scores):.4f}")
    print(f"  Mean RMSE (Kelvin) : {np.mean(rmse_scores):.4f}")
    print("=" * 55)
    print(f"\n🎉 All outputs saved in: {eval_out}")


if __name__ == '__main__':
    main()