import os
import re
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_metric


# ==========================================================
# 0. CONFIGURATION
# ==========================================================

base_dir = r"../../../../Dataset"

script_dir = os.path.dirname(os.path.abspath(__file__))

heatmap_folder  = os.path.join(script_dir, "Heatmaps_t2m_gray_11x11")
eval_out        = os.path.join(script_dir, "Eval_SwinLSTM_2024")
model_save_path = os.path.join(script_dir, "SwinLSTM_t2m_best.pth")
stats_path      = os.path.join(script_dir, "dataset_stats.json")
mask_path       = os.path.join(script_dir, "land_mask.npy")
coords_path     = os.path.join(script_dir, "grid_coords.json")
csv_path        = os.path.join(script_dir, "absolute_results.csv")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

input_len     = 8
target_offset = 4

img_size = 11
IN_CHANNELS = 3

EMBED_DIM   = 96
NUM_HEADS   = 6
WINDOW_SIZE = 3
NUM_BLOCKS  = 4
MLP_RATIO   = 4.0
DROP_RATE   = 0.0

batch_size = 1

FORCE_REGENERATE_PNGS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Device: {device}")
print(f"✅ Dataset Path: {base_dir}")


# ==========================================================
# 1. HELPER FUNCTIONS
# ==========================================================

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

    if h == 0:
        return "12am"
    elif h < 12:
        return f"{h}am"
    elif h == 12:
        return "12pm"
    else:
        return f"{h - 12}pm"


def _build_suptitle(seq_timestamps, tgt_timestamp):

    t_start = seq_timestamps[0]
    t_end   = seq_timestamps[-1]

    start_date = _fmt_date(t_start)
    start_time = _fmt_time(t_start)

    end_date = _fmt_date(t_end)
    end_time = _fmt_time(t_end)

    out_date = _fmt_date(tgt_timestamp)
    out_time = _fmt_time(tgt_timestamp)

    if start_date == end_date:
        input_part = (
            f"Input: {start_date} | "
            f"{start_time} - {end_time}"
        )
    else:
        input_part = (
            f"Input: {start_date} {start_time} - "
            f"{end_date} {end_time}"
        )

    output_part = (
        f"Output: {out_date} | {out_time}"
    )

    return f"{input_part}  |  {output_part}"


def _find_nc_files(base_dir):

    years = ['2020', '2021', '2022', '2023', '2024']

    nc_files = []

    for y in years:
        year_dir = os.path.join(base_dir, y)

        if os.path.exists(year_dir):
            for root, _, files in os.walk(year_dir):
                for f in files:
                    if f.endswith('.nc'):
                        nc_files.append(os.path.join(root, f))

    return sorted(nc_files)


# ==========================================================
# 2. DATA PREP / PNG GENERATION
# ==========================================================

def ensure_data_ready():

    nc_files = _find_nc_files(base_dir)

    if not nc_files:
        raise RuntimeError(f"❌ No NetCDF files found in {base_dir}")

    print(f"✅ Found {len(nc_files)} NetCDF files")

    # ------------------------------------------------------
    # STATS
    # ------------------------------------------------------

    if not os.path.exists(stats_path):

        print("⚠️ Stats file missing. Scanning NetCDF files...")

        all_vals = []

        for path in tqdm(nc_files, desc="Scanning Stats"):

            try:
                ds = xr.open_dataset(path)
                da = ds['t2m']
                all_vals.append(da.values.flatten())

            except Exception as e:
                print(f"Skipping {path}: {e}")

        full_arr = np.concatenate(all_vals)

        g_min = float(np.nanmin(full_arr))
        g_max = float(np.nanmax(full_arr))

        with open(stats_path, 'w') as f:
            json.dump({"min": g_min, "max": g_max}, f)

        print(f"✅ Stats saved: Min={g_min:.2f}K, Max={g_max:.2f}K")

    else:

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        g_min = stats['min']
        g_max = stats['max']

        print(f"✅ Stats loaded: Min={g_min:.2f}K, Max={g_max:.2f}K")

    # ------------------------------------------------------
    # READ DATA
    # ------------------------------------------------------

    all_t2m = []

    for path in tqdm(nc_files, desc="Reading NetCDF"):

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

            if 'latitude' in da.coords:
                da = da.sortby('latitude')

            if 'longitude' in da.coords:
                da = da.sortby('longitude')

            if os.path.exists(coords_path):

                with open(coords_path, 'r') as cf:
                    coords = json.load(cf)

                target_lats = np.array(coords['lats'])[::-1]
                target_lons = np.array(coords['lons'])

                da = da.sel(
                    latitude=target_lats,
                    longitude=target_lons
                )

            all_t2m.append(da)

        except Exception as e:
            print(f"Skipping {path}: {e}")

    t2m_all = xr.concat(all_t2m, dim='time')

    t2m_arr = t2m_all.values
    times   = t2m_all.time.values

    if t2m_arr.shape[1:] != (11, 11):
        raise ValueError(
            f"Expected native grid shape (11,11), got {t2m_arr.shape[1:]}"
        )

    print(f"✅ Native spatial shape: {t2m_arr.shape[1]}×{t2m_arr.shape[2]}")

    # ------------------------------------------------------
    # SAVE GRID COORDS
    # ------------------------------------------------------

    if not os.path.exists(coords_path):

        try:

            raw_lats = t2m_all.latitude.values
            raw_lons = t2m_all.longitude.values

            lats_flipped = np.flipud(raw_lats).tolist()
            lons_list    = raw_lons.tolist()

            with open(coords_path, 'w') as f:
                json.dump(
                    {
                        'lats': lats_flipped,
                        'lons': lons_list
                    },
                    f
                )

            print(
                f"✅ Grid coords saved: "
                f"lat {lats_flipped[0]:.2f}→{lats_flipped[-1]:.2f}, "
                f"lon {lons_list[0]:.2f}→{lons_list[-1]:.2f}"
            )

        except Exception as e:
            print(f"⚠️ Could not save grid coords: {e}")

    else:
        print("✅ Grid coords file already exists")

    # ------------------------------------------------------
    # LAND MASK
    # ------------------------------------------------------

    if not os.path.exists(mask_path):

        land_mask_raw = None

        for i in range(len(t2m_arr)):

            if not np.all(np.isnan(t2m_arr[i])):

                land_mask_raw = np.flipud(~np.isnan(t2m_arr[i]))
                break

        if land_mask_raw is not None:

            np.save(mask_path, land_mask_raw)

            print(
                f"✅ Land mask saved ({land_mask_raw.shape}): "
                f"{land_mask_raw.sum()} land / {land_mask_raw.size} total pixels"
            )

    else:
        print("✅ Land mask file already exists")

    # ------------------------------------------------------
    # PNG GENERATION
    # ------------------------------------------------------

    existing_pngs = glob.glob(os.path.join(heatmap_folder, "*.png"))

    if FORCE_REGENERATE_PNGS or len(existing_pngs) == 0:

        print("⚠️ Generating 11×11 PNGs...")

        for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):

            if pd.Timestamp(times[i]).hour % 3 != 0:
                continue

            temp = np.flipud(t2m_arr[i])

            norm = np.clip(
                (temp - g_min) / (g_max - g_min),
                0.0,
                1.0
            )

            norm = np.nan_to_num(norm, nan=0.0)

            img = Image.fromarray(
                (norm * 255).astype(np.uint8),
                mode="L"
            )

            ts_str = np.datetime_as_string(
                times[i],
                unit='h'
            ).replace(":", "-")

            img.save(
                os.path.join(
                    heatmap_folder,
                    f"t2m_{i:05d}_{ts_str}.png"
                )
            )

        print(f"✅ PNGs saved in: {heatmap_folder}")

    else:
        print(
            f"✅ Found {len(existing_pngs)} PNGs. "
            f"Skipping PNG regeneration."
        )

    return g_min, g_max


# ==========================================================
# 3. DATASET
# ==========================================================

class HeatmapSeqFromPaths(Dataset):

    def __init__(self, tuples, transform=None):
        self.tuples = tuples
        self.transform = transform

    def __len__(self):
        return len(self.tuples)

    def _load_img(self, p):

        img = Image.open(p).convert("L")

        if self.transform:
            return self.transform(img)

        return transforms.ToTensor()(img)

    @staticmethod
    def _time_encoding(ts):

        hour = ts.hour + getattr(ts, 'minute', 0) / 60.0
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

            seq_3ch.append(
                torch.cat([frame, sin_map, cos_map], dim=0)
            )

        seq = torch.stack(seq_3ch, dim=0)

        return seq, tgt


# ==========================================================
# 4. SWIN MODEL
# ==========================================================

def window_partition(x, window_size):

    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    ).contiguous()

    return windows.view(
        -1,
        window_size,
        window_size,
        C
    )


def window_reverse(windows, window_size, H, W):

    B = int(
        windows.shape[0] /
        (H * W / window_size / window_size)
    )

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )

    return x.permute(
        0,
        1,
        3,
        2,
        4,
        5
    ).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
    ):

        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1),
                num_heads
            )
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])

        coords = torch.stack(
            torch.meshgrid(
                [coords_h, coords_w],
                indexing='ij'
            )
        )

        coords_flat = torch.flatten(coords, 1)

        rel_coords = (
            coords_flat[:, :, None] -
            coords_flat[:, None, :]
        )

        rel_coords = rel_coords.permute(
            1,
            2,
            0
        ).contiguous()

        rel_coords[:, :, 0] += window_size[0] - 1
        rel_coords[:, :, 1] += window_size[1] - 1
        rel_coords[:, :, 0] *= 2 * window_size[1] - 1

        rel_pos_idx = rel_coords.sum(-1)

        self.register_buffer(
            'relative_position_index',
            rel_pos_idx
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias
        )

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                C // self.num_heads
            )
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv.unbind(0)

        attn = (
            q * self.scale
        ) @ k.transpose(-2, -1)

        N_win = (
            self.window_size[0] *
            self.window_size[1]
        )

        rel_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ]
            .view(N_win, N_win, -1)
            .permute(2, 0, 1)
            .contiguous()
        )

        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:

            nW = mask.shape[0]

            attn = (
                attn.view(
                    B_ // nW,
                    nW,
                    self.num_heads,
                    N,
                    N
                )
                + mask.unsqueeze(1).unsqueeze(0)
            )

            attn = attn.view(
                -1,
                self.num_heads,
                N,
                N
            )

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B_, N, C)
        )

        x = self.proj(x)

        x = self.proj_drop(x)

        return x

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads,
                 window_size=3, shift_size=0, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.window_size = min(window_size, min(input_resolution))
        self.shift_size = shift_size if self.window_size < min(input_resolution) else 0

        H, W = input_resolution

        self.pad_h = (self.window_size - H % self.window_size) % self.window_size
        self.pad_w = (self.window_size - W % self.window_size) % self.window_size
        self.H_pad = H + self.pad_h
        self.W_pad = W + self.pad_w

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        if self.shift_size > 0:
            img_mask = torch.zeros(1, self.H_pad, self.W_pad, 1)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = (
                attn_mask
                .masked_fill(attn_mask != 0, -100.0)
                .masked_fill(attn_mask == 0, 0.0)
            )
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Token count {L} ≠ H×W={H*W}"

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, 0, 0, self.pad_w, 0, self.pad_h))

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        wins = window_partition(x, self.window_size)
        wins = wins.view(-1, self.window_size ** 2, C)
        wins = self.attn(wins, mask=self.attn_mask)
        wins = wins.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(wins, self.window_size, self.H_pad, self.W_pad)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if self.pad_h > 0 or self.pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x

class SwinLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        input_resolution
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.input_proj = nn.Linear(
            input_dim + hidden_dim,
            hidden_dim
        )

        self.swin_blocks = nn.ModuleList()

        for i in range(NUM_BLOCKS):

            shift = (
                WINDOW_SIZE // 2
                if (i % 2 == 1)
                else 0
            )

            self.swin_blocks.append(
                SwinTransformerBlock(
                    dim=hidden_dim,
                    input_resolution=input_resolution,
                    num_heads=NUM_HEADS,
                    window_size=WINDOW_SIZE,
                    shift_size=shift
                )
            )

        self.gate_proj = nn.Linear(
            hidden_dim,
            4 * hidden_dim
        )

        self.norm_h = nn.LayerNorm(hidden_dim)

    def forward(self, x, h, c):

        z = self.input_proj(
            torch.cat([x, h], dim=-1)
        )

        for blk in self.swin_blocks:
            z = blk(z)

        gates = self.gate_proj(z)

        i_g, f_g, o_g, g_g = torch.chunk(
            gates,
            4,
            dim=-1
        )

        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        o_g = torch.sigmoid(o_g)
        g_g = torch.tanh(g_g)

        c_new = f_g * c + i_g * g_g

        h_new = self.norm_h(
            o_g * torch.tanh(c_new)
        )

        return h_new, c_new

    def init_hidden(self, B, device):

        N = (
            self.input_resolution[0] *
            self.input_resolution[1]
        )

        h = torch.zeros(
            B,
            N,
            self.hidden_dim,
            device=device
        )

        c = torch.zeros(
            B,
            N,
            self.hidden_dim,
            device=device
        )

        return h, c


class SwinLSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed_dim = EMBED_DIM
        self.img_size = img_size

        self.frame_embed = nn.Sequential(
            nn.Linear(IN_CHANNELS, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM)
        )

        self.cell = SwinLSTMCell(
            input_dim=EMBED_DIM,
            hidden_dim=EMBED_DIM,
            input_resolution=(img_size, img_size)
        )

        self.head = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                3,
                padding=2,
                dilation=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        angle_step = 2.0 * math.pi * 3.0 / 24.0

        self.register_buffer(
            '_cos_d',
            torch.tensor(math.cos(angle_step))
        )

        self.register_buffer(
            '_sin_d',
            torch.tensor(math.sin(angle_step))
        )

    def _embed_frame(self, x):

        B, C, H, W = x.shape

        tokens = (
            x.permute(0, 2, 3, 1)
            .reshape(B, H * W, C)
        )

        return self.frame_embed(tokens)

    def _to_spatial(self, h):

        B, N, C = h.shape

        H = W = self.img_size

        return (
            h.reshape(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _reconstruct(self, h):

        feat = self._to_spatial(h)

        return self.head(feat)

    def _advance_time(self, sin_map, cos_map):

        new_sin = (
            sin_map * self._cos_d +
            cos_map * self._sin_d
        )

        new_cos = (
            cos_map * self._cos_d -
            sin_map * self._sin_d
        )

        return new_sin, new_cos

    def forward(self, seq):

        B, T, C, H, W = seq.shape

        h, c = self.cell.init_hidden(B, seq.device)

        for t in range(T):

            x_emb = self._embed_frame(seq[:, t])

            h, c = self.cell(x_emb, h, c)

        prev_pred = self._reconstruct(h)

        last_sin = seq[:, -1, 1:2].clone()
        last_cos = seq[:, -1, 2:3].clone()

        for _ in range(target_offset):

            last_sin, last_cos = self._advance_time(
                last_sin,
                last_cos
            )

            x_next = torch.cat(
                [prev_pred, last_sin, last_cos],
                dim=1
            )

            h, c = self.cell(
                self._embed_frame(x_next),
                h,
                c
            )

            prev_pred = self._reconstruct(h)

        return prev_pred


# ==========================================================
# 5. CSV / TEST TUPLES
# ==========================================================

def load_existing_csv():

    if not os.path.exists(csv_path):
        raise RuntimeError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"time", "mae_abs", "rmse_abs", "ssim"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV missing columns: {sorted(missing)}"
        )

    df["time"] = pd.to_datetime(df["time"])

    return df


def build_test_tuples_from_pngs():

    pngs = sorted(
        glob.glob(
            os.path.join(
                heatmap_folder,
                "*.png"
            )
        )
    )

    dt_re = re.compile(
        r'.*?_(\d{4}-\d{2}-\d{2}T\d{2})'
    )

    file_times = []

    for p in pngs:

        m = dt_re.search(os.path.basename(p))

        if m:

            ts = (
                np.datetime64(m.group(1))
                .astype('datetime64[m]')
                .astype(object)
            )

            file_times.append((p, ts))

    file_times.sort(key=lambda x: x[1])

    paths      = [x[0] for x in file_times]
    years      = [x[1].year for x in file_times]
    timestamps = [x[1] for x in file_times]

    EXPECTED_INTERVAL = pd.Timedelta(hours=3)

    samples_by_year = {}

    total_window = input_len + target_offset

    for i in range(len(paths) - total_window + 1):

        all_idxs = list(range(i, i + total_window))

        if any(
            pd.Timestamp(timestamps[all_idxs[k + 1]]) -
            pd.Timestamp(timestamps[all_idxs[k]])
            != EXPECTED_INTERVAL
            for k in range(len(all_idxs) - 1)
        ):
            continue

        y_set = {years[j] for j in all_idxs}

        if len(y_set) != 1:
            continue

        seq_idxs = all_idxs[:input_len]

        targ_idx = all_idxs[input_len + target_offset - 1]

        yr = years[targ_idx]

        samples_by_year.setdefault(
            yr,
            []
        ).append((seq_idxs, targ_idx))

    test_year = 2024

    test_tuples = [

        (
            [paths[idx] for idx in s],
            paths[t],
            [timestamps[idx] for idx in s]
        )

        for s, t in samples_by_year.get(test_year, [])
    ]

    return test_tuples


# ==========================================================
# 6. GEO HELPERS
# ==========================================================

def get_geo_extent_and_ticks():

    if os.path.exists(coords_path):

        with open(coords_path, "r") as f:
            coords = json.load(f)

        grid_lats = np.array(coords["lats"])
        grid_lons = np.array(coords["lons"])

        half_dlon = abs(grid_lons[1] - grid_lons[0]) / 2.0
        half_dlat = abs(grid_lats[1] - grid_lats[0]) / 2.0

        img_extent = [
            grid_lons[0] - half_dlon,
            grid_lons[-1] + half_dlon,
            grid_lats[-1] - half_dlat,
            grid_lats[0] + half_dlat,
        ]

        lon_ticks = np.round(
            np.linspace(
                grid_lons[0],
                grid_lons[-1],
                min(5, len(grid_lons))
            ),
            1
        )

        lat_ticks = np.round(
            np.linspace(
                grid_lats[-1],
                grid_lats[0],
                min(5, len(grid_lats))
            ),
            1
        )

        return img_extent, lon_ticks, lat_ticks

    return None, None, None


def load_land_mask():

    if os.path.exists(mask_path):

        land_mask_raw = np.load(mask_path)

        return land_mask_raw

    return None


# ==========================================================
# 7. PLOTTING
# ==========================================================

def plot_best_sample(
    tgt_abs,
    pred_abs,
    error_abs,
    seq_ts_best,
    tgt_ts_best,
    mae,
    rmse,
    mse_val,
    s,
    out_path,
    img_extent,
    lon_ticks,
    lat_ticks
):

    vmin_shared = min(
        tgt_abs.min(),
        pred_abs.min()
    )

    vmax_shared = max(
        tgt_abs.max(),
        pred_abs.max()
    )

    err_absmax = max(
        abs(error_abs.min()),
        abs(error_abs.max())
    )

    if err_absmax <= 0:
        err_absmax = 1.0

    suptitle_str = _build_suptitle(
        seq_ts_best,
        tgt_ts_best
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
        constrained_layout=False
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        top=0.82,
        bottom=0.18,
        wspace=0.38
    )

    def _add_panel(
        ax,
        data,
        title,
        cmap,
        vmin,
        vmax,
        cbar_label
    ):

        kwargs = dict(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            aspect='auto'
        )

        if img_extent is not None:

            kwargs["extent"] = img_extent
            kwargs["origin"] = "upper"

        im = ax.imshow(data, **kwargs)

        ax.set_title(
            title,
            fontsize=11,
            fontweight='bold',
            pad=6
        )

        ax.set_xlabel(
            "Longitude →",
            fontsize=9
        )

        ax.set_ylabel(
            "Latitude →",
            fontsize=9
        )

        if lon_ticks is not None:

            ax.set_xticks(lon_ticks)

            ax.set_xticklabels(
                [f"{v:.1f}" for v in lon_ticks],
                fontsize=7.5
            )

        if lat_ticks is not None:

            ax.set_yticks(lat_ticks)

            ax.set_yticklabels(
                [f"{v:.1f}" for v in lat_ticks],
                fontsize=7.5
            )

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        cbar.set_label(
            cbar_label,
            fontsize=8
        )

        cbar.ax.tick_params(labelsize=7)

    _add_panel(
        axes[0],
        tgt_abs,
        "Actual",
        "gray",
        vmin_shared,
        vmax_shared,
        "2m Temperature (K)"
    )

    _add_panel(
        axes[1],
        pred_abs,
        "Predicted",
        "gray",
        vmin_shared,
        vmax_shared,
        "2m Temperature (K)"
    )

    _add_panel(
        axes[2],
        error_abs,
        "Error = Actual - Predicted",
        "gray",
        -err_absmax,
        err_absmax,
        "Error (K)"
    )

    fig.suptitle(
        suptitle_str,
        fontsize=11,
        fontweight='bold',
        y=0.96
    )

    metrics_str = (
        f"MSE: {mse_val:.4f}    "
        f"MAE: {mae:.4f}    "
        f"RMSE: {rmse:.4f}    "
        f"SSIM: {s:.4f}"
    )

    fig.text(
        0.5,
        0.04,
        metrics_str,
        ha='center',
        va='center',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='#444444',
            linewidth=1.0,
            alpha=0.9
        )
    )

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches='tight'
    )

    plt.close()

    print(f"🖼️ Saved: {out_path}")


# ==========================================================
# 8. MAIN
# ==========================================================

def main():

    g_min, g_max = ensure_data_ready()

    temp_range = g_max - g_min

    land_mask = load_land_mask()

    img_extent, lon_ticks, lat_ticks = (
        get_geo_extent_and_ticks()
    )

    print("\n📄 Loading CSV...")

    df = load_existing_csv()

    print("📦 Rebuilding test tuples...")

    test_tuples = build_test_tuples_from_pngs()

    print(f"✅ Found {len(test_tuples)} test tuples")

    best_row = df.loc[df["mae_abs"].idxmin()]

    best_time = pd.Timestamp(best_row["time"])

    print("\n🏆 BEST SAMPLE")
    print(best_row)

    target_tuple = None
    target_idx = None

    for idx, tup in enumerate(test_tuples):

        seq_paths, tgt_path, seq_ts = tup

        m = re.search(
            r'_(\d{4}-\d{2}-\d{2}T\d{2})',
            os.path.basename(tgt_path)
        )

        if m:

            tgt_ts = pd.Timestamp(m.group(1))

            if tgt_ts == best_time:

                target_tuple = tup
                target_idx = idx
                break

    if target_tuple is None:
        raise RuntimeError(
            f"Could not find tuple for {best_time}"
        )

    transform = transforms.ToTensor()

    dataset = HeatmapSeqFromPaths(
        [target_tuple],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    print("\n🧠 Loading SwinLSTM model...")

    model = SwinLSTMModel().to(device)

    model.load_state_dict(
        torch.load(
            model_save_path,
            map_location=device
        )
    )

    model.eval()

    with torch.no_grad():

        for seq, tgt in loader:

            seq = seq.to(device)

            pred = model(seq)

            pred_np = pred.squeeze().cpu().numpy()
            tgt_np  = tgt.squeeze().cpu().numpy()

    pred_abs = pred_np * temp_range + g_min
    tgt_abs  = tgt_np  * temp_range + g_min

    if land_mask is not None:

        mask = land_mask.astype(bool)

        pred_valid = pred_abs[mask]
        tgt_valid  = tgt_abs[mask]

    else:

        pred_valid = pred_abs.flatten()
        tgt_valid  = tgt_abs.flatten()

    mse_val = np.mean((tgt_valid - pred_valid) ** 2)

    mae = np.mean(np.abs(tgt_valid - pred_valid))

    rmse = np.sqrt(mse_val)

    s = ssim_metric(
        tgt_abs,
        pred_abs,
        data_range=tgt_abs.max() - tgt_abs.min()
    )

    error_abs = tgt_abs - pred_abs

    seq_paths, tgt_path, seq_ts_best = target_tuple

    m = re.search(
        r'_(\d{4}-\d{2}-\d{2}T\d{2})',
        os.path.basename(tgt_path)
    )

    tgt_ts_best = pd.Timestamp(m.group(1))

    out_path = os.path.join(
        eval_out,
        "best_sample_comparison.png"
    )

    plot_best_sample(
        tgt_abs,
        pred_abs,
        error_abs,
        seq_ts_best,
        tgt_ts_best,
        mae,
        rmse,
        mse_val,
        s,
        out_path,
        img_extent,
        lon_ticks,
        lat_ticks
    )

    print("\n✅ DONE")
    print(f"Best sample index : {target_idx}")
    print(f"Best timestamp    : {best_time}")
    print(f"MAE               : {mae:.4f}")
    print(f"RMSE              : {rmse:.4f}")
    print(f"SSIM              : {s:.4f}")


if __name__ == "__main__":
    main()