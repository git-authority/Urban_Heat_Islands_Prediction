
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
from datetime import datetime, timedelta

import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_metric


# ==========================================================
# 0. CONFIGURATION
# ==========================================================
base_dir    = r"../../../../Dataset"
script_dir  = os.path.dirname(os.path.abspath(__file__))

heatmap_folder  = os.path.join(script_dir, "Heatmaps_t2m_gray_fast")
eval_out        = os.path.join(script_dir, "Eval_UNetConvLSTM_2024")
model_save_path = os.path.join(script_dir, "UNetConvLSTM_t2m_best.pth")
stats_path      = os.path.join(script_dir, "dataset_stats.json")
mask_path       = os.path.join(script_dir, "land_mask.npy")
coords_path     = os.path.join(script_dir, "grid_coords.json")
csv_path        = os.path.join(script_dir, "absolute_results_prompt_2_group_norm.csv")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

input_len  = 8
img_size   = 11
batch_size = 1
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Device: {device}")
print(f"✅ Dataset Path: {base_dir}")


# ==========================================================
# 1. HELPERS
# ==========================================================
def _ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _fmt_date(dt) -> str:
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    return f"{dt.day}{_ordinal_suffix(dt.day)} {dt.strftime('%B')}, {dt.year}"


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
    return f"{h - 12}pm"


def _build_suptitle(seq_timestamps, tgt_timestamp) -> str:
    start = seq_timestamps[0]
    end   = seq_timestamps[-1]
    start_date = _fmt_date(start)
    start_time = _fmt_time(start)
    end_date   = _fmt_date(end)
    end_time   = _fmt_time(end)
    out_date   = _fmt_date(tgt_timestamp)
    out_time   = _fmt_time(tgt_timestamp)

    if start_date == end_date:
        input_part = f"Input: {start_date} | {start_time} - {end_time}"
    else:
        input_part = f"Input: {start_date} {start_time} - {end_date} {end_time}"

    return f"{input_part}  |  Output: {out_date} | {out_time}"


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


def _parse_ts_from_png_name(png_path):
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2})', os.path.basename(png_path))
    if not m:
        return None
    return pd.to_datetime(m.group(1)).to_pydatetime()


def _time_encoding(ts):
    hour = ts.hour + getattr(ts, "minute", 0) / 60.0
    doy = ts.timetuple().tm_yday
    return (
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * doy / 365.0),
        math.cos(2 * math.pi * doy / 365.0),
    )


# ==========================================================
# 2. DATA PREP / PNG GENERATION
# ==========================================================
def ensure_data_ready():
    nc_files = _find_nc_files(base_dir)
    if not nc_files:
        raise RuntimeError(f"No NetCDF files found in {base_dir}")

    print(f"✅ Found {len(nc_files)} NetCDF files")

    if not os.path.exists(stats_path):
        print("⚠️ Stats file missing. Scanning NetCDF files...")
        all_vals = []
        for path in tqdm(nc_files, desc="Scanning Stats"):
            try:
                ds = xr.open_dataset(path)
                da = ds['t2m'].isel(latitude=slice(0, 11), longitude=slice(0, 11))
                all_vals.append(da.values.flatten())
            except Exception:
                pass

        full_arr = np.concatenate(all_vals)
        g_min = float(np.nanmin(full_arr))
        g_max = float(np.nanmax(full_arr))

        with open(stats_path, 'w') as f:
            json.dump({"min": g_min, "max": g_max}, f)
        print(f"✅ Stats saved: Min={g_min:.2f}K, Max={g_max:.2f}K")

    with open(stats_path, 'r') as f:
        stats = json.load(f)
    g_min, g_max = stats['min'], stats['max']
    print(f"✅ Stats loaded: Min={g_min:.2f}K, Max={g_max:.2f}K")

    import shutil
    if os.path.exists(heatmap_folder):
        shutil.rmtree(heatmap_folder)
    os.makedirs(heatmap_folder, exist_ok=True)
    if os.path.exists(mask_path):
        os.remove(mask_path)

    print("⚠️ Generating PNGs (fresh)...")

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
            all_t2m.append(da)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    t2m_all = xr.concat(all_t2m, dim='time')
    t2m_sub = t2m_all.isel(latitude=slice(0, 11), longitude=slice(0, 11))
    t2m_arr = t2m_sub.values
    times   = t2m_sub.time.values

    land_mask_raw = None
    for i in range(len(t2m_arr)):
        if not np.all(np.isnan(t2m_arr[i])):
            land_mask_raw = ~np.isnan(t2m_arr[i])
            break
    if land_mask_raw is not None:
        land_mask_raw = np.flipud(land_mask_raw)
        np.save(mask_path, land_mask_raw)

    for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
        if pd.Timestamp(times[i]).hour % 3 != 0:
            continue
        temp = t2m_arr[i]
        norm = (temp - g_min) / (g_max - g_min)
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)
        norm = np.flipud(norm)
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        ts_str = np.datetime_as_string(times[i], unit='h').replace(":", "-")
        img.save(os.path.join(heatmap_folder, f"t2m_{i:05d}_{ts_str}.png"))

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

    def _load(self, p):
        tensor = self._load_img(p)

        hour = 0
        m = re.search(r'T(\d{2})', os.path.basename(p))
        if m:
            hour = int(m.group(1))

        day_of_year = 1
        dm = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(p))
        if dm:
            day_of_year = datetime.strptime(dm.group(1), '%Y-%m-%d').timetuple().tm_yday

        sin_ch = torch.full_like(tensor, np.sin(2 * np.pi * hour / 24))
        cos_ch = torch.full_like(tensor, np.cos(2 * np.pi * hour / 24))
        sin_doy = torch.full_like(tensor, np.sin(2 * np.pi * day_of_year / 365))
        cos_doy = torch.full_like(tensor, np.cos(2 * np.pi * day_of_year / 365))

        return torch.cat([tensor, sin_ch, cos_ch, sin_doy, cos_doy], dim=0)

    def __getitem__(self, idx):
        seq_paths, tgt_path, seq_timestamps = self.tuples[idx]
        seq = torch.stack([self._load(p) for p in seq_paths], dim=0)
        tgt = self._load_img(tgt_path)

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


# ==========================================================
# 4. MODEL
# ==========================================================
_GN_GROUPS = 8

def _gn(num_ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(_GN_GROUPS, num_ch)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            _gn(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_op(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size,
            padding=kernel_size // 2, bias=bias
        )

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_seq):
        T, B, C, H, W = input_seq.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=input_seq.device, dtype=input_seq.dtype)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(input_seq[t], h, c)
        return h

class UNetConvLSTM(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 1):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.convlstm = ConvLSTM(input_dim=128, hidden_dim=128, kernel_size=3)
        self.time_proj = nn.Linear(4, 128)
        self.up1 = Up(in_ch=128 + 64, out_ch=64)
        self.up2 = Up(in_ch=64 + 32, out_ch=32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.skip_lstm1 = ConvLSTM(input_dim=32, hidden_dim=32, kernel_size=3)
        self.skip_lstm2 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3)

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        skip1_list, skip2_list, bot_list = [], [], []

        for t in range(T):
            frame = x[:, t]
            s1 = self.inc(frame)
            s2 = self.down1(s1)
            b  = self.down2(s2)
            skip1_list.append(s1)
            skip2_list.append(s2)
            bot_list.append(b)

        skip1_seq = torch.stack(skip1_list, dim=0)
        skip2_seq = torch.stack(skip2_list, dim=0)
        skip1_temporal = self.skip_lstm1(skip1_seq)
        skip2_temporal = self.skip_lstm2(skip2_seq)
        h = self.convlstm(torch.stack(bot_list, 0))

        if target_time is not None:
            time_feat = self.time_proj(target_time)
            h = h + time_feat.view(B, 128, 1, 1)

        d = self.up1(h, skip2_temporal)
        d = self.up2(d, skip1_temporal)
        out = self.outc(d)

        last_frame = x[:, -1, 0:1, :, :]
        return out + last_frame


# ==========================================================
# 5. DATA / CSV / SEQUENCE BUILDING
# ==========================================================
def load_existing_csv():
    if not os.path.exists(csv_path):
        raise RuntimeError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time", "mae_abs", "rmse_abs", "ssim"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_test_tuples_from_pngs():
    pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
    if not pngs:
        raise RuntimeError(f"No PNG files found in {heatmap_folder}")

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

    EXPECTED_INTERVAL = timedelta(hours=3)
    samples_by_year = {}
    total_window = input_len + 4  # 8 input + 4-ahead target (same as training)

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
        targ_idx = all_idxs[input_len + 4 - 1]
        yr = years[targ_idx]
        samples_by_year.setdefault(yr, []).append((seq_idxs, targ_idx))

    test_year = 2024
    test_tuples = [
        (
            [paths[idx] for idx in s],
            paths[t],
            [timestamps[idx] for idx in s],
        )
        for s, t in samples_by_year.get(test_year, [])
    ]

    if not test_tuples:
        raise RuntimeError("No 2024 test tuples found.")
    return test_tuples, dt_re


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
        lon_ticks = np.round(np.linspace(grid_lons[0], grid_lons[-1], min(5, len(grid_lons))), 1)
        lat_ticks = np.round(np.linspace(grid_lats[-1], grid_lats[0], min(5, len(grid_lats))), 1)
        print(f"✅ Grid coords loaded: lat {grid_lats[0]:.2f}→{grid_lats[-1]:.2f}, lon {grid_lons[0]:.2f}→{grid_lons[-1]:.2f}")
        return img_extent, lon_ticks, lat_ticks
    print("⚠️ No grid coords file — axis ticks will be pixel indices")
    return None, None, None


def load_land_mask():
    if os.path.exists(mask_path):
        land_mask_raw = np.load(mask_path)
        print(f"✅ Land mask loaded: {land_mask_raw.sum()} / {land_mask_raw.size} land pixels")
        return land_mask_raw
    print("⚠️ No land mask found — metrics over all pixels")
    return None


# ==========================================================
# 6. PLOTTING
# ==========================================================
def plot_best_sample(tgt_abs, pred_abs, error_abs, seq_ts_best, tgt_ts_best,
                     mae, rmse, mse_val, s, out_path, img_extent, lon_ticks, lat_ticks):
    vmin_shared = min(tgt_abs.min(), pred_abs.min())
    vmax_shared = max(tgt_abs.max(), pred_abs.max())
    err_absmax = max(abs(error_abs.min()), abs(error_abs.max()))
    if err_absmax <= 0:
        err_absmax = 1.0

    suptitle_str = _build_suptitle(seq_ts_best, tgt_ts_best)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.18, wspace=0.38)

    def _add_panel(ax, data, title, cmap, vmin, vmax, cbar_label):
        kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto')
        if img_extent is not None:
            kwargs["extent"] = img_extent
            kwargs["origin"] = "lower"
        im = ax.imshow(data, **kwargs)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Longitude →", fontsize=9)
        ax.set_ylabel("Latitude →", fontsize=9)
        if lon_ticks is not None:
            ax.set_xticks(lon_ticks)
            ax.set_xticklabels([f"{v:.1f}" for v in lon_ticks], fontsize=7.5)
        if lat_ticks is not None:
            ax.set_yticks(lat_ticks)
            ax.set_yticklabels([f"{v:.1f}" for v in lat_ticks], fontsize=7.5)
        ax.tick_params(axis='both', which='both', length=3, width=0.6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    _add_panel(axes[0], tgt_abs,  "Actual",    "gray", vmin_shared, vmax_shared, "2m Temperature (K)")
    _add_panel(axes[1], pred_abs,  "Predicted", "gray", vmin_shared, vmax_shared, "2m Temperature (K)")
    _add_panel(axes[2], error_abs, "Error = Actual - Predicted", "gray", -err_absmax, err_absmax, "Error (K)")

    fig.suptitle(suptitle_str, fontsize=11, fontweight='bold', y=0.96)

    metrics_str = (
        f"MSE: {mse_val:.4f}    "
        f"MAE: {mae:.4f}    "
        f"RMSE: {rmse:.4f}    "
        f"SSIM: {s:.4f}"
    )
    fig.text(
        0.5, 0.04, metrics_str, ha='center', va='center',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#444444', linewidth=1.0, alpha=0.9),
    )

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🖼️  Saved: {os.path.basename(out_path)}")


# ==========================================================
# 7. MAIN
# ==========================================================
def main():
    g_min, g_max = ensure_data_ready()
    temp_range = g_max - g_min
    print(f"📉 Denormalisation Key: Min={g_min:.2f}K, Range={temp_range:.2f}K")

    df = load_existing_csv()
    print(f"✅ Loaded CSV: {os.path.basename(csv_path)} ({len(df)} rows)")
    print(f"✅ CSV columns: {list(df.columns)}")

    best_idx = df["mae_abs"].idxmin()
    best_row = df.loc[best_idx]
    best_time = pd.to_datetime(best_row["time"])
    print(f"🏆 Best sample time: {best_time}")
    print(f"🏆 Lowest MAE in CSV: {best_row['mae_abs']:.6f}")

    test_tuples, dt_re = build_test_tuples_from_pngs()

    match_idx = None
    for i, (_, tgt_path, _) in enumerate(test_tuples):
        m = dt_re.search(os.path.basename(tgt_path))
        if not m:
            continue
        tgt_ts = pd.to_datetime(m.group(1))
        if tgt_ts == best_time:
            match_idx = i
            break

    if match_idx is None:
        raise RuntimeError(f"Could not find a 2024 test sample matching best CSV time: {best_time}")

    print(f"✅ Matching test sample index in 2024 set: {match_idx}")

    if not os.path.exists(model_save_path):
        raise RuntimeError(f"Model file not found: {model_save_path}")

    model = UNetConvLSTM(in_channels=5).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print("✅ Best model weights loaded.")

    land_mask = load_land_mask()
    img_extent, lon_ticks, lat_ticks = get_geo_extent_and_ticks()

    transform = transforms.Compose([transforms.ToTensor()])
    ds = HeatmapSeqFromPaths([test_tuples[match_idx]], transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    seq, tgt, tgt_time = next(iter(loader))
    seq = seq.to(device)
    tgt_time = tgt_time.to(device)

    with torch.no_grad():
        with autocast(enabled=(device.type == "cuda")):
            pred_t = model(seq, tgt_time).cpu().float()

    pred_np = pred_t.numpy().squeeze()
    tgt_np  = tgt.numpy().squeeze()

    pred_np = np.nan_to_num(pred_np, nan=0.0)
    tgt_np  = np.nan_to_num(tgt_np, nan=0.0)

    pred_abs = np.flipud(pred_np * temp_range + g_min)
    tgt_abs  = np.flipud(tgt_np  * temp_range + g_min)
    error_abs = tgt_abs - pred_abs

    if land_mask is not None:
        pred_land = pred_abs[land_mask]
        tgt_land  = tgt_abs[land_mask]
    else:
        pred_land = pred_abs.flatten()
        tgt_land  = tgt_abs.flatten()

    diff = pred_land - tgt_land
    mae  = float(np.mean(np.abs(diff)))
    mse  = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    pred_ssim = pred_np.copy()
    tgt_ssim  = tgt_np.copy()
    if land_mask is not None:
        pred_ssim[~land_mask] = 0.0
        tgt_ssim[~land_mask]  = 0.0

    try:
        s = float(ssim_metric(tgt_ssim, pred_ssim, data_range=1.0, win_size=7))
    except Exception:
        s = 0.0

    seq_ts_best = test_tuples[match_idx][2]
    tgt_path = test_tuples[match_idx][1]
    m = dt_re.search(os.path.basename(tgt_path))
    tgt_ts_best = pd.to_datetime(m.group(1)) if m else best_time

    out_path = os.path.join(eval_out, "best_sample_comparison.png")
    plot_best_sample(
        tgt_abs=tgt_abs,
        pred_abs=pred_abs,
        error_abs=error_abs,
        seq_ts_best=seq_ts_best,
        tgt_ts_best=tgt_ts_best,
        mae=mae,
        rmse=rmse,
        mse_val=mse,
        s=s,
        out_path=out_path,
        img_extent=img_extent,
        lon_ticks=lon_ticks,
        lat_ticks=lat_ticks,
    )

    print("\n" + "=" * 55)
    print("  Best Sample Summary")
    print("=" * 55)
    print(f"  Time  : {best_time}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MSE   : {mse:.4f}")
    print(f"  SSIM  : {s:.4f}")
    print("=" * 55)
    print(f"\n🎉 Output saved in: {eval_out}")


if __name__ == '__main__':
    main()
