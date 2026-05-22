
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

# CHANGE THIS if your NetCDF dataset is elsewhere
base_dir = r"../../../../../Dataset"

# This script should sit in the same folder as:
#   ConvLSTM_t2m_best.pth
#   absolute_results.csv
# Optional (will be created if missing):
#   dataset_stats.json
#   land_mask.npy
#   grid_coords.json
#   Heatmaps_t2m_gray_11x11/
#   Eval_ConvLSTM_2024/
script_dir = os.path.dirname(os.path.abspath(__file__))

heatmap_folder  = os.path.join(script_dir, "Heatmaps_t2m_gray_11x11")
eval_out        = os.path.join(script_dir, "Eval_ConvLSTM_2024")
model_save_path = os.path.join(script_dir, "ConvLSTM_t2m_best.pth")
stats_path      = os.path.join(script_dir, "dataset_stats.json")
mask_path       = os.path.join(script_dir, "land_mask.npy")
coords_path     = os.path.join(script_dir, "grid_coords.json")
csv_path        = os.path.join(script_dir, "absolute_results.csv")

os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(eval_out, exist_ok=True)

input_len     = 8
target_offset = 4

img_size    = 11
IN_CHANNELS = 3
batch_size  = 1
HIDDEN_DIMS = (32, 64, 64)
kernel_size = 3
dropout_p   = 0.02

# If True, regenerate PNGs even if they exist
FORCE_REGENERATE_PNGS = True

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


def _build_suptitle(seq_timestamps, tgt_timestamp) -> str:

    t_start = seq_timestamps[0]
    t_end   = seq_timestamps[-1]

    start_date = _fmt_date(t_start)
    start_time = _fmt_time(t_start)

    end_date = _fmt_date(t_end)
    end_time = _fmt_time(t_end)

    out_date = _fmt_date(tgt_timestamp)
    out_time = _fmt_time(tgt_timestamp)

    # Same date
    if start_date == end_date:
        input_part = (
            f"Input: {start_date} | "
            f"{start_time} - {end_time}"
        )

    # Crosses midnight
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


def load_grayscale_image(path):
    img = Image.open(path).convert("L")
    return np.array(img).astype(np.float32) / 255.0


# ==========================================================
# 2. DATA PREP / PNG GENERATION
# ==========================================================

def ensure_data_ready():
    """
    Ensures:
      - dataset_stats.json exists
      - grid_coords.json exists
      - land_mask.npy exists
      - heatmap PNGs exist
    """
    nc_files = _find_nc_files(base_dir)
    if not nc_files:
        raise RuntimeError(f"❌ No NetCDF files found in {base_dir}")

    print(f"✅ Found {len(nc_files)} NetCDF files")

    # ---- Global min/max ----
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
        g_min, g_max = stats['min'], stats['max']
        print(f"✅ Stats loaded: Min={g_min:.2f}K, Max={g_max:.2f}K")

    # ---- Read and concatenate ----
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

            # ==========================================================
            # RECONSTRUCT EXACT 11x11 TRAINING GRID
            # ==========================================================
            # Your original model was trained on an already-cropped
            # 11x11 geographical subset.
            #
            # grid_coords.json contains the exact latitude/longitude
            # points used during training.
            #
            # We recreate that exact subset here.
            # ==========================================================

            if os.path.exists(coords_path):

                with open(coords_path, 'r') as cf:
                    coords = json.load(cf)

                # Stored lats were flipped for plotting.
                # Reverse back to ascending order.
                target_lats = np.array(coords['lats'])[::-1]
                target_lons = np.array(coords['lons'])

                # Select exact training coordinates
                da = da.sel(
                    latitude=target_lats,
                    longitude=target_lons
                )

            all_t2m.append(da)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not all_t2m:
        raise RuntimeError("No usable NetCDF files could be read.")

    t2m_all = xr.concat(all_t2m, dim='time')
    t2m_arr = t2m_all.values
    times   = t2m_all.time.values

    if t2m_arr.shape[1:] != (11, 11):
        raise ValueError(f"Expected native grid shape (11,11), got {t2m_arr.shape[1:]}")

    print(f"✅ Native spatial shape: {t2m_arr.shape[1]}×{t2m_arr.shape[2]}")

    # ---- Save grid coordinates ----
    if not os.path.exists(coords_path):
        try:
            raw_lats = t2m_all.latitude.values
            raw_lons = t2m_all.longitude.values
            lats_flipped = np.flipud(raw_lats).tolist()
            lons_list = raw_lons.tolist()
            with open(coords_path, 'w') as f:
                json.dump({'lats': lats_flipped, 'lons': lons_list}, f)
            print(f"✅ Grid coords saved: lat {lats_flipped[0]:.2f}→{lats_flipped[-1]:.2f}, "
                  f"lon {lons_list[0]:.2f}→{lons_list[-1]:.2f}")
        except Exception as e:
            print(f"⚠️ Could not save grid coords: {e}")
    else:
        print("✅ Grid coords file already exists")

    # ---- Save land mask ----
    if not os.path.exists(mask_path):
        land_mask_raw = None
        for i in range(len(t2m_arr)):
            if not np.all(np.isnan(t2m_arr[i])):
                land_mask_raw = np.flipud(~np.isnan(t2m_arr[i]))
                break
        if land_mask_raw is not None:
            np.save(mask_path, land_mask_raw)
            print(f"✅ Land mask saved ({land_mask_raw.shape}): "
                  f"{land_mask_raw.sum()} land / {land_mask_raw.size} total pixels")
    else:
        print("✅ Land mask file already exists")

    # ---- Save heatmap PNGs ----
    existing_pngs = glob.glob(os.path.join(heatmap_folder, "*.png"))
    if FORCE_REGENERATE_PNGS or len(existing_pngs) == 0:
        print("⚠️ Generating 11×11 PNGs...")
        for i in tqdm(range(len(t2m_arr)), desc="Saving PNGs"):
            if pd.Timestamp(times[i]).hour % 3 != 0:
                continue
            temp = np.flipud(t2m_arr[i])
            norm = np.clip((temp - g_min) / (g_max - g_min), 0.0, 1.0)
            norm = np.nan_to_num(norm, nan=0.0)
            img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
            ts_str = np.datetime_as_string(times[i], unit='h').replace(":", "-")
            img.save(os.path.join(heatmap_folder, f"t2m_{i:05d}_{ts_str}.png"))
        print(f"✅ PNGs saved in: {heatmap_folder}")
    else:
        print(f"✅ Found {len(existing_pngs)} PNGs. Skipping PNG regeneration.")

    return g_min, g_max


# ==========================================================
# 3. MODEL
# ==========================================================

class HeatmapSeqFromPaths(Dataset):
    """
    Returns:
        seq : (input_len, 3, 11, 11)
        tgt : (1, 11, 11)
    """
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
    def _time_encoding(ts) -> tuple:
        hour = ts.hour + getattr(ts, 'minute', 0) / 60.0
        angle = 2.0 * math.pi * hour / 24.0
        return math.sin(angle), math.cos(angle)

    def __getitem__(self, idx):
        seq_paths, tgt_path, seq_timestamps = self.tuples[idx]
        frames = [self._load_img(p) for p in seq_paths]
        tgt = self._load_img(tgt_path)

        seq_3ch = []
        for frame, ts in zip(frames, seq_timestamps):
            sin_val, cos_val = self._time_encoding(ts)
            sin_map = torch.full_like(frame, sin_val)
            cos_map = torch.full_like(frame, cos_val)
            seq_3ch.append(torch.cat([frame, sin_map, cos_map], dim=0))

        seq = torch.stack(seq_3ch, dim=0)
        return seq, tgt


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout_p=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)
        self.dropout_p = dropout_p
        num_groups = max(g for g in range(1, 9) if hidden_dim % g == 0)
        self.gn = nn.GroupNorm(num_groups, hidden_dim)

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
    def __init__(self, in_channels=3, hidden_dims=(32, 64, 64), kernel_size=3, dropout_p=0.05):
        super().__init__()
        in_dims = [in_channels] + list(hidden_dims[:-1])

        self.layers = nn.ModuleList([
            ConvLSTMCell(in_dims[i], hidden_dims[i], kernel_size, dropout_p)
            for i in range(len(hidden_dims))
        ])

        self.res_projs = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            in_d = in_dims[i]
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
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + self.res_projs[li - 1](inp)
                hiddens[li] = (hnext, cnext)
                inp = hnext

        return torch.sigmoid(self.refine(inp))


# ==========================================================
# 4. CSV / SEQUENCE BUILDING
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
    """
    Reconstruct the 2024 test tuples from PNG filenames.
    """
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

    if not file_times:
        raise RuntimeError("Could not parse timestamps from PNG filenames.")

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
            pd.Timestamp(timestamps[all_idxs[k + 1]]) - pd.Timestamp(timestamps[all_idxs[k]]) != EXPECTED_INTERVAL
            for k in range(len(all_idxs) - 1)
        ):
            continue

        y_set = {years[j] for j in all_idxs}
        if len(y_set) != 1:
            continue

        seq_idxs = all_idxs[:input_len]
        targ_idx = all_idxs[input_len + target_offset - 1]
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


# ==========================================================
# 5. PLOTTING HELPERS
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

        lon_ticks = np.round(np.linspace(grid_lons[0], grid_lons[-1], min(5, len(grid_lons))), 1)
        lat_ticks = np.round(np.linspace(grid_lats[-1], grid_lats[0], min(5, len(grid_lats))), 1)

        print(f"✅ Grid coords loaded: lat {grid_lats[0]:.2f}→{grid_lats[-1]:.2f}, "
              f"lon {grid_lons[0]:.2f}→{grid_lons[-1]:.2f}")
        return img_extent, lon_ticks, lat_ticks

    print("⚠️ No grid coords file — axis ticks will be pixel indices")
    return None, None, None


def load_land_mask():
    if os.path.exists(mask_path):
        land_mask_raw = np.load(mask_path)
        if land_mask_raw.shape != (img_size, img_size):
            _mimg = Image.fromarray(land_mask_raw.astype(np.uint8) * 255).convert("L")
            _mimg = _mimg.resize((img_size, img_size), Image.NEAREST)
            land_mask = np.array(_mimg) > 0
        else:
            land_mask = land_mask_raw
        print(f"✅ Land mask loaded: {land_mask.sum()} / {land_mask.size} land pixels")
        return land_mask

    print("⚠️ No land mask found — metrics over all pixels")
    return None


def plot_best_sample(tgt_abs, pred_abs, error_abs, seq_ts_best, tgt_ts_best,
                     mae, rmse, mse_val, s, out_path, img_extent, lon_ticks, lat_ticks):
    vmin_shared = min(tgt_abs.min(), pred_abs.min())
    vmax_shared = max(tgt_abs.max(), pred_abs.max())
    err_absmax = max(abs(error_abs.min()), abs(error_abs.max()))
    if err_absmax <= 0:
        err_absmax = 1.0

    suptitle_str = _build_suptitle(seq_ts_best, tgt_ts_best)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.18, wspace=0.38)

    def _add_panel(ax, data, title, cmap, vmin, vmax, cbar_label):
        kwargs = dict(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            aspect='auto',
        )
        if img_extent is not None:
            kwargs["extent"] = img_extent
            kwargs["origin"] = "upper"

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
        0.5, 0.04,
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
            alpha=0.9,
        ),
    )

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🖼️  Saved: {os.path.basename(out_path)}")


# ==========================================================
# 6. MAIN
# ==========================================================

def main():
    # ---- generate / load necessary files ----
    g_min, g_max = ensure_data_ready()
    temp_range = g_max - g_min
    print(f"📉 Denormalisation Key: Min={g_min:.2f}K, Range={temp_range:.2f}K")

    # ---- load existing CSV (already created by you) ----
    df = load_existing_csv()
    print(f"✅ Loaded CSV: {os.path.basename(csv_path)} ({len(df)} rows)")
    print(f"✅ CSV columns: {list(df.columns)}")

    # ---- best sample by MAE ----
    best_idx = df["mae_abs"].idxmin()
    best_row = df.loc[best_idx]
    best_time = pd.to_datetime(best_row["time"])

    print(f"🏆 Best sample time: {best_time}")
    print(f"🏆 Lowest MAE in CSV: {best_row['mae_abs']:.6f}")

    # ---- reconstruct test tuples from PNGs ----
    test_tuples, dt_re = build_test_tuples_from_pngs()

    # locate the tuple whose target timestamp matches the best CSV row
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
        raise RuntimeError(
            f"Could not find a 2024 test sample matching best CSV time: {best_time}"
        )

    print(f"✅ Matching test sample index in 2024 set: {match_idx}")

    # ---- model ----
    if not os.path.exists(model_save_path):
        raise RuntimeError(f"Model file not found: {model_save_path}")

    model = ResidualConvLSTMWithRefine(
        in_channels=IN_CHANNELS,
        hidden_dims=HIDDEN_DIMS,
        kernel_size=kernel_size,
        dropout_p=dropout_p,
    ).to(device)

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print("✅ Best model weights loaded.")

    # ---- support files ----
    land_mask = load_land_mask()
    img_extent, lon_ticks, lat_ticks = get_geo_extent_and_ticks()

    # ---- run only the best sample ----
    transform = transforms.Compose([transforms.ToTensor()])
    ds = HeatmapSeqFromPaths([test_tuples[match_idx]], transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    seq, tgt = next(iter(loader))
    seq = seq.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred_t = model(seq).cpu().float()

    pred_np = pred_t.numpy().squeeze()
    tgt_np  = tgt.numpy().squeeze()

    pred_np = np.nan_to_num(pred_np, nan=0.0)
    tgt_np  = np.nan_to_num(tgt_np, nan=0.0)

    pred_abs = pred_np * temp_range + g_min
    tgt_abs  = tgt_np  * temp_range + g_min
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
