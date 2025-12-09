import os
import xarray as xr
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# 👇 LOCAL: base_dir is the Dataset folder
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = Path(script_dir).resolve().parents[2] / "Dataset"
base_dir = str(base_dir)

years = ["2020", "2021", "2022", "2023", "2024"]

nc_files = []

for y in years:
    year_dir = os.path.join(base_dir, y)
    for (
        root,
        dirs,
        files,
    ) in os.walk(year_dir):
        for f in files:
            if f.endswith(".nc"):
                nc_files.append(os.path.join(root, f))

nc_files = sorted(nc_files)
print("Total NetCDF files found:", len(nc_files))
print("Example file:", nc_files[0] if nc_files else "NONE")

all_t2m = []

print("Reading NetCDF files...")
for path in nc_files:
    print("  Reading:", path)
    ds0 = xr.open_dataset(path)

    # --- take only t2m ---
    da = ds0["t2m"]

    # Case A: data uses 'valid_time' as the main time dimension
    if "valid_time" in da.dims and "time" not in da.dims:
        da = da.rename({"valid_time": "time"})

    # Case B: data uses (time, step) and maybe 'valid_time' coord
    if "time" in da.dims and "step" in da.dims:
        # build a 1D time coordinate
        if "valid_time" in ds0:
            vt = ds0["valid_time"]
            new_time = vt.values.reshape(-1)  # flatten (time, step)
        else:
            # fallback: just create an index-based time
            new_time = np.arange(da.sizes["time"] * da.sizes["step"])

        # stack (time, step) → single dimension
        da = da.stack(new_time=("time", "step"))
        da = da.assign_coords(new_time=("new_time", new_time))
        da = da.rename({"new_time": "time"})
        da = da.drop_vars(["step"], errors="ignore")

    # ensure latitude / longitude are sorted
    if "latitude" in da.coords:
        da = da.sortby("latitude")
    if "longitude" in da.coords:
        da = da.sortby("longitude")

    all_t2m.append(da)

# concat everything along the unified 'time' dimension
t2m_all = xr.concat(all_t2m, dim="time")
print("Unified t2m shape:", t2m_all.shape)

# ------------------------------------------------------------------
# HEATMAPS (preview: sea blank / NaN)
# ------------------------------------------------------------------

out_dir = os.path.join(script_dir, "Heatmaps_t2m")
os.makedirs(out_dir, exist_ok=True)


def plot_t2m_heatmap(da, timestamp, save_path=None):
    """
    Preview heatmap: keep NaNs (sea) so they show as blank.
    """
    # keep only real ERA5 grid points (odd indices)
    da = da.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))

    temp = da.values  # contains NaNs over sea
    lats = da["latitude"].values
    lons = da["longitude"].values

    extent = [
        float(lons.min()),
        float(lons.max()),
        float(lats.min()),
        float(lats.max()),
    ]

    plt.figure(figsize=(4, 4))

    cmap = plt.get_cmap("gray")
    try:
        cmap = cmap.copy()
    except Exception:
        cmap = matplotlib.colors.ListedColormap(cmap(np.linspace(0, 1, cmap.N)))
    # sea (NaN) -> white blank region
    cmap.set_bad(color="white")

    im = plt.imshow(
        np.ma.masked_invalid(temp),  # mask NaNs
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )

    # nicer ticks
    num_xticks = 4
    num_yticks = 4
    xticks = np.linspace(extent[0], extent[1], num_xticks)
    yticks = np.linspace(extent[2], extent[3], num_yticks)
    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.xlabel("Longitude   →")
    plt.ylabel("Latitude   ↑")

    ts = np.datetime_as_string(timestamp, unit="h")
    plt.title(f"2 m Temperature (t2m) at {ts}")

    # No colorbar to match paper‐style look (optional).
    # cbar = plt.colorbar(im)
    # cbar.set_label("Temperature (K)")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    # always close, never block
    plt.close()


# preview heatmap from first timestamp (paper-style)
t0 = t2m_all.time.values[0]
da0 = t2m_all.sel(time=t0)
plot_t2m_heatmap(da0, t0, save_path=os.path.join(out_dir, "Preview_t2m.png"))

print("Saved preview heatmap to:", os.path.join(out_dir, "Preview_t2m.png"))


import numpy as np  # keep structure similar

out_dir = os.path.join(script_dir, "Heatmaps_t2m_gray")
os.makedirs(out_dir, exist_ok=True)
print("Saving preview heatmaps to:", out_dir)

times = t2m_all.time.values
print("Total timestamps:", len(times))

for i in range(5):
    t = times[i]
    da_t = t2m_all.isel(time=i)

    ts_str = np.datetime_as_string(t, unit="h").replace(":", "-")
    fname = f"t2m_{i:05d}_{ts_str}.png"
    save_path = os.path.join(out_dir, fname)

    print("  Saving", fname)
    plot_t2m_heatmap(da_t, t, save_path)

print("✅ First 5 preview heatmaps saved.")

import xarray as xr  # original duplicate import
import numpy as np

# 1. keep only the real grid (odd indices)
t2m_sub_raw = t2m_all.isel(latitude=slice(1, None, 2), longitude=slice(1, None, 2))

# array with NaNs kept for sea mask
t2m_arr_raw = t2m_sub_raw.values  # (time, H, W)
times = t2m_sub_raw.time.values
print("Subsampled t2m shape (raw with NaNs):", t2m_arr_raw.shape)

# compute sea mask from NetCDFs: True where ALL timesteps are NaN -> sea
sea_mask = np.isnan(t2m_arr_raw).all(axis=0)  # H x W
land_mask = ~sea_mask
print("Sea mask computed. Land pixels:", int(land_mask.sum()))

# 2. For PNGs we can fill NaNs (only for visualization / model inputs),
#    but we keep sea_mask so we NEVER use those pixels in losses/metrics.
t2m_sub = t2m_sub_raw.interpolate_na(dim="latitude", method="linear")
t2m_sub = t2m_sub.interpolate_na(dim="longitude", method="linear")
t2m_sub = t2m_sub.ffill(dim="latitude").bfill(dim="latitude")
t2m_sub = t2m_sub.ffill(dim="longitude").bfill(dim="longitude")

t2m_arr = t2m_sub.values  # shape: (time, H, W), filled for PNGs
print("Subsampled t2m shape (filled for PNGs):", t2m_arr.shape)

out_dir = os.path.join(script_dir, "Heatmaps_t2m_gray_fast")
os.makedirs(out_dir, exist_ok=True)
print("Saving all grayscale images to:", out_dir)

# global min/max over all time,lat,lon (from filled array)
global_min = np.nanmin(t2m_arr)
global_max = np.nanmax(t2m_arr)
print("Temp range:", global_min, "→", global_max)

from PIL import Image
import numpy as np
from tqdm import tqdm

n_time, H, W = t2m_arr.shape
print("Total timestamps:", n_time)

for i in tqdm(range(n_time)):
    temp = t2m_arr[i]  # (H, W)

    # normalize 0–1
    norm = (temp - global_min) / (global_max - global_min)
    norm = np.clip(norm, 0.0, 1.0)

    # clean NaNs/Infs before cast (should be none, but safe)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

    # to 0–255 uint8
    img_arr = (norm * 255).astype(np.uint8)

    # create greyscale image
    img = Image.fromarray(img_arr, mode="L")  # "L" = 8-bit greyscale

    # filename with timestamp
    ts_str = np.datetime_as_string(times[i], unit="h").replace(":", "-")
    fname = f"t2m_{i:05d}_{ts_str}.png"
    save_path = os.path.join(out_dir, fname)

    img.save(save_path)

print("✅ All greyscale heatmaps saved to:", out_dir)

# Save sea_mask for later
np.save(os.path.join(script_dir, "sea_mask.npy"), sea_mask)
print("Saved sea_mask to:", os.path.join(script_dir, "sea_mask.npy"))

# ------------------------------------------------------------------
# Model1
# Full pipeline: load heatmap PNGs -> train SwinLSTM on 2020-2023
# -> test on 2024 -> metrics + time-series plots
# ------------------------------------------------------------------

import os, re, glob, json
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from skimage.metrics import structural_similarity as ssim_metric
import pandas as pd

# ---------- User params ----------
heatmap_folder = os.path.join(script_dir, "Heatmaps_t2m_gray_fast")  # <- your folder
model_save_path = os.path.join(script_dir, "SwinLSTM_t2m_best.pth")
eval_out = os.path.join(script_dir, "Eval_SwinLSTM_2024")
os.makedirs(eval_out, exist_ok=True)

input_len = 8
img_size = 224
batch_size = 4  # adjust if you have GPU memory
num_epochs = 5  # increase if you want
lr = 1e-4

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device for Model1: {device} (CUDA available: {use_cuda})")

# ---------- collect PNG files and parse timestamps ----------
pngs = sorted(glob.glob(os.path.join(heatmap_folder, "*.png")))
if len(pngs) == 0:
    raise RuntimeError(f"No PNGs found in {heatmap_folder}. Check the path.")
print("Found PNGs:", len(pngs))

# filenames are like: t2m_00000_2020-04-01T00.png
dt_re = re.compile(r".*?_(\d{4}-\d{2}-\d{2}T\d{2})")  # captures YYYY-MM-DDTHH

file_times = []
for p in pngs:
    m = dt_re.search(os.path.basename(p))
    if not m:
        # try alternate pattern with full timestamp before .png
        name = os.path.basename(p)
        # fallback: search for '2020'..'2024'
        yr_m = re.search(r"(20\d{2})", name)
        if yr_m:
            # fallback: use file modified time
            ts = datetime.fromtimestamp(os.path.getmtime(p))
        else:
            raise RuntimeError("Could not parse timestamp from filename: " + p)
    else:
        ts = np.datetime64(m.group(1))
        # convert to python datetime for year extraction
        ts = np.datetime64(m.group(1)).astype("datetime64[m]").astype(object)
    file_times.append((p, ts))

# Build mapping: list sorted by timestamp (should already be sorted by filename)
file_times = sorted(file_times, key=lambda x: x[1])
# create arrays
paths = [ft[0] for ft in file_times]
times = [ft[1] for ft in file_times]
years = [t.year if hasattr(t, "year") else int(str(t)[:4]) for t in times]

print("Example file/time:", paths[0], times[0], "years range:", min(years), max(years))

# ---------- Create sequence samples but only within same year ----------
# For each start index i, we require all files i..i+input_len (inputs)
# and i+input_len (target) to belong to the SAME YEAR.
samples_by_year = {}
for i in range(0, len(paths) - input_len):
    # indices for the sequence and target
    seq_idxs = list(range(i, i + input_len))
    targ_idx = i + input_len
    y_seq = {years[j] for j in seq_idxs + [targ_idx]}
    if len(y_seq) == 1:
        yr = years[targ_idx]
        samples_by_year.setdefault(yr, []).append((seq_idxs, targ_idx))

for yr in sorted(samples_by_year.keys()):
    print(f"Year {yr}: {len(samples_by_year[yr])} samples")

# Ensure train years (2020-2023) exist and test year 2024 exists
train_years = [2020, 2021, 2022, 2023]
test_year = 2024
for y in train_years:
    if y not in samples_by_year:
        print(f"Warning: no samples found for train year {y}")
if test_year not in samples_by_year:
    raise RuntimeError(
        "No samples found for test year 2024 — check filenames/timestamps."
    )

# Build lists of (seq_paths, target_path) for train and test
train_samples = []
for y in train_years:
    train_samples += samples_by_year.get(y, [])
# flatten to paths
train_tuples = [([paths[idx] for idx in s], paths[t]) for s, t in train_samples]
test_tuples = [
    ([paths[idx] for idx in s], paths[t]) for s, t in samples_by_year[test_year]
]

print("Train samples:", len(train_tuples), "Test samples (2024):", len(test_tuples))


# ---------- Dataset class that accepts precomputed sequence path lists ----------
class HeatmapSeqFromPaths(Dataset):
    def __init__(self, tuples, transform=None):
        # tuples: list of ( [seq_path,...], target_path )
        self.tuples = tuples
        self.transform = transform

    def __len__(self):
        return len(self.tuples)

    def _load(self, p):
        img = Image.open(p).convert("L")
        if self.transform:
            return self.transform(img)  # 1xHxW tensor
        return transforms.ToTensor()(img)

    def __getitem__(self, idx):
        seq_paths, tgt_path = self.tuples[idx]
        seq = torch.stack([self._load(p) for p in seq_paths], dim=0)  # T x 1 x H x W
        tgt = self._load(tgt_path)  # 1 x H x W
        return seq, tgt


# transforms (resize + to tensor, keep values in [0,1])
transform = transforms.Compose(
    [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
)

# dataset objects
full_train_ds = HeatmapSeqFromPaths(train_tuples, transform=transform)
test_ds = HeatmapSeqFromPaths(test_tuples, transform=transform)

# chronological split: first part -> train, last 10% -> val
n_train_total = len(full_train_ds)
n_val = max(1, int(0.10 * n_train_total))
n_train = n_train_total - n_val
train_indices = list(range(0, n_train))
val_indices = list(range(n_train, n_train_total))
train_ds = Subset(full_train_ds, train_indices)
val_ds = Subset(full_train_ds, val_indices)

print("Train/Val/Test sizes:", len(train_ds), len(val_ds), len(test_ds))

# IMPORTANT: num_workers=0 for Windows (no multiprocessing issues)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
)

# ---------- sea mask at model resolution ----------
sea_mask_np = np.load(os.path.join(script_dir, "sea_mask.npy"))  # H_orig x W_orig
from PIL import Image as PILImage

mask_img = PILImage.fromarray((~sea_mask_np).astype(np.uint8) * 255)  # land=255
mask_img = mask_img.resize((img_size, img_size), resample=PILImage.NEAREST)
land_mask_resized = np.array(mask_img) > 0  # HxW bool
sea_mask_resized = ~land_mask_resized
land_mask_t = torch.from_numpy(land_mask_resized).to(device)  # HxW


# ---------- Model definition ----------
class SwinLSTM(nn.Module):
    def __init__(self, hidden_dim=256, input_len=8, img_size=224):
        super().__init__()
        self.input_len = input_len
        self.img_size = img_size
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        F = self.encoder.num_features
        self.lstm = nn.LSTM(
            input_size=F, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, img_size * img_size)

    def forward(self, x):
        # x: B x T x 1 x H x W
        B, T, C, H, W = x.shape
        x = x.repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W)
        feats = self.encoder(x)  # (B*T) x F
        feats = feats.view(B, T, -1)  # B x T x F
        _, (h_n, _) = self.lstm(feats)  # h_n: num_layers x B x hidden
        h_last = h_n[-1]  # B x hidden
        out = self.fc(h_last)  # B x (H*W)
        out = out.view(B, 1, self.img_size, self.img_size)
        return out


# instantiate
model = SwinLSTM(hidden_dim=256, input_len=input_len, img_size=img_size).to(device)
criterion = nn.MSELoss(reduction="none")  # we'll mask sea pixels
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("Model params:", sum(p.numel() for p in model.parameters()) / 1e6, "M params")

# ---------- Climatology (simple mean over training targets, normalized units) ----------
print("Computing climatology (mean over training targets)...")
clim_sum = np.zeros((img_size, img_size), dtype=np.float64)
count = 0
for seq, tgt in DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0):
    clim_sum += tgt.numpy()[0, 0]
    count += 1
climatology = (clim_sum / max(1, count)).astype(np.float32)
climatology[sea_mask_resized] = np.nan  # sea -> NaN
np.save(os.path.join(eval_out, "climatology_norm.npy"), climatology)
print("Saved climatology_norm.npy to:", eval_out)

# ---------- Training loop (save best on val, land-only loss) ----------
best_val = float("inf")
best_state = None
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    n_seen = 0
    for seq, tgt in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        seq = seq.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        pred = model(seq)

        diff2 = (pred - tgt) ** 2  # B x 1 x H x W
        mask = land_mask_t.unsqueeze(0).unsqueeze(0).expand_as(diff2)
        masked_vals = diff2.masked_select(mask)
        loss = (
            masked_vals.mean()
            if masked_vals.numel()
            else torch.tensor(0.0, device=device)
        )

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * seq.size(0)
        n_seen += seq.size(0)
    train_loss /= max(1, n_seen)

    # val
    model.eval()
    val_loss = 0.0
    n_val_seen = 0
    with torch.no_grad():
        for seq, tgt in val_loader:
            seq = seq.to(device)
            tgt = tgt.to(device)
            pred = model(seq)
            diff2 = (pred - tgt) ** 2
            mask = land_mask_t.unsqueeze(0).unsqueeze(0).expand_as(diff2)
            masked_vals = diff2.masked_select(mask)
            loss = (
                masked_vals.mean()
                if masked_vals.numel()
                else torch.tensor(0.0, device=device)
            )
            val_loss += loss.item() * seq.size(0)
            n_val_seen += seq.size(0)
    val_loss /= max(1, n_val_seen)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, model_save_path)
        print("  🔥 Saved best model to:", model_save_path)

# ---------- Load best model for evaluation ----------
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
else:
    model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()
print("Loaded best model for evaluation.")

# ---------- Blending: persistence vs model on validation (land-only) ----------
print("Computing blending alpha (model vs persistence) on validation set...")
val_preds = []
val_actuals = []
val_pers = []

with torch.no_grad():
    for seq, tgt in val_loader:
        seq = seq.to(device)  # B x T x 1 x H x W
        tgt = tgt.to(device)  # B x 1 x H x W
        pred = model(seq)  # B x 1 x H x W
        pers = seq[:, -1:, :, :, :]  # last frame

        val_preds.append(pred.cpu().numpy())
        val_actuals.append(tgt.cpu().numpy())
        val_pers.append(pers.cpu().numpy())

if len(val_preds):
    Pm = np.concatenate(val_preds, axis=0)[:, 0]  # N x H x W
    A = np.concatenate(val_actuals, axis=0)[:, 0]  # N x H x W
    Pp = np.concatenate(val_pers, axis=0)[:, 0, 0]  # N x H x W

    mask_flat = land_mask_resized.ravel()
    Pm_flat = Pm.reshape(Pm.shape[0], -1)[:, mask_flat]
    Pp_flat = Pp.reshape(Pp.shape[0], -1)[:, mask_flat]
    A_flat = A.reshape(A.shape[0], -1)[:, mask_flat]

    numer = np.nansum((A_flat - Pp_flat) * (Pm_flat - Pp_flat))
    denom = np.nansum((Pm_flat - Pp_flat) ** 2)
    if denom <= 1e-12:
        alpha = 1.0
    else:
        alpha = float(numer / denom)
    alpha = float(np.clip(alpha, 0.0, 1.0))
else:
    alpha = 1.0

print(f"Blending alpha (model weight) learned on val: {alpha:.4f}")

# ---------- Evaluate on 2024 test set: compute MSE, RMSE, MAE, SSIM per sample (land-only, blended) ----------
mse_list = []
mae_list = []
rmse_list = []
ssim_list = []

with torch.no_grad():
    for seq, tgt in tqdm(test_loader, desc="Evaluating test 2024 (blended)"):
        seq = seq.to(device)  # 1 x T x 1 x H x W
        tgt = tgt.to(device)  # 1 x 1 x H x W
        pred = model(seq)  # 1 x 1 x H x W
        pers = seq[:, -1:, :, :, :]  # 1 x 1 x H x W

        pred_np = pred.cpu().numpy()[0, 0]
        tgt_np = tgt.cpu().numpy()[0, 0]
        pers_np = pers.cpu().numpy()[0, 0, 0]

        # blend in normalized space
        blend_np = alpha * pred_np + (1.0 - alpha) * pers_np

        # land-only metrics
        land_pred = blend_np[land_mask_resized]
        land_true = tgt_np[land_mask_resized]

        mse_v = float(np.mean((land_pred - land_true) ** 2))
        mae_v = float(np.mean(np.abs(land_pred - land_true)))
        rmse_v = float(np.sqrt(mse_v))

        # SSIM: set sea pixels to same values in pred & target
        if ssim_metric is not None:
            a = tgt_np.astype(np.float64).copy()
            p = blend_np.astype(np.float64).copy()
            try:
                p[sea_mask_resized] = a[sea_mask_resized]
            except Exception:
                pass
            try:
                ssim_v = float(ssim_metric(a, p, data_range=1.0))
            except Exception:
                ssim_v = float("nan")
        else:
            ssim_v = float("nan")

        mse_list.append(mse_v)
        mae_list.append(mae_v)
        rmse_list.append(rmse_v)
        ssim_list.append(ssim_v)

# Build time ordering (timestamps for each test sample)
test_times = []
for tup in test_tuples:
    # extract target timestamp from filename
    fn = os.path.basename(tup[1])
    m = dt_re.search(fn)
    if m:
        ts = np.datetime64(m.group(1))
        ts_py = ts.astype("datetime64[m]").astype(object)
    else:
        ts_py = datetime.fromtimestamp(os.path.getmtime(tup[1]))
    test_times.append(ts_py)

# sort results by time and reorder metrics accordingly
order = np.argsort(test_times)
test_times_sorted = [test_times[i] for i in order]
mse_sorted = [mse_list[i] for i in order]
rmse_sorted = [rmse_list[i] for i in order]
mae_sorted = [mae_list[i] for i in order]
ssim_sorted = [ssim_list[i] for i in order]

# ---------- Save per-sample CSV and aggregated metrics ----------
df = pd.DataFrame(
    {
        "timestamp": [str(t) for t in test_times_sorted],
        "mse": mse_sorted,
        "rmse": rmse_sorted,
        "mae": mae_sorted,
        "ssim": ssim_sorted,
    }
)
df.to_csv(os.path.join(eval_out, "swinlstm_test_2024_per_sample.csv"), index=False)

summary = {
    "MSE_mean": float(np.mean(mse_sorted)),
    "MSE_std": float(np.std(mse_sorted)),
    "RMSE_mean": float(np.mean(rmse_sorted)),
    "RMSE_std": float(np.std(rmse_sorted)),
    "MAE_mean": float(np.mean(mae_sorted)),
    "MAE_std": float(np.std(mae_sorted)),
    "SSIM_mean": float(np.mean(ssim_sorted)),
    "SSIM_std": float(np.std(ssim_sorted)),
    "n_test": len(mse_sorted),
    "alpha": alpha,
}
with open(os.path.join(eval_out, "swinlstm_test_2024_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Test summary:", json.dumps(summary, indent=2))
print(
    "Per-sample CSV saved to:",
    os.path.join(eval_out, "swinlstm_test_2024_per_sample.csv"),
)

# ---------- Plot time-series figures as requested ----------
plt.figure(figsize=(12, 4))
plt.plot(test_times_sorted, ssim_sorted, "-", marker=".", linewidth=0.8)
plt.xlabel("Time (2024)")
plt.ylabel("SSIM")
plt.title("Figure 1: Time series of SSIM for test data (2024) of SwinLSTM (blended)")
plt.grid(alpha=0.3)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(eval_out, "Figure1_SSIM_timeseries_2024.png"), dpi=150)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(test_times_sorted, mae_sorted, "-", marker=".", linewidth=0.8)
plt.xlabel("Time (2024)")
plt.ylabel("MAE (normalized)")
plt.title("Figure 2: Time series of MAE for test data (2024) of SwinLSTM (blended)")
plt.grid(alpha=0.3)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(eval_out, "Figure2_MAE_timeseries_2024.png"), dpi=150)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(test_times_sorted, rmse_sorted, "-", marker=".", linewidth=0.8)
plt.xlabel("Time (2024)")
plt.ylabel("RMSE (normalized)")
plt.title("Figure 3: Time series of RMSE for test data (2024) of SwinLSTM (blended)")
plt.grid(alpha=0.3)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(eval_out, "Figure3_RMSE_timeseries_2024.png"), dpi=150)
plt.close()

print("Figures saved to:", eval_out)
