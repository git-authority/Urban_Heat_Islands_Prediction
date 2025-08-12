import os
import numpy as np
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# ---------------- CONFIG ----------------
folder_path = "../Dataset/2024"  # as provided
out_dir = "ConvLSTM_v2"
os.makedirs(out_dir, exist_ok=True)

variable_name = "t2m"
input_len = 8
target_offset = 4
batch_size = 8
lr = 1e-4
epochs = 100  # increase if you want longer training
val_split = 0.2
seed = 42
USE_NORMALIZATION = True
SAMPLE_IDX = 0  # index to visualize (or 'mean')
early_stop_patience = 15
hidden_dim = 64
num_layers = 2

plt.rcParams["font.family"] = "Times New Roman"
# ----------------------------------------

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    try:
        print("CUDA available. Device:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())
    except Exception:
        print("CUDA available but failed to query device name")
else:
    print("CUDA not available — using CPU")


# ---------------- Dataset ----------------
class NonSlidingMaskDataset(TorchDataset):
    """Reads all .nc files in folder_path and builds non-overlapping windows.
    Keeps NaNs for sea in original frames; dataset returns inputs with NaNs filled
    (per-frame land-mean) so model can train, and external sea_mask is available
    for plotting and masked loss computation.
    """

    def __init__(self, folder_path, variable_name, input_len=8, target_offset=4):
        import os
        import numpy as np
        from netCDF4 import Dataset as ncDataset

        self.frames = []
        self.input_len = input_len
        self.target_offset = target_offset

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if not files:
            raise ValueError(f"No .nc files found in {folder_path}")

        for fn in files:
            path = os.path.join(folder_path, fn)
            try:
                ds = ncDataset(path)
                if variable_name not in ds.variables:
                    ds.close()
                    continue
                var = ds.variables[variable_name]
                arr = np.array(var[:])  # could be (T,H,W) or (H,W)
                # convert fill values to np.nan when possible
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
            raise ValueError("No frames loaded from NetCDF files")

        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent frame shapes: {shapes}")

        self.H, self.W = self.frames[0].shape

        stacked = np.stack(self.frames, axis=0)  # (T,H,W)
        self.sea_mask = np.isnan(stacked).all(axis=0)  # True == sea

        # non-overlapping starts
        stride = self.input_len + self.target_offset
        starts = []
        s = 0
        while True:
            end = s + self.input_len
            target_idx = end - 1 + self.target_offset
            if target_idx < len(self.frames):
                starts.append(s)
                s += stride
            else:
                break
        if len(starts) == 0:
            raise ValueError("Not enough frames for input_len/target_offset")
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len
        inp = np.stack(self.frames[s:e], axis=0)  # (C,H,W) with NaNs
        tgt = self.frames[e - 1 + self.target_offset]  # (H,W) with NaNs

        # fill NaNs per-frame with land mean for training stability
        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~self.sea_mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            frame_f = np.where(np.isnan(frame), fill, frame)
            inp_filled[i] = frame_f

        land_vals_tgt = tgt[~self.sea_mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        # return (C,H,W) and (1,H,W) tensors
        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
        )


# ---------------- ConvLSTM model ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad
        )

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
        return hnext, cnext

    def init_hidden(self, b, spatial, device):
        H, W = spatial
        return (
            torch.zeros(b, self.hidden_dim, H, W, device=device),
            torch.zeros(b, self.hidden_dim, H, W, device=device),
        )


class SameSizeConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_len = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(ConvLSTMCell(in_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.final = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device
        x_time = x.unsqueeze(2)  # (B,C,1,H,W)
        hiddens = [l.init_hidden(B, (H, W), device) for l in self.layers]
        last = None
        for t in range(C):
            frame = x_time[:, t, :, :, :]
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                hiddens[li] = (hnext, cnext)
                inp = hnext
            last = inp
        out = self.final(last)
        return out


# ---------------- Helpers ----------------
def compute_norm(dataset, train_idx):
    s = 0.0
    ss = 0.0
    cnt = 0
    for i in train_idx:
        X, y = dataset[i]
        arr = torch.cat([X.view(-1), y.view(-1)], dim=0).numpy()
        s += arr.sum()
        ss += (arr**2).sum()
        cnt += arr.size
    mean = s / cnt
    var = ss / cnt - mean**2
    std = np.sqrt(max(var, 1e-12))
    return float(mean), float(std)


def pretty_cb(cb, fmt="%.2f"):
    cb.ax.tick_params(labelsize=9)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))


# ---------------- Prepare data ----------------
print("Loading dataset...")
dataset = NonSlidingMaskDataset(
    folder_path, variable_name, input_len=input_len, target_offset=target_offset
)
n = len(dataset)
indices = np.arange(n)
np.random.seed(seed)
np.random.shuffle(indices)
split = int(np.floor(val_split * n))
val_idx = indices[:split]
train_idx = indices[split:]
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Samples: total={n}, train={len(train_set)}, val={len(val_set)}")
sea_mask = dataset.sea_mask  # (H,W) True==sea

# normalization
norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    print("Computing normalization from training set...")
    norm_mean, norm_std = compute_norm(dataset, train_idx)
    print("norm mean, std:", norm_mean, norm_std)

# Model, optimizer, scheduler
model = SameSizeConvLSTM(
    in_channels=input_len, hidden_dim=hidden_dim, num_layers=num_layers
).to(device)
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)
criterion = nn.SmoothL1Loss(reduction="none")  # we will apply mask and reduce manually

# prepare land mask tensor (1,1,H,W)
land_mask_np = (~sea_mask).astype(np.float32)
land_mask = torch.from_numpy(land_mask_np).unsqueeze(0).unsqueeze(0).to(device)
land_pixels = float(land_mask.sum().item())
print(f"Land pixels: {int(land_pixels)}")

# Colormaps
cmap_temp = plt.get_cmap("gray").copy()
cmap_temp.set_bad("white")
cmap_err = plt.get_cmap("gray").copy()
cmap_err.set_bad("white")

# ---------------- Training ----------------
train_losses = []
val_losses = []
best_val = float("inf")
early_stop_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    run = 0.0
    seen = 0
    pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{epochs}")
    for X, y in pbar:
        X = X.to(device)
        y = y.to(device)
        if USE_NORMALIZATION:
            X = (X - norm_mean) / norm_std
            y = (y - norm_mean) / norm_std
        opt.zero_grad()
        out = model(X)
        if out.shape != y.shape:
            out = F.interpolate(
                out, size=y.shape[2:], mode="bilinear", align_corners=False
            )
        loss_map = criterion(out, y)  # (B,1,H,W)
        mask_b = land_mask.repeat(X.size(0), 1, 1, 1)
        loss = (loss_map * mask_b).sum() / (mask_b.sum() + 1e-12)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item() * X.size(0)
        seen += X.size(0)
        pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})
    train_loss = run / max(1, seen)
    train_losses.append(train_loss)

    # validation
    model.eval()
    vr = 0.0
    vseen = 0
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv = Xv.to(device)
            yv = yv.to(device)
            if USE_NORMALIZATION:
                Xv = (Xv - norm_mean) / norm_std
                yv = (yv - norm_mean) / norm_std
            outv = model(Xv)
            if outv.shape != yv.shape:
                outv = F.interpolate(
                    outv, size=yv.shape[2:], mode="bilinear", align_corners=False
                )
            loss_map_v = criterion(outv, yv)
            mask_bv = land_mask.repeat(Xv.size(0), 1, 1, 1)
            lv = (loss_map_v * mask_bv).sum() / (mask_bv.sum() + 1e-12)
            vr += lv.item() * Xv.size(0)
            vseen += Xv.size(0)
    val_loss = vr / max(1, vseen)
    val_losses.append(val_loss)
    sched.step(val_loss)
    print(f"Epoch {epoch:03d} Train={train_loss:.6f} Val={val_loss:.6f}")

    # save best model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(out_dir, "best_model.pth"),
        )
        print(f"Saved best model (val {val_loss:.6f})")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(
                f"Early stopping after {epoch} epochs (no improvement for {early_stop_patience} epochs)"
            )
            break

# save train/val loss plot
loss_fig_path = os.path.join(out_dir, "Train_Val_Loss.png")
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (masked SmoothL1)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_fig_path, dpi=200)
plt.close()
print("Saved loss curve to:", loss_fig_path)

# ---------------- Inference on val set ----------------
model.eval()
all_preds = []
all_actuals = []
with torch.no_grad():
    for Xv, yv in val_loader:
        Xv = Xv.to(device)
        yv = yv.to(device)
        Xv_in = (Xv - norm_mean) / norm_std if USE_NORMALIZATION else Xv
        outv = model(Xv_in)
        if outv.shape != yv.shape:
            outv = F.interpolate(
                outv, size=yv.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            outv = outv * norm_std + norm_mean
        all_preds.append(outv.cpu().numpy())
        all_actuals.append(yv.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)
actuals = np.concatenate(all_actuals, axis=0)

# metrics (raw units)
diffs = actuals - preds
mse = float(np.nanmean(diffs**2))
mae = float(np.nanmean(np.abs(diffs)))
rmse = float(np.sqrt(mse))
print("VAL METRICS:", mse, mae, rmse)

# ---------------- Load lat/lon for plotting ----------------
nc0 = None
for fn in sorted(os.listdir(folder_path)):
    if fn.endswith(".nc"):
        nc0 = os.path.join(folder_path, fn)
        break
if nc0 is None:
    raise FileNotFoundError("No .nc file for lat/lon lookup")

ds0 = Dataset(nc0)
# try common variable names
lat_names = [k for k in ["latitude", "lat", "y"] if k in ds0.variables]
lon_names = [k for k in ["longitude", "lon", "x"] if k in ds0.variables]
if not lat_names or not lon_names:
    raise RuntimeError("Latitude/Longitude variables not found in sample .nc")
lats = ds0.variables[lat_names[0]][:].astype(float)
lons = ds0.variables[lon_names[0]][:].astype(float)
ds0.close()

# determine origin for imshow based on lat ordering
origin = "lower"
if lats[0] > lats[-1]:
    origin = "upper"

# pick sample for visualization
if SAMPLE_IDX == "mean":
    actual_full = np.mean(actuals, axis=0)[0]
    pred_full = np.mean(preds, axis=0)[0]
else:
    idx = int(SAMPLE_IDX)
    idx = max(0, min(idx, preds.shape[0] - 1))
    actual_full = actuals[idx, 0]
    pred_full = preds[idx, 0]

sea_mask_plot = dataset.sea_mask  # unchanged, no flips
mask = sea_mask_plot
actual_masked = np.ma.masked_where(mask, actual_full)
pred_masked = np.ma.masked_where(mask, pred_full)
error_masked = np.ma.masked_where(mask, actual_full - pred_full)

# colormaps (re-assign for clarity)
cmap_temp = plt.get_cmap("gray").copy()
cmap_temp.set_bad("white")
cmap_err = plt.get_cmap("gray").copy()
cmap_err.set_bad("white")

combined = np.concatenate(
    [actual_masked.filled(np.nan).ravel(), pred_masked.filled(np.nan).ravel()]
)
combined = combined[~np.isnan(combined)]
if combined.size == 0:
    raise RuntimeError("No valid land pixels to plot")
vmin = float(np.nanmin(combined))
vmax = float(np.nanmax(combined))
err_abs = float(np.nanmax(np.abs(error_masked.filled(0.0))))

# extent for imshow: lon_min, lon_max, lat_min, lat_max
extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

im0 = axes[0].imshow(
    actual_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[0].set_title("Actual")
axes[0].set_ylabel("Latitude   →")
axes[0].set_xlabel("Longitude   →")

im1 = axes[1].imshow(
    pred_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[1].set_title("Predicted")
axes[1].set_xlabel("Longitude   →")

im2 = axes[2].imshow(
    error_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_err,
    vmin=-err_abs,
    vmax=err_abs,
    interpolation="nearest",
)
axes[2].set_title("Error (Actual - Predicted)")
axes[2].set_xlabel("Longitude   →")

for ax, im in zip(axes, [im0, im1, im2]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    if im is im2:
        cbar.set_label("Error (units)")
        pretty_cb(cbar, fmt="%.3f")
    else:
        cbar.set_label("2m Temperature (units)")
        pretty_cb(cbar, fmt="%.2f")

metrics_text = f"MSE: {mse:.6f} MAE: {mae:.6f} RMSE: {rmse:.6f}"
fig.text(
    0.02,
    0.02,
    metrics_text,
    fontsize=10,
    va="bottom",
    ha="left",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
)

plt.tight_layout(rect=[0, 0.05, 0.98, 0.95])
plot_path = os.path.join(out_dir, "Actual_Predicted_Error.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved plot to:", plot_path)
print("Done.")
