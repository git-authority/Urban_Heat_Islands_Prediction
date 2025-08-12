import os
import numpy as np
from netCDF4 import Dataset
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# -------------- CONFIG --------------
folder_path = "../Dataset/2024"
out_dir = "ConvLSTM_v1"
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4
batch_size = 8
lr = 1e-4
epochs = 100  # adjust as needed
val_split = 0.2
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True  # recommended
SAMPLE_IDX = 0  # which validation sample to visualize (or 'mean')
# flipping options
FLIP_X = False  # flip final heatmaps w.r.t. x-axis (longitude) -- disabled
FLIP_Y = False  # flip final heatmaps w.r.t. y-axis (latitude) -- disabled
plt.rcParams["font.family"] = "Times New Roman"
# -------------- END CONFIG --------------

torch.manual_seed(seed)
np.random.seed(seed)


# ---------------- Dataset (keep NaNs; build sea_mask) ----------------
class NonSlidingMaskDataset(TorchDataset):
    def __init__(self, folder_path, input_len=8, target_offset=4):
        import os, numpy as np
        from netCDF4 import Dataset as ncDataset

        self.frames = []  # will keep NaNs for sea cells
        self.input_len = input_len
        self.target_offset = target_offset

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if not files:
            raise ValueError("No .nc files found")

        for fn in files:
            path = os.path.join(folder_path, fn)
            try:
                ds = ncDataset(path)
                if "t2m" not in ds.variables:
                    ds.close()
                    continue
                var = ds.variables["t2m"]
                arr = np.array(var[:])  # keep whatever fill-values are present
                # detect _FillValue / missing_value and convert to np.nan
                if hasattr(var, "_FillValue"):
                    arr = np.where(arr == var._FillValue, np.nan, arr)
                if hasattr(var, "missing_value"):
                    arr = np.where(arr == var.missing_value, np.nan, arr)
                # append frames (keep NaNs)
                if arr.ndim == 3:
                    for t in range(arr.shape[0]):
                        self.frames.append(arr[t].astype(np.float32))
                elif arr.ndim == 2:
                    self.frames.append(arr.astype(np.float32))
                ds.close()
            except Exception as e:
                print(f"Skipping {fn}: {e}")

        if len(self.frames) == 0:
            raise ValueError("No frames loaded")

        # ensure consistent shape
        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        # determine sea mask: pixels that are NaN across all times (True = sea)
        stacked = np.stack(self.frames, axis=0)  # (T,H,W)
        self.sea_mask = np.isnan(stacked).all(axis=0)  # (H,W)

        # build non-overlapping starts
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
            raise ValueError("Not enough frames for chosen input_len/target_offset")
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len
        inp = np.stack(self.frames[s:e], axis=0)  # (C,H,W) with NaNs in sea
        tgt = self.frames[e - 1 + self.target_offset]  # (H,W) with NaNs in sea

        # For training: fill NaNs in each time slice with that frame's land-mean so model can train.
        # But keep sea_mask externally so plotting can mask them white.
        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~self.sea_mask]
            if land_vals.size == 0:
                fill = 0.0
            else:
                fill = float(np.nanmean(land_vals))
            frame_f = np.where(np.isnan(frame), fill, frame)
            inp_filled[i] = frame_f

        land_vals_tgt = tgt[~self.sea_mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
        )


# ---------------- ConvLSTM (unchanged) ----------------
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
        x_time = x.unsqueeze(2)
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
    # cb is a matplotlib.colorbar.Colorbar
    cb.ax.tick_params(labelsize=9)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))


# ---------------- Prepare data ----------------
print("Loading dataset...")
print(f"Using device: {device}")
if torch.cuda.is_available():
    try:
        print(f"CUDA available. Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    except Exception:
        print("CUDA is available but failed to query device name.")
else:
    print("CUDA not available; using CPU.")


dataset = NonSlidingMaskDataset(
    folder_path, input_len=input_len, target_offset=target_offset
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

# model
model = SameSizeConvLSTM(in_channels=input_len, hidden_dim=64, num_layers=2).to(device)
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)
criterion = nn.MSELoss()

# training (no model saved)
train_losses = []
val_losses = []
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
        loss = criterion(out, y)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item() * X.size(0)
        seen += X.size(0)
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
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
            lv = criterion(outv, yv)
            vr += lv.item() * Xv.size(0)
            vseen += Xv.size(0)
    val_loss = vr / max(1, vseen)
    val_losses.append(val_loss)
    sched.step(val_loss)
    print(f"Epoch {epoch:03d} Train={train_loss:.6f} Val={val_loss:.6f}")

# save train/val loss plot
loss_fig_path = os.path.join(out_dir, "Train_Val_Loss.png")
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_fig_path, dpi=200)
plt.close()
print("Saved loss curve to:", loss_fig_path)

# inference on val set (convert back from norm)
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


# load lat/lon for extent and potentially flip lat


def find_first_nc(folder):
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".nc"):
            return os.path.join(folder, fn)
    raise FileNotFoundError


nc0 = find_first_nc(folder_path)
ds0 = Dataset(nc0)
lats = ds0.variables["latitude"][:].astype(float)
lons = ds0.variables["longitude"][:].astype(float)
ds0.close()

# determine image origin based on latitude ordering (do NOT flip data arrays)
# We avoid flipping arrays entirely — instead set the `origin` parameter for imshow.
origin = "lower"
if lats[0] > lats[-1]:
    # lat array is descending (north->south). Using origin='upper' will render
    # the image without modifying the array but with correct geographic orientation.
    origin = "upper"

# select sample to display
if SAMPLE_IDX == "mean":
    actual_full = np.mean(actuals, axis=0)[0]
    pred_full = np.mean(preds, axis=0)[0]
else:
    idx = int(SAMPLE_IDX)
    idx = max(0, min(idx, preds.shape[0] - 1))
    actual_full = actuals[idx, 0]
    pred_full = preds[idx, 0]

# sea mask (do not flip)
sea_mask_plot = dataset.sea_mask

# Do NOT apply any additional flips to the arrays (user requested no flips).

# compute extent normally — do not swap min/max values.
lon_min = float(lons.min())
lon_max = float(lons.max())
lat_min = float(lats.min())
lat_max = float(lats.max())
extent = [lon_min, lon_max, lat_min, lat_max]

# mask sea area (True -> sea)
mask = sea_mask_plot
actual_masked = np.ma.masked_where(mask, actual_full)
pred_masked = np.ma.masked_where(mask, pred_full)
error_masked = np.ma.masked_where(mask, actual_full - pred_full)

# set colormaps and make mask white
cmap_temp = plt.get_cmap("gray").copy()
cmap_temp.set_bad("white")
cmap_err = plt.get_cmap("gray").copy()
cmap_err.set_bad("white")

# vmin,vmax from valid land pixels
combined = np.concatenate(
    [
        actual_masked.filled(np.nan).ravel(),
        pred_masked.filled(np.nan).ravel(),
    ]
)
combined = combined[~np.isnan(combined)]
if combined.size == 0:
    raise RuntimeError("No valid land pixels to plot.")
vmin = float(np.nanmin(combined))
vmax = float(np.nanmax(combined))
err_abs = float(np.nanmax(np.abs(error_masked.filled(0.0))))

# Plot full domain (no crop) with sea white; axis labels as requested
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
axes[0].set_title("Actual", fontsize=12)
axes[0].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[0].tick_params(labelsize=9)

im1 = axes[1].imshow(
    pred_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[1].set_title("Predicted", fontsize=11)
axes[1].set_xlabel("Longitude   →", fontsize=11)
axes[1].tick_params(labelsize=9)

im2 = axes[2].imshow(
    error_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_err,
    vmin=-err_abs,
    vmax=err_abs,
    interpolation="nearest",
)
axes[2].set_title("Error (Actual - Predicted)", fontsize=11)
axes[2].set_xlabel("Longitude   →", fontsize=11)
axes[2].tick_params(labelsize=9)

# colorbars using make_axes_locatable so tight_layout works
for ax, im in zip(axes, [im0, im1, im2]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    if im is im2:
        cbar.set_label("Error (units)", fontsize=10)
        pretty_cb(cbar, fmt="%.3f")
    else:
        cbar.set_label("2m Temperature (units)", fontsize=10)
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
save_path = os.path.join(out_dir, "Actual_Predicted_Error.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved plot to:", save_path)
print("Done.")
