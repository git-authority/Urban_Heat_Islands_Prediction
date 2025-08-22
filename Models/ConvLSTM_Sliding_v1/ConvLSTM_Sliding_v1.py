# improved_conv_lstm_sampled_with_ssim_on_plot.py
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
folder_path = "../../Dataset/2024"
out_dir = "ConvLSTM_v1_updated"
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4  # in the downsampled timeline (kept same)
SAMPLE_STEP = 3  # <-- downsample the raw frames by 3 (every 3 hours)
batch_size = 8
lr = 1e-4
epochs = 100
val_split = 0.2
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True
SAMPLE_IDX = 0
plt.rcParams["font.family"] = "Times New Roman"
# -------------- END CONFIG --------------

torch.manual_seed(seed)
np.random.seed(seed)


# ---------------- Dataset (sliding window applied) ----------------
class SlidingMaskDataset(TorchDataset):
    def __init__(self, folder_path, input_len=8, target_offset=4, sample_step=1):
        import os, numpy as np
        from netCDF4 import Dataset as ncDataset

        self.frames = []
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

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
                arr = np.array(var[:])
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
            raise ValueError("No frames loaded")

        # --- APPLY DOWNSAMPLING HERE ---
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]

        # ensure consistent shape
        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)

        # ---------------- SLIDING WINDOW CHANGE ----------------
        starts = []
        for s in range(len(self.frames) - input_len - target_offset + 1):
            starts.append(s)
        if len(starts) == 0:
            raise ValueError("Not enough frames for chosen input_len/target_offset")
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len
        inp = np.stack(self.frames[s:e], axis=0)
        tgt = self.frames[e - 1 + self.target_offset]

        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~self.sea_mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        land_vals_tgt = tgt[~self.sea_mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        return (
            torch.from_numpy(inp_filled).float(),
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),
        )


# ---------------- ConvLSTM (modified to accept kernel_size and increased capacity) ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5):
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
    def __init__(self, in_channels, hidden_dim=128, num_layers=3, kernel_size=5):
        super().__init__()
        self.input_len = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size=kernel_size))
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
def compute_norm_from_anomalies(dataset, train_idx, climatology):
    # climatology: numpy array HxW
    s = 0.0
    ss = 0.0
    cnt = 0
    clim = climatology.astype(np.float64)
    for i in train_idx:
        X, y = dataset[i]  # torch tensors
        Xn = X.numpy() - clim[np.newaxis, :, :]
        yn = y.numpy().squeeze(0) - clim
        arr = np.concatenate([Xn.ravel(), yn.ravel()]).astype(np.float64)
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


def compute_mean_ssim(preds, actuals, sea_mask):
    """
    preds, actuals: numpy arrays with shape (N,1,H,W)
    sea_mask: boolean array (H,W) True -> sea
    Returns mean SSIM across samples computed on land-only by forcing sea pixels to actual values
    (so they don't contribute to difference). If scikit-image not available returns None.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except Exception:
        return None

    if preds.size == 0 or actuals.size == 0:
        return None

    N = preds.shape[0]
    ssim_vals = []
    H, W = sea_mask.shape
    for i in range(N):
        a = actuals[i, 0].astype(np.float64).copy()
        p = preds[i, 0].astype(np.float64).copy()
        # Force sea pixels to actual values so SSIM measures land-only differences
        p[sea_mask] = a[sea_mask]
        dr = float(a.max() - a.min())
        if dr == 0.0:
            dr = 1e-6
        try:
            s = ssim_fn(a, p, data_range=dr)
        except Exception:
            s = np.nan
        ssim_vals.append(s)
    ssim_vals = np.array(ssim_vals, dtype=np.float64)
    if np.all(np.isnan(ssim_vals)):
        return None
    return float(np.nanmean(ssim_vals))


# ---------------- Prepare data ----------------
print("Loading dataset...")
print(f"Using device: {device}")

dataset = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
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

print(
    f"Samples (downsampled by {SAMPLE_STEP}): total={n}, train={len(train_set)}, val={len(val_set)}"
)
sea_mask = dataset.sea_mask
H, W = dataset.H, dataset.W

# ---------------- Compute climatology (per-grid mean of targets on training set) ----------------
print("Computing per-grid climatology from training targets...")
clim_sum = np.zeros((H, W), dtype=np.float64)
count = 0
for i in train_idx:
    _, y = dataset[i]  # (1,H,W) torch
    clim_sum += y.numpy().squeeze(0)
    count += 1
climatology = (clim_sum / max(1, count)).astype(np.float32)
np.save(os.path.join(out_dir, "climatology.npy"), climatology)
print("Saved climatology to:", os.path.join(out_dir, "climatology.npy"))

# ---------------- Compute normalization from anomalies (train set) ----------------
norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    print("Computing normalization (mean/std) from anomaly training set...")
    norm_mean, norm_std = compute_norm_from_anomalies(dataset, train_idx, climatology)
    print("norm mean, std (anomalies):", norm_mean, norm_std)

# prepare climatology & masks as torch tensors for training/inference
clim_t = torch.from_numpy(climatology).float().to(device)  # HxW
clim_t = clim_t.unsqueeze(0).unsqueeze(0)  # 1x1xHxW -- will broadcast to BxCxHxW
sea_mask_t = torch.from_numpy(sea_mask).to(device)  # HxW, True=sea
land_mask_t = (~sea_mask_t).to(device).unsqueeze(0).unsqueeze(0)  # 1x1xHxW

# ---------------- model (INCREASED CAPACITY) ----------------
model = SameSizeConvLSTM(
    in_channels=input_len, hidden_dim=128, num_layers=3, kernel_size=5
).to(device)
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)
criterion_map = nn.SmoothL1Loss(reduction="none")

# ---------------- training (with masked Smooth L1 on anomalies) ----------------
train_losses = []
val_losses = []
for epoch in range(1, epochs + 1):
    model.train()
    run = 0.0
    seen = 0
    pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{epochs}")
    for X, y in pbar:
        X = X.to(device)  # BxCxHxW
        y = y.to(device)  # Bx1xHxW

        # subtract climatology (broadcast) -> anomalies
        X_anom = X - clim_t  # BxCxHxW
        y_anom = y - clim_t  # Bx1xHxW

        # normalize anomalies
        if USE_NORMALIZATION:
            X_anom = (X_anom - norm_mean) / norm_std
            y_anom = (y_anom - norm_mean) / norm_std

        opt.zero_grad()
        out = model(X_anom)  # Bx1xHxW (anomaly prediction in normalized space)
        if out.shape != y_anom.shape:
            out = F.interpolate(
                out, size=y_anom.shape[2:], mode="bilinear", align_corners=False
            )

        # masked Smooth L1: compute elementwise map then restrict to land pixels
        map_loss = criterion_map(out, y_anom)  # Bx1xHxW
        mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)  # Bx1xHxW
        masked_vals = map_loss.masked_select(mask)
        if masked_vals.numel() == 0:
            loss = torch.tensor(0.0, device=device)
        else:
            loss = masked_vals.mean()
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

    # validation (compute anomaly mse/mae on land only)
    model.eval()
    vr = 0.0
    vseen = 0
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv = Xv.to(device)
            yv = yv.to(device)

            Xv_anom = Xv - clim_t
            yv_anom = yv - clim_t
            if USE_NORMALIZATION:
                Xv_anom = (Xv_anom - norm_mean) / norm_std
                yv_anom = (yv_anom - norm_mean) / norm_std

            outv = model(Xv_anom)
            if outv.shape != yv_anom.shape:
                outv = F.interpolate(
                    outv, size=yv_anom.shape[2:], mode="bilinear", align_corners=False
                )

            # compute scalar loss on land (in anomaly normalized space)
            map_loss = criterion_map(outv, yv_anom)
            mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
            masked_vals = map_loss.masked_select(mask)
            if masked_vals.numel() == 0:
                lv = torch.tensor(0.0, device=device)
            else:
                lv = masked_vals.mean()
            vr += lv.item() * Xv.size(0)
            vseen += Xv.size(0)
    val_loss = vr / max(1, vseen)
    val_losses.append(val_loss)
    sched.step(val_loss)
    print(f"Epoch {epoch:03d} Train={train_loss:.6f} Val={val_loss:.6f}")

# save loss curve
loss_fig_path = os.path.join(out_dir, "Train_Val_Loss.png")
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (SmoothL1 on anomalies)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_fig_path, dpi=200)
plt.close()
print("Saved loss curve to:", loss_fig_path)

# ---------------- inference on val set (produce absolute predictions) ----------------
model.eval()
all_preds = []
all_actuals = []
with torch.no_grad():
    for Xv, yv in val_loader:
        Xv = Xv.to(device)
        yv = yv.to(device)

        Xv_anom = Xv - clim_t
        if USE_NORMALIZATION:
            Xv_anom = (Xv_anom - norm_mean) / norm_std

        outv = model(Xv_anom)
        if outv.shape != yv.shape:
            outv = F.interpolate(
                outv, size=yv.shape[2:], mode="bilinear", align_corners=False
            )
        # denormalize & add climatology back to get absolute temps
        if USE_NORMALIZATION:
            outv = outv * norm_std + norm_mean
        outv_abs = outv + clim_t  # add climatology back (broadcast)
        all_preds.append(outv_abs.cpu().numpy())
        all_actuals.append(yv.cpu().numpy())

preds = (
    np.concatenate(all_preds, axis=0)
    if len(all_preds)
    else np.empty((0, 1, dataset.H, dataset.W))
)
actuals = (
    np.concatenate(all_actuals, axis=0)
    if len(all_actuals)
    else np.empty((0, 1, dataset.H, dataset.W))
)

# compute val metrics on land-only
if preds.size:
    mask_flat = (~dataset.sea_mask).ravel()
    preds_flat = preds.reshape(preds.shape[0], -1)
    actuals_flat = actuals.reshape(actuals.shape[0], -1)

    # compute per-pixel aggregated metrics (land-only)
    diffs = actuals_flat[:, mask_flat] - preds_flat[:, mask_flat]
    mse = float(np.nanmean(diffs**2))
    mae = float(np.nanmean(np.abs(diffs)))
    rmse = float(np.sqrt(mse))
    print(
        "Model VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            mse, mae, rmse
        )
    )

    # compute SSIM averaged over validation samples (land-only via sea fill trick)
    mean_ssim_val = compute_mean_ssim(preds, actuals, dataset.sea_mask)
    if mean_ssim_val is None:
        print("Model VAL SSIM (land-only): SSIM not computed (scikit-image missing or no preds).")
    else:
        print(f"Model VAL SSIM (land-only): SSIM={mean_ssim_val:.6f}")
else:
    print("No validation predictions available.")
    mean_ssim_val = None

# ---------------- Persistence baseline on validation (last input frame) ----------------
print("Computing persistence baseline on validation set...")
pers_preds = []
pers_actuals = []
with torch.no_grad():
    for Xv, yv in val_loader:
        # last input frame (absolute units already in dataset)
        last = Xv[:, -1:, :, :].cpu().numpy()  # Bx1xHxW
        pers_preds.append(last)
        pers_actuals.append(yv.cpu().numpy())
pers_preds = (
    np.concatenate(pers_preds, axis=0)
    if len(pers_preds)
    else np.empty((0, 1, dataset.H, dataset.W))
)
pers_actuals = (
    np.concatenate(pers_actuals, axis=0)
    if len(pers_actuals)
    else np.empty((0, 1, dataset.H, dataset.W))
)
if pers_preds.size:
    pers_diffs = (
        pers_actuals.reshape(pers_actuals.shape[0], -1)[:, mask_flat]
        - pers_preds.reshape(pers_preds.shape[0], -1)[:, mask_flat]
    )
    pers_mse = float(np.nanmean(pers_diffs**2))
    pers_mae = float(np.nanmean(np.abs(pers_diffs)))
    pers_rmse = float(np.sqrt(pers_mse))
    print(
        "Persistence VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            pers_mse, pers_mae, pers_rmse
        )
    )

    # SSIM for persistence baseline (land-only)
    mean_ssim_pers = compute_mean_ssim(pers_preds, pers_actuals, dataset.sea_mask)
    if mean_ssim_pers is None:
        print("Persistence VAL SSIM (land-only): SSIM not computed (scikit-image missing or no preds).")
    else:
        print(f"Persistence VAL SSIM (land-only): SSIM={mean_ssim_pers:.6f}")
else:
    print("No persistence predictions available.")
    mean_ssim_pers = None

# ---------------- Bias correction (linear) fitted on val predictions -> apply & report corrected metrics ----------------
print("Fitting linear bias correction on val set (land-only)...")
preds_vec = (
    preds.reshape(-1, preds.shape[-2] * preds.shape[-1])
    if preds.size
    else np.empty((0, H * W))
)
actuals_vec = (
    actuals.reshape(-1, actuals.shape[-2] * actuals.shape[-1])
    if actuals.size
    else np.empty((0, H * W))
)
if preds.size:
    # flatten across samples and spatial dims
    p_flat = preds_vec[:, :].reshape(-1)
    a_flat = actuals_vec[:, :].reshape(-1)
    land_mask_flat = (~dataset.sea_mask).ravel()
    sel = np.repeat(land_mask_flat[np.newaxis, :], preds.shape[0], axis=0).reshape(-1)
    p_sel = p_flat[sel]
    a_sel = a_flat[sel]
    if p_sel.size >= 2:
        # fit linear regression a = slope * p + intercept
        slope, intercept = np.polyfit(p_sel, a_sel, 1)
    else:
        slope, intercept = 1.0, 0.0
    print(f"Bias correction: slope={slope:.6f}, intercept={intercept:.6f}")
    # apply correction
    preds_corrected = slope * preds + intercept
    # compute corrected metrics on land-only
    pc_flat = preds_corrected.reshape(preds_corrected.shape[0], -1)[:, mask_flat]
    a_flat_mat = actuals_flat[:, mask_flat]
    dif_corr = a_flat_mat - pc_flat
    mse_corr = float(np.nanmean(dif_corr**2))
    mae_corr = float(np.nanmean(np.abs(dif_corr)))
    rmse_corr = float(np.sqrt(mse_corr))
    print(
        "Bias-corrected VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            mse_corr, mae_corr, rmse_corr
        )
    )

    # SSIM for bias-corrected preds (land-only)
    mean_ssim_corr = compute_mean_ssim(preds_corrected, actuals, dataset.sea_mask)
    if mean_ssim_corr is None:
        print("Bias-corrected VAL SSIM (land-only): SSIM not computed (scikit-image missing or no preds).")
    else:
        print(f"Bias-corrected VAL SSIM (land-only): SSIM={mean_ssim_corr:.6f}")
else:
    slope, intercept = 1.0, 0.0
    mean_ssim_corr = None
    print("No preds to fit bias correction.")

# ---------------- Build the requested sample in the DOWNSAMPLED timeline ----------
# Original requested input times: indices [0,3,6,9,12,15,18,21] (original timeline)
# After downsampling by SAMPLE_STEP=3, these map to indices [0,1,2,3,4,5,6,7] (contiguous)
# Original requested target index 33 maps to 33 // 3 = 11 in the downsampled timeline.
ORIG_INPUT_INDICES = [0, 3, 6, 9, 12, 15, 18, 21]
ORIG_TARGET_INDEX = 33

# Confirm mapping
ds_input_idxs = [i // SAMPLE_STEP for i in ORIG_INPUT_INDICES]
ds_target_idx = ORIG_TARGET_INDEX // SAMPLE_STEP
assert ds_input_idxs == list(
    range(input_len)
), f"expected contiguous downsampled inputs 0..{input_len-1}, got {ds_input_idxs}"

# Use dataset.frames (these are already downsampled inside the dataset)
frames_ds = dataset.frames  # downsampled frames list
T_ds = len(frames_ds)
if ds_target_idx < 0 or ds_target_idx >= T_ds:
    raise IndexError(
        f"Downsampled target index {ds_target_idx} out of range (0..{T_ds-1})"
    )

input_arr = np.stack([frames_ds[i] for i in ds_input_idxs], axis=0).astype(
    np.float32
)  # (C,H,W)
target_arr = frames_ds[ds_target_idx].astype(np.float32)

# Fill NaNs same as dataset.__getitem__
inp_filled = np.empty_like(input_arr, dtype=np.float32)
for i in range(input_arr.shape[0]):
    frame = input_arr[i]
    land_vals = frame[~dataset.sea_mask]
    fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
    inp_filled[i] = np.where(np.isnan(frame), fill, frame)

land_vals_tgt = target_arr[~dataset.sea_mask]
fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
tgt_filled = np.where(np.isnan(target_arr), fill_t, target_arr).astype(np.float32)

# inference on the single sample (apply same preprocessing: subtract climatology, normalize)
model.eval()
with torch.no_grad():
    X_sample = torch.from_numpy(inp_filled).unsqueeze(0).float().to(device)  # 1xCxHxW
    y_sample = (
        torch.from_numpy(tgt_filled).unsqueeze(0).unsqueeze(0).float().to(device)
    )  # 1x1xHxW

    X_sample_anom = X_sample - clim_t  # subtract climatology
    if USE_NORMALIZATION:
        X_sample_anom = (X_sample_anom - norm_mean) / norm_std

    out_sample = model(X_sample_anom)
    if out_sample.shape[2:] != tgt_filled.shape:
        out_sample = F.interpolate(
            out_sample, size=tgt_filled.shape, mode="bilinear", align_corners=False
        )
    if USE_NORMALIZATION:
        out_sample = out_sample * norm_std + norm_mean
    out_sample_abs = out_sample + clim_t  # add climatology back
    pred_sample = out_sample_abs.cpu().numpy()[0, 0]

# apply bias correction to sample prediction
pred_sample_bc = slope * pred_sample + intercept

actual_sample = tgt_filled
mask = dataset.sea_mask
actual_masked = np.ma.masked_where(mask, actual_sample)
pred_masked = np.ma.masked_where(mask, pred_sample)
pred_masked_bc = np.ma.masked_where(mask, pred_sample_bc)
error_masked = np.ma.masked_where(mask, actual_sample - pred_sample)
error_masked_bc = np.ma.masked_where(mask, actual_sample - pred_sample_bc)

# sample metrics (land-only) before and after bias correction
land_actual = actual_sample[~mask]
land_pred = pred_sample[~mask]
land_pred_bc = pred_sample_bc[~mask]
if land_actual.size:
    dif = land_actual - land_pred
    mse_sample = float(np.nanmean(dif**2))
    mae_sample = float(np.nanmean(np.abs(dif)))
    rmse_sample = float(np.sqrt(mse_sample))
    # corrected
    difc = land_actual - land_pred_bc
    mse_sample_bc = float(np.nanmean(difc**2))
    mae_sample_bc = float(np.nanmean(np.abs(difc)))
    rmse_sample_bc = float(np.sqrt(mse_sample_bc))
else:
    mse_sample = mae_sample = rmse_sample = float("nan")
    mse_sample_bc = mae_sample_bc = rmse_sample_bc = float("nan")

print(
    f"Sample metrics (downsampled target idx {ds_target_idx}): BEFORE BC -> MSE={mse_sample:.6f}, MAE={mae_sample:.6f}, RMSE={rmse_sample:.6f}"
)
print(
    f"Sample metrics (downsampled target idx {ds_target_idx}): AFTER BC  -> MSE={mse_sample_bc:.6f}, MAE={mae_sample_bc:.6f}, RMSE={rmse_sample_bc:.6f}"
)

# Compute SSIM for this sample (land-only via sea fill trick) if possible
try:
    from skimage.metrics import structural_similarity as ssim_fn

    a = actual_sample.astype(np.float64).copy()
    p = pred_sample.astype(np.float64).copy()
    p[mask] = a[mask]
    dr = float(a.max() - a.min())
    if dr == 0:
        dr = 1e-6
    s_sample = float(ssim_fn(a, p, data_range=dr))
    # corrected
    pbc = pred_sample_bc.astype(np.float64).copy()
    pbc[mask] = a[mask]
    s_sample_bc = float(ssim_fn(a, pbc, data_range=dr))
    print(f"Sample SSIM (land-only): BEFORE BC SSIM={s_sample:.6f}   AFTER BC SSIM={s_sample_bc:.6f}")
except Exception:
    s_sample = float("nan")
    s_sample_bc = float("nan")
    print(
        "Sample SSIM not computed (scikit-image missing). Install scikit-image to enable SSIM computation."
    )

# ----------------- lat/lon and plotting ---------------
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

origin = "lower"
if lats[0] > lats[-1]:
    origin = "upper"

lon_min = float(lons.min())
lon_max = float(lons.max())
lat_min = float(lats.min())
lat_max = float(lats.max())
extent = [lon_min, lon_max, lat_min, lat_max]

num_xticks = min(6, lons.size)
num_yticks = min(6, lats.size)
xtick_idxs = np.linspace(0, lons.size - 1, num_xticks).astype(int)
ytick_idxs = np.linspace(0, lats.size - 1, num_yticks).astype(int)
xticks = lons[xtick_idxs]
yticks = lats[ytick_idxs]

# cmap gray, masked -> white
cmap_temp = plt.get_cmap("gray")
try:
    cmap_temp = cmap_temp.copy()
except Exception:
    cmap_temp = mpl.colors.ListedColormap(cmap_temp(np.linspace(0, 1, cmap_temp.N)))
cmap_temp.set_bad("white")
cmap_err = plt.get_cmap("gray")
try:
    cmap_err = cmap_err.copy()
except Exception:
    cmap_err = mpl.colors.ListedColormap(cmap_err(np.linspace(0, 1, cmap_err.N)))
cmap_err.set_bad("white")

combined = np.concatenate(
    [actual_masked.filled(np.nan).ravel(), pred_masked.filled(np.nan).ravel()]
)
combined = combined[~np.isnan(combined)]
if combined.size == 0:
    raise RuntimeError("No valid land pixels to plot.")
vmin = float(np.nanmin(combined))
vmax = float(np.nanmax(combined))
err_abs = float(np.nanmax(np.abs((actual_masked - pred_masked).filled(0.0))))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "\n\nInput: 1st April, 2024 | 12am - 9pm       |       Output: 2nd April, 2024 | 9am\n\n",
    fontsize=14,
    fontweight="bold",
    y=0.96,
)

im0 = axes[0].imshow(
    actual_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[0].set_title("Actual", fontsize=14)
axes[0].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[0].set_xticks(xticks)
axes[0].set_yticks(yticks)

im1 = axes[1].imshow(
    pred_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_temp,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)
axes[1].set_title("Predicted", fontsize=14)
axes[1].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[1].set_xticks(xticks)
axes[1].set_yticks(yticks)

im2 = axes[2].imshow(
    error_masked,
    origin=origin,
    extent=extent,
    cmap=cmap_err,
    vmin=-err_abs,
    vmax=err_abs,
    interpolation="nearest",
)
axes[2].set_title("Error = Actual - Predicted", fontsize=14)
axes[2].set_ylabel("Latitude   →", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Longitude   →", fontsize=11, fontweight="bold")
axes[2].set_xticks(xticks)
axes[2].set_yticks(yticks)

for ax, im in zip(axes, [im0, im1, im2]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    if im is im2:
        cbar.set_label("Error (units)", fontsize=10)
        pretty_cb(cbar, fmt="%.3f")
    else:
        cbar.set_label("2m Temperature (units)", fontsize=10)
        pretty_cb(cbar, fmt="%.2f")

# include SSIM (sample) beside RMSE in the plotted metrics
metrics_text = (
    f"MSE: {mse_sample:.6f}   MAE: {mae_sample:.6f}   RMSE: {rmse_sample:.6f}   SSIM: {s_sample:.6f}"
)
# reserve more space at the bottom for the metrics (increase bottom from 0.05 -> 0.15)
plt.tight_layout(rect=[0, 0.15, 1, 0.94])

# put metrics centered in the reserved band and lift them up
fig.text(
    0.5,  # center under all three plots
    0.02,  # y position (in figure coords)
    metrics_text,
    fontsize=14,
    va="center",
    ha="center",
    fontname="Times New Roman",
    bbox=dict(
        facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.3"
    ),
)
save_path = os.path.join(out_dir, "Actual_Predicted_Error.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved plot to:", save_path)
print("Done.")
