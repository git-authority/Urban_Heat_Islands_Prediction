#!/usr/bin/env python3
# plot_output_swinlstm_v2_arch.py
import os
import numpy as np
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# Optional imports
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

# ---------------- CONFIG ----------------
folder_path = "../../Dataset"  # parent folder with year subfolders
out_dir = "SwinLSTM_Sliding_v2"  # where model + maps were saved
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3

batch_size = 6
val_split = 0.20
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True
PRINT_EVERY = 1

# Use V2 architecture hyperparams
hidden_dim = 256
num_layers = 3
dropout_p = 0.05

WINDOW_SIZE = 4
ATTN_HEADS = 8

SSIM_WEIGHT = 0.0
GRAD_WEIGHT = 0.8

plt.rcParams["font.family"] = "Times New Roman"
torch.manual_seed(seed)
np.random.seed(seed)
print(f"✅ Using device: {device}")


# ---------------- Dataset ----------------
class SlidingMaskDataset(TorchDataset):
    def __init__(
        self,
        folder_path,
        input_len=8,
        target_offset=4,
        sample_step=1,
        include_years=None,
        exclude_years=None,
    ):
        import os
        from netCDF4 import Dataset as ncDataset

        self.frames = []
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

        month_map = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        entries = sorted(os.listdir(folder_path))
        year_dirs = []
        for e in entries:
            p = os.path.join(folder_path, e)
            if os.path.isdir(p):
                year_dirs.append(e)

        if include_years is not None:
            year_dirs = [y for y in year_dirs if str(y) in set(map(str, include_years))]
        if exclude_years is not None:
            year_dirs = [
                y for y in year_dirs if str(y) not in set(map(str, exclude_years))
            ]

        try:
            year_dirs = sorted(year_dirs, key=lambda x: int(x))
        except Exception:
            year_dirs = sorted(year_dirs)

        files_paths = []
        for y in year_dirs:
            yd = os.path.join(folder_path, y)
            month_files = [f for f in sorted(os.listdir(yd)) if f.endswith(".nc")]

            def month_key(fn):
                name = os.path.splitext(fn)[0].lower()
                return month_map.get(name, 999), fn

            month_files = sorted(month_files, key=month_key)
            for mf in month_files:
                files_paths.append(os.path.join(yd, mf))

        if not files_paths:
            raise ValueError(
                f"No .nc files found in {folder_path} for years {year_dirs}"
            )

        for path in files_paths:
            fn = os.path.basename(path)
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

        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]

        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)

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


# ---------------- Model (take from SwinLSTM_v2) ----------------
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class SwinLSTMCell(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, window_size=4, num_heads=8, dropout_p=0.05
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.in_proj = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.attn = LocalWindowAttention(
            dim=hidden_dim, window_size=window_size, num_heads=num_heads
        )
        self.gates_conv = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        ph = (ws - (H % ws)) % ws
        pw = (ws - (W % ws)) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        num_win = (Hp // ws) * (Wp // ws)
        x = x.view(B * num_win, ws * ws, C)
        return x, (Hp, Wp, Hp // ws, Wp // ws)

    def window_unpartition(self, x_w, grid_info):
        Hp, Wp, num_h, num_w = grid_info[0], grid_info[1], grid_info[2], grid_info[3]
        ws = self.window_size
        B_numwin, N, C = x_w.shape
        B = B_numwin // (num_h * num_w)
        x = x_w.view(B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)
        return x

    def forward(self, x, hidden):
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.in_proj(comb)
        x_win, grid_info = self.window_partition(proj)
        attn_out = self.attn(x_win)
        attn_map = self.window_unpartition(attn_out, grid_info)
        attn_map = attn_map[:, :, :H, :W]
        attn_map = self.gn(attn_map)
        if self.training and self.dropout_p > 0:
            attn_map = F.dropout2d(attn_map, p=self.dropout_p)
        conv = self.gates_conv(attn_map)
        ci, cf, co, cg = torch.chunk(conv, 4, dim=1)
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


class ResidualSwinLSTMWithRefine(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_layers=3,
        window_size=4,
        num_heads=8,
        dropout_p=0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                SwinLSTMCell(
                    in_dim,
                    hidden_dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    dropout_p=dropout_p,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device
        hiddens = [l.init_hidden(B, (H, W), device) for l in self.layers]
        last = None
        for t in range(C):
            frame = x[:, t : t + 1, :, :]
            inp = frame
            for li, layer in enumerate(self.layers):
                h, c = hiddens[li]
                hnext, cnext = layer(inp, (h, c))
                if li > 0:
                    hnext = hnext + inp
                hiddens[li] = (hnext, cnext)
                inp = hnext
            last = inp
        out = self.refine(last)
        return out


# ---------------- Helper functions ----------------
def pretty_cb(cb, fmt="%.2f"):
    cb.ax.tick_params(labelsize=9)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))


def compute_mean_ssim(preds, actuals, sea_mask):
    if ssim_fn is None:
        return None
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)
    if isinstance(actuals, torch.Tensor):
        actuals_np = actuals.detach().cpu().numpy()
    else:
        actuals_np = np.asarray(actuals)
    mask_np = np.asarray(sea_mask)
    if mask_np.ndim >= 2:
        H, W = mask_np.shape[-2], mask_np.shape[-1]
    else:
        if preds_np.ndim >= 2:
            H, W = preds_np.shape[-2], preds_np.shape[-1]
        elif actuals_np.ndim >= 2:
            H, W = actuals_np.shape[-2], actuals_np.shape[-1]
        else:
            raise ValueError("Cannot infer H,W for SSIM from inputs")

    def normalize_arr(arr, name):
        arr = np.asarray(arr)
        if arr.ndim < 2:
            raise ValueError(f"Unsupported {name} shape for SSIM: {arr.shape}")
        if arr.shape[-2:] != (H, W):
            raise ValueError(
                f"Unsupported {name} shape for SSIM (expected last dims H,W): {arr.shape}"
            )
        lead_shape = arr.shape[:-2]
        if lead_shape == ():
            arr_norm = arr.reshape(1, H, W)
        else:
            N = int(np.prod(lead_shape))
            arr_norm = arr.reshape(N, H, W)
        return arr_norm

    preds_norm = normalize_arr(preds_np, "preds")
    actuals_norm = normalize_arr(actuals_np, "actuals")
    N = preds_norm.shape[0]
    masks = []
    if mask_np.ndim == 2:
        masks = [mask_np.astype(bool) for _ in range(N)]
    elif mask_np.ndim == 3:
        if mask_np.shape[0] == N:
            masks = [mask_np[i].astype(bool) for i in range(N)]
        elif mask_np.shape[0] == 1:
            masks = [mask_np[0].astype(bool) for _ in range(N)]
        else:
            try:
                if mask_np.shape[-2:] == (H, W):
                    m_collapsed = mask_np.reshape(-1, H, W)
                    if m_collapsed.shape[0] == N:
                        masks = [m_collapsed[i].astype(bool) for i in range(N)]
                    else:
                        masks = [m_collapsed[0].astype(bool) for _ in range(N)]
                else:
                    raise ValueError
            except Exception:
                raise ValueError(
                    f"Unsupported sea_mask shape for SSIM: {mask_np.shape}"
                )
    else:
        raise ValueError(f"Unsupported sea_mask ndim for SSIM: {mask_np.ndim}")
    ssim_vals = []
    for i in range(N):
        a = actuals_norm[i].astype(np.float64).copy()
        p = preds_norm[i].astype(np.float64).copy()
        m = masks[i]
        try:
            p[m] = a[m]
        except Exception:
            pass
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


def gradient_loss_torch(pred, target, mask):
    sobel_x = (
        torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
            device=pred.device,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = sobel_x.transpose(2, 3)
    grad_px = F.conv2d(pred, sobel_x, padding=1)
    grad_py = F.conv2d(pred, sobel_y, padding=1)
    grad_tx = F.conv2d(target, sobel_x, padding=1)
    grad_ty = F.conv2d(target, sobel_y, padding=1)
    gdiff = torch.abs(grad_px - grad_tx) + torch.abs(grad_py - grad_ty)
    gmask = mask.expand_as(gdiff)
    sel = gdiff.masked_select(gmask)
    if sel.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return sel.mean()


def fit_columnwise_bias(actuals, preds, sea_mask, smooth_sigma=2.0):
    N, H, W = preds.shape
    slope_cols = np.ones(W, dtype=np.float32)
    intercept_cols = np.zeros(W, dtype=np.float32)
    for j in range(W):
        p_col = preds[:, :, j].reshape(-1)
        a_col = actuals[:, :, j].reshape(-1)
        lat_land_mask = ~sea_mask[:, j]
        if lat_land_mask.sum() == 0:
            slope_cols[j] = 1.0
            intercept_cols[j] = 0.0
            continue
        sel = np.tile(lat_land_mask, N)
        p_sel = p_col[sel]
        a_sel = a_col[sel]
        if p_sel.size >= 5 and np.nanstd(p_sel) > 1e-6:
            s, b = np.polyfit(p_sel, a_sel, 1)
        else:
            s, b = 1.0, 0.0
        slope_cols[j] = float(s)
        intercept_cols[j] = float(b)
    if gaussian_filter is not None:
        slope_cols = gaussian_filter(slope_cols, sigma=smooth_sigma)
        intercept_cols = gaussian_filter(intercept_cols, sigma=smooth_sigma)
    slope_cols = np.clip(slope_cols, 0.75, 1.25)
    slope_map = np.tile(slope_cols[np.newaxis, :], (H, 1))
    intercept_map = np.tile(intercept_cols[np.newaxis, :], (H, 1))
    return slope_map.astype(np.float32), intercept_map.astype(np.float32)


# ---------------- Prepare splits ----------------
print("Loading dataset and preparing train/val/test splits (chronological)...")
all_years = sorted(
    [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
)
print("Detected year folders:", all_years)
TEST_YEAR = "2024" if "2024" in all_years else all_years[-1]
trainval_years = [y for y in all_years if y != TEST_YEAR]
if not trainval_years:
    raise ValueError("No years available for train/val after excluding test year.")

dataset_trainval = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=trainval_years,
)
dataset_test = SlidingMaskDataset(
    folder_path,
    input_len=input_len,
    target_offset=target_offset,
    sample_step=SAMPLE_STEP,
    include_years=[TEST_YEAR],
)

n_tv = len(dataset_trainval)
print(f"Train+Val samples (chronological across years {trainval_years}): total={n_tv}")

indices_tv = np.arange(n_tv)
split_idx = int(np.floor((1.0 - val_split) * n_tv))
train_idx = indices_tv[:split_idx]
val_idx = indices_tv[split_idx:]

train_set = Subset(dataset_trainval, train_idx)
val_set = Subset(dataset_trainval, val_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

sea_mask = dataset_trainval.sea_mask
H, W = dataset_trainval.H, dataset_trainval.W

# ---------------- climatology ----------------
print("Computing per-grid climatology from training targets...")
clim_sum = np.zeros((H, W), dtype=np.float64)
count = 0
for i in train_idx:
    _, y = dataset_trainval[i]
    clim_sum += y.numpy().squeeze(0)
    count += 1
climatology = (clim_sum / max(1, count)).astype(np.float32)
np.save(os.path.join(out_dir, "climatology.npy"), climatology)
print("Saved climatology to:", os.path.join(out_dir, "climatology.npy"))


# ---------------- normalization ----------------
def compute_norm_from_anomalies(dataset_obj, train_idx_local, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    clim = climatology.astype(np.float64)
    for i in train_idx_local:
        X, y = dataset_obj[i]
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


# try to load saved normalization if present; otherwise compute from train_idx
norm_mean, norm_std = 0.0, 1.0
norm_npz = os.path.join(out_dir, "norm_mean_std.npz")
if os.path.exists(norm_npz):
    arr = np.load(norm_npz)
    norm_mean = float(arr["mean"])
    norm_std = float(arr["std"])
    print("Loaded norm mean/std from:", norm_npz, norm_mean, norm_std)
else:
    if USE_NORMALIZATION:
        print("Computing normalization (mean/std) from anomaly training set...")
        norm_mean, norm_std = compute_norm_from_anomalies(
            dataset_trainval, train_idx, climatology
        )
        print("norm mean, std (anomalies):", norm_mean, norm_std)

clim_t = torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)
sea_mask_t = torch.from_numpy(sea_mask).to(device)
land_mask_t = (~sea_mask_t).to(device).unsqueeze(0).unsqueeze(0)

# ---------------- load model ----------------
model = ResidualSwinLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    window_size=WINDOW_SIZE,
    num_heads=ATTN_HEADS,
    dropout_p=dropout_p,
).to(device)

checkpoint_path = os.path.join(out_dir, "best_model.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
print("Loading model from:", checkpoint_path)

state = torch.load(checkpoint_path, map_location=device)
# introspect
if isinstance(state, dict):
    print("Checkpoint top-level keys:", list(state.keys())[:40])
# Attempt strict load, fallback to common alternatives
try:
    model.load_state_dict(state)
    print("Model loaded with strict=True")
except Exception as e:
    print("Strict load failed:", e)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        try:
            model.load_state_dict(state["state_dict"])
            print("Loaded model from state['state_dict']")
        except Exception as e2:
            print("Loading state['state_dict'] failed:", e2)
            model.load_state_dict(state, strict=False)
            print("Loaded model with strict=False from top-level dict")
    else:
        model.load_state_dict(state, strict=False)
        print("Loaded model with strict=False")

model.to(device)
model.eval()

# ---------------- blending alpha / bias maps load if present ----------------
alpha_path = os.path.join(out_dir, "blending_alpha.npy")
if os.path.exists(alpha_path):
    try:
        alpha = float(np.load(alpha_path))
        print("Loaded blending alpha:", alpha)
    except Exception:
        alpha = 1.0
        print("Failed to load alpha, using 1.0")
else:
    alpha = 1.0
    print("Blending alpha not found; using alpha=1.0")

slope_path = os.path.join(out_dir, "bias_slope_map.npy")
intercept_path = os.path.join(out_dir, "bias_intercept_map.npy")
if os.path.exists(slope_path) and os.path.exists(intercept_path):
    slope_map = np.load(slope_path)
    intercept_map = np.load(intercept_path)
    print("Loaded bias slope/intercept maps from:", out_dir)
else:
    slope_map = np.ones((H, W), dtype=np.float32)
    intercept_map = np.zeros((H, W), dtype=np.float32)
    print("Bias maps not found; using identity/no-shift maps.")

# ---------------- inference on validation set (for blend & bias fit) ----------------
print("Running inference on validation set for bias correction & blending...")
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
        if USE_NORMALIZATION:
            outv = outv * norm_std + norm_mean
        outv_abs = outv + clim_t
        all_preds.append(outv_abs.cpu().numpy())
        all_actuals.append(yv.cpu().numpy())

preds = np.concatenate(all_preds, axis=0) if len(all_preds) else np.empty((0, 1, H, W))
actuals = (
    np.concatenate(all_actuals, axis=0) if len(all_actuals) else np.empty((0, 1, H, W))
)

if preds.size:
    mask_flat = (~dataset_trainval.sea_mask).ravel()
    preds_flat = preds.reshape(preds.shape[0], -1)
    actuals_flat = actuals.reshape(actuals.shape[0], -1)
    diffs = actuals_flat[:, mask_flat] - preds_flat[:, mask_flat]
    mse = float(np.nanmean(diffs**2))
    mae = float(np.nanmean(np.abs(diffs)))
    rmse = float(np.sqrt(mse))
    mean_ssim_val = compute_mean_ssim(preds, actuals, dataset_trainval.sea_mask)
    print(
        "Model VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            mse, mae, rmse
        )
    )
    if mean_ssim_val is not None:
        print(f"Model VAL SSIM (land-only): SSIM={mean_ssim_val:.6f}")
else:
    print("No validation predictions available.")
    mean_ssim_val = None

# ---------------- persistence baseline & blending coefficient ----------------
print(
    "Computing persistence baseline and optimal blending coefficient (alpha) on val (if not loaded)..."
)
pers_preds = []
pers_actuals = []
with torch.no_grad():
    for Xv, yv in val_loader:
        last = Xv[:, -1:, :, :].cpu().numpy()
        pers_preds.append(last)
        pers_actuals.append(yv.cpu().numpy())
pers_preds = (
    np.concatenate(pers_preds, axis=0) if len(pers_preds) else np.empty((0, 1, H, W))
)
pers_actuals = (
    np.concatenate(pers_actuals, axis=0)
    if len(pers_actuals)
    else np.empty((0, 1, H, W))
)

if (not os.path.exists(alpha_path)) and pers_preds.size and preds.size:
    Pp = pers_preds.reshape(pers_preds.shape[0], -1)[:, mask_flat]
    Pm = preds.reshape(preds.shape[0], -1)[:, mask_flat]
    A = actuals.reshape(actuals.shape[0], -1)[:, mask_flat]
    numer = np.nansum((A - Pp) * (Pm - Pp))
    denom = np.nansum((Pm - Pp) ** 2)
    if denom <= 1e-12:
        alpha = 1.0
    else:
        alpha = float(numer / denom)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    print(f"Blending alpha (model weight) learned on val: {alpha:.4f}")
    try:
        np.save(os.path.join(out_dir, "blending_alpha.npy"), np.array(alpha))
        print("Saved blending alpha to:", out_dir)
    except Exception:
        pass

if pers_preds.size and preds.size:
    preds_blend = alpha * preds[:, 0] + (1.0 - alpha) * pers_preds[:, 0]
    preds_blend = preds_blend[:, None, :, :]
else:
    preds_blend = preds.copy()

# If bias maps were missing earlier, fit now and save
if (
    not (os.path.exists(slope_path) and os.path.exists(intercept_path))
) and preds_blend.size:
    print("Fitting slope/intercept maps now from blended preds...")
    Pblend = preds_blend[:, 0]
    A = actuals[:, 0]
    slope_map, intercept_map = fit_columnwise_bias(
        A, Pblend, dataset_trainval.sea_mask, smooth_sigma=2.0
    )
    try:
        np.save(slope_path, slope_map)
        np.save(intercept_path, intercept_map)
        print("Saved slope/intercept maps to:", out_dir)
    except Exception:
        print("Failed to save slope/intercept maps.")

# Apply bias correction to blended preds
if preds_blend.size:
    preds_corr = preds_blend.copy()
    preds_corr[:, :] = preds_blend * slope_map[None, :, :] + intercept_map[None, :, :]
    pc_flat = preds_corr.reshape(preds_corr.shape[0], -1)[:, mask_flat]
    a_flat_mat = actuals_flat[:, mask_flat]
    dif_corr = a_flat_mat - pc_flat
    mse_corr = float(np.nanmean(dif_corr**2))
    mae_corr = float(np.nanmean(np.abs(dif_corr)))
    rmse_corr = float(np.sqrt(mse_corr))
    print(
        "Bias-corrected (blended) VAL METRICS (land-only): MSE={:.6f} MAE={:.6f} RMSE={:.6f}".format(
            mse_corr, mae_corr, rmse_corr
        )
    )
    mean_ssim_corr = compute_mean_ssim(preds_corr, actuals, dataset_trainval.sea_mask)
    if mean_ssim_corr is not None:
        print(
            f"Bias-corrected (blended) VAL SSIM (land-only): SSIM={mean_ssim_corr:.6f}"
        )
else:
    preds_corr = preds_blend
    mean_ssim_corr = None

# ---------------- Build the requested sample in the DOWNSAMPLED timeline ----------
ORIG_INPUT_INDICES = [0, 3, 6, 9, 12, 15, 18, 21]
ORIG_TARGET_INDEX = 33
ds_input_idxs = [i // SAMPLE_STEP for i in ORIG_INPUT_INDICES]
ds_target_idx = ORIG_TARGET_INDEX // SAMPLE_STEP
assert ds_input_idxs == list(
    range(input_len)
), f"Expected contiguous downsampled inputs 0..{input_len-1}, got {ds_input_idxs}"

frames_ds = dataset_trainval.frames
T_ds = len(frames_ds)
if ds_target_idx < 0 or ds_target_idx >= T_ds:
    raise IndexError(
        f"Downsampled target index {ds_target_idx} out of range (0..{T_ds-1})"
    )

input_arr = np.stack([frames_ds[i] for i in ds_input_idxs], axis=0).astype(np.float32)
target_arr = frames_ds[ds_target_idx].astype(np.float32)

# fill NaNs
inp_filled = np.empty_like(input_arr, dtype=np.float32)
for i in range(input_arr.shape[0]):
    frame = input_arr[i]
    land_vals = frame[~dataset_trainval.sea_mask]
    fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
    inp_filled[i] = np.where(np.isnan(frame), fill, frame)
land_vals_tgt = target_arr[~dataset_trainval.sea_mask]
fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
tgt_filled = np.where(np.isnan(target_arr), fill_t, target_arr).astype(np.float32)

# inference on single sample
model.eval()
with torch.no_grad():
    X_sample = torch.from_numpy(inp_filled).unsqueeze(0).float().to(device)
    y_sample = torch.from_numpy(tgt_filled).unsqueeze(0).unsqueeze(0).float().to(device)
    X_sample_anom = X_sample - clim_t
    if USE_NORMALIZATION:
        X_sample_anom = (X_sample_anom - norm_mean) / norm_std
    out_sample = model(X_sample_anom)
    if out_sample.shape[2:] != tgt_filled.shape:
        out_sample = F.interpolate(
            out_sample, size=tgt_filled.shape, mode="bilinear", align_corners=False
        )
    if USE_NORMALIZATION:
        out_sample = out_sample * norm_std + norm_mean
    out_sample_abs = out_sample + clim_t
    pred_sample = out_sample_abs.cpu().numpy()[0, 0]

pers_sample = inp_filled[-1]
pred_sample_blend = alpha * pred_sample + (1.0 - alpha) * pers_sample
pred_sample_bc = pred_sample_blend * slope_map + intercept_map

actual_sample = tgt_filled
mask = dataset_trainval.sea_mask
actual_masked = np.ma.masked_where(mask, actual_sample)
pred_masked = np.ma.masked_where(mask, pred_sample_bc)
error_masked = np.ma.masked_where(mask, actual_sample - pred_sample_bc)

# sample metrics
land_actual = actual_sample[~mask]
land_pred_bc = pred_sample_bc[~mask]
if land_actual.size:
    difc = land_actual - land_pred_bc
    mse_sample_bc = float(np.nanmean(difc**2))
    mae_sample_bc = float(np.nanmean(np.abs(difc)))
    rmse_sample_bc = float(np.sqrt(mse_sample_bc))
else:
    mse_sample_bc = mae_sample_bc = rmse_sample_bc = float("nan")

if ssim_fn is not None:
    try:
        a = actual_sample.astype(np.float64).copy()
        pbc = pred_sample_bc.astype(np.float64).copy()
        pbc[mask] = a[mask]
        dr = float(a.max() - a.min())
        if dr == 0:
            dr = 1e-6
        s_sample_bc = float(ssim_fn(a, pbc, data_range=dr))
    except Exception:
        s_sample_bc = float("nan")
else:
    s_sample_bc = float("nan")

print(
    f"Sample metrics (downsampled target idx {ds_target_idx}): AFTER BC+BLEND -> MSE={mse_sample_bc:.6f}, MAE={mae_sample_bc:.6f}, RMSE={rmse_sample_bc:.6f}, SSIM={s_sample_bc}"
)


# ----------------- lat/lon and plotting (robust) ---------------
def find_first_nc(folder):
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".nc"):
            return os.path.join(folder, fn)
    raise FileNotFoundError


def find_first_nc_recursive(root_folder):
    for dirpath, _, files in os.walk(root_folder):
        for fn in sorted(files):
            if fn.lower().endswith(".nc"):
                p = os.path.join(dirpath, fn)
                try:
                    ds_try = Dataset(p)
                    vars_set = set(ds_try.variables.keys())
                    lat_cands = ("latitude", "lat", "nav_lat")
                    lon_cands = ("longitude", "lon", "nav_lon")
                    has_lat = any(n in vars_set for n in lat_cands)
                    has_lon = any(n in vars_set for n in lon_cands)
                    has_t2m = "t2m" in vars_set
                    ds_try.close()
                    if (has_lat and has_lon) or has_t2m:
                        return p
                except Exception:
                    continue
    raise FileNotFoundError(
        f"No readable .nc with lat/lon (or t2m) found under {root_folder}"
    )


nc0 = None
try:
    nc0 = find_first_nc_recursive(folder_path)
except FileNotFoundError:
    try:
        nc0 = find_first_nc(folder_path)
    except Exception:
        nc0 = None

if nc0 is None:
    # fallback: use grid indices
    print("No .nc found for lat/lon; using grid indices for extent/ticks.")
    lats = np.arange(H).astype(float)
    lons = np.arange(W).astype(float)
    origin = "lower"
else:
    print("Using .nc for lat/lon from:", nc0)
    ds0 = Dataset(nc0)
    lat_name = next(
        (n for n in ("latitude", "lat", "nav_lat") if n in ds0.variables), None
    )
    lon_name = next(
        (n for n in ("longitude", "lon", "nav_lon") if n in ds0.variables), None
    )
    if lat_name is None or lon_name is None:
        if "t2m" in ds0.variables:
            arr = ds0.variables["t2m"][:]
            ds0.close()
            if arr.ndim == 3:
                H_, W_ = arr.shape[1], arr.shape[2]
            elif arr.ndim == 2:
                H_, W_ = arr.shape[0], arr.shape[1]
            else:
                raise RuntimeError(f"Unexpected t2m dims in {nc0}: {arr.shape}")
            lats = np.arange(H_).astype(float)
            lons = np.arange(W_).astype(float)
            origin = "lower"
            print("Warning: lat/lon not found; using integer grid coords.")
        else:
            vars_list = list(ds0.variables.keys())
            ds0.close()
            raise RuntimeError(
                f"Found {nc0} but couldn't find latitude/longitude variables. Variables: {vars_list}"
            )
    else:
        lats = ds0.variables[lat_name][:].astype(float)
        lons = ds0.variables[lon_name][:].astype(float)
        ds0.close()
        origin = "lower"
        if lats.size > 1 and lats[0] > lats[-1]:
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

err_data = (actual_masked - pred_masked).filled(np.nan)
err_vals = err_data[~np.isnan(err_data)]
if err_vals.size == 0:
    vmin_err, vmax_err = -0.5, 0.5
else:
    vmin_err = float(np.nanmin(err_vals))
    vmax_err = float(np.nanmax(err_vals))
    if np.isclose(vmin_err, vmax_err):
        pad = max(1e-4, abs(vmin_err) * 0.001)
        vmin_err -= pad
        vmax_err += pad

err_ticks = [vmin_err, 0.0, vmax_err]

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
    vmin=vmin_err,
    vmax=vmax_err,
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
        cbar.set_ticks(err_ticks)
        cbar.ax.set_yticklabels([f"{v:.003f}" for v in err_ticks])
        pretty_cb(cbar, fmt="%.3f")
    else:
        cbar.set_label("2m Temperature (units)", fontsize=10)
        pretty_cb(cbar, fmt="%.2f")

metrics_text = f"MSE: {mse_sample_bc:.6f}   MAE: {mae_sample_bc:.6f}   RMSE: {rmse_sample_bc:.6f}   SSIM: {s_sample_bc:.6f}"
plt.tight_layout(rect=[0, 0.15, 1, 0.94])
fig.text(
    0.5,
    0.02,
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
print("Saved slope/intercept maps and climatology in:", out_dir)
print("Done.")
