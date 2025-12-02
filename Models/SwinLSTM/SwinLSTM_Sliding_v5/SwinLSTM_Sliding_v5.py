# run_swinlstm_full.py
import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset, num2date
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from tqdm import tqdm

# optional metrics
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

# ---------------- CONFIG ----------------
folder_path = "../../Dataset"  # top-level folder with '2020','2021',... subfolders
out_dir = "SwinLSTM_Sliding_v5"
os.makedirs(out_dir, exist_ok=True)

input_len = 8
target_offset = 4
SAMPLE_STEP = 3  # downsample raw frames by this (every 3 hours)

batch_size = 6
lr = 2e-4
epochs = 80
val_split = 0.20  # chronological split on 2020-2023
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZATION = True
PRINT_EVERY = 1

hidden_dim = 128
num_layers = 2
window_size = 4
num_heads = 4
dropout_p = 0.08

GRAD_WEIGHT = 0.8

mpl.rcParams["font.family"] = "Times New Roman"
# ---------------- end config ----------------

torch.manual_seed(seed)
np.random.seed(seed)

print(f"✅ Using device: {device}")


# ---------------- Dataset (frames + optional timestamps) ----------------
class SlidingMaskDataset(TorchDataset):
    """
    Loads frames (t2m) from year/month NetCDFs under folder_path/year/*.nc
    Builds:
      - self.frames: list of 2D arrays (float32)
      - self.times: list of numpy.datetime64 objects or None entries (one per frame) when available
      - self.sea_mask: True where all frames are nan (sea)
      - sliding windows defined by self.starts for input_len + target_offset
    """

    def __init__(
        self,
        folder_path,
        input_len=8,
        target_offset=4,
        sample_step=1,
        include_years=None,
    ):
        self.frames = []
        self.times = []  # optional timestamps aligned to frames list (one per frame)
        self.input_len = input_len
        self.target_offset = target_offset
        self.sample_step = int(sample_step)

        # gather year folders
        years = sorted(
            [
                d
                for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d))
            ]
        )
        if include_years is not None:
            years = [y for y in years if str(y) in set(map(str, include_years))]
        # month ordering by filename if possible
        month_map = {
            m: i + 1
            for i, m in enumerate(
                [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
            )
        }

        files_paths = []
        for y in years:
            yd = os.path.join(folder_path, y)
            month_files = [f for f in sorted(os.listdir(yd)) if f.endswith(".nc")]

            def month_key(fn):
                name = os.path.splitext(fn)[0].lower()
                return month_map.get(name, 999), fn

            month_files = sorted(month_files, key=month_key)
            for mf in month_files:
                files_paths.append((y, os.path.join(yd, mf)))

        if not files_paths:
            raise ValueError(
                f"No .nc files found under {folder_path} for years {years}"
            )

        # iterate files and extract frames & times
        for year, path in files_paths:
            try:
                ds = Dataset(path)
                if "t2m" not in ds.variables:
                    ds.close()
                    continue
                var = ds.variables["t2m"]
                arr = np.array(var[:])  # usually shape (T,H,W)
                # handle missing/fill values
                if hasattr(var, "_FillValue"):
                    arr = np.where(arr == var._FillValue, np.nan, arr)
                if hasattr(var, "missing_value"):
                    arr = np.where(arr == var.missing_value, np.nan, arr)

                # attempt to read time variable if present
                times_this_file = []
                if "time" in ds.variables:
                    try:
                        tvar = ds.variables["time"]
                        # convert netCDF time to datetimes
                        try:
                            tvals = num2date(tvar[:], tvar.units)
                        except Exception:
                            # fallback: try direct cast of numeric values as hours since epoch
                            tvals = None
                        if tvals is not None:
                            # ensure timezone-naive numpy.datetime64
                            for tv in tvals:
                                # tv may be datetime.datetime or cftime; convert to numpy.datetime64
                                try:
                                    npdt = np.datetime64(tv.isoformat())
                                except Exception:
                                    try:
                                        npdt = np.datetime64(str(tv))
                                    except Exception:
                                        npdt = None
                                times_this_file.append(npdt)
                    except Exception:
                        times_this_file = []

                # push frames and times
                if arr.ndim == 3:
                    T = arr.shape[0]
                    for t in range(T):
                        self.frames.append(arr[t].astype(np.float32))
                        if len(times_this_file) == T:
                            self.times.append(times_this_file[t])
                        else:
                            self.times.append(None)
                elif arr.ndim == 2:
                    self.frames.append(arr.astype(np.float32))
                    if len(times_this_file) == 1:
                        self.times.append(times_this_file[0])
                    else:
                        self.times.append(None)
                ds.close()
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if len(self.frames) == 0:
            raise ValueError("No frames loaded")

        # downsample in time
        if self.sample_step > 1:
            self.frames = self.frames[:: self.sample_step]
            self.times = self.times[:: self.sample_step]

        # ensure consistent shape
        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent frame shapes: {shapes}")
        self.H, self.W = self.frames[0].shape

        stacked = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stacked).all(axis=0)

        # sliding windows for input->target
        starts = []
        self.sample_times = []  # aligned to sliding windows (target timestamp)
        for s in range(len(self.frames) - input_len - target_offset + 1):
            starts.append(s)
            tgt_idx = s + input_len - 1 + target_offset
            # store the timestamp for the target if available
            self.sample_times.append(
                self.times[tgt_idx] if tgt_idx < len(self.times) else None
            )

        if len(starts) == 0:
            raise ValueError("Not enough frames for chosen input_len/target_offset")
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len
        inp = np.stack(self.frames[s:e], axis=0)  # C,H,W
        tgt = self.frames[e - 1 + self.target_offset]  # H,W

        # fill NaNs per frame using land mean
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

    def get_sample_time(self, idx):
        return self.sample_times[idx]  # numpy.datetime64 or None


# ---------------- SwinLSTM model ----------------
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: B, N, C
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # B,N,heads,hd
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
    def __init__(self, input_dim, hidden_dim, window_size=4, num_heads=4, dropout=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.in_proj = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.attn = LocalWindowAttention(dim=hidden_dim, num_heads=num_heads)
        self.gates = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)
        self.dropout = dropout

    def _partition_windows(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        num_win = (Hp // ws) * (Wp // ws)
        x = x.view(B * num_win, ws * ws, C)
        return x, (Hp, Wp, pad_h, pad_w)

    def _unpartition_windows(self, x_w, Hp, Wp, pad_h, pad_w, H, W):
        B_numwin, N, C = x_w.shape
        ws = self.window_size
        num_h = Hp // ws
        num_w = Wp // ws
        B = B_numwin // (num_h * num_w)
        x = x_w.view(B, num_h, num_w, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

    def forward(self, x, hidden):
        h, c = hidden
        B, _, H, W = x.shape
        comb = torch.cat([x, h], dim=1)
        proj = self.in_proj(comb)
        x_win, (Hp, Wp, pad_h, pad_w) = self._partition_windows(proj)
        attn_out = self.attn(x_win)
        attn_map = self._unpartition_windows(attn_out, Hp, Wp, pad_h, pad_w, H, W)
        attn_map = self.gn(attn_map)
        if self.training and self.dropout > 0:
            attn_map = F.dropout2d(attn_map, p=self.dropout)
        g = self.gates(attn_map)
        ci, cf, co, cg = torch.chunk(g, 4, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        o = torch.sigmoid(co)
        gg = torch.tanh(cg)
        cnext = f * c + i * gg
        hnext = o * torch.tanh(cnext)
        hnext = self.gn(hnext)
        if self.training and self.dropout > 0:
            hnext = F.dropout2d(hnext, p=self.dropout)
        return hnext, cnext

    def init_hidden(self, batch, H, W, device):
        return (
            torch.zeros(batch, self.hidden_dim, H, W, device=device),
            torch.zeros(batch, self.hidden_dim, H, W, device=device),
        )


class ResidualSwinLSTMWithRefine(nn.Module):
    def __init__(
        self,
        in_channels=8,
        hidden_dim=128,
        num_layers=2,
        window_size=4,
        num_heads=4,
        dropout=0.08,
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                SwinLSTMCell(
                    in_dim,
                    hidden_dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.input_skip_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        hiddens = [l.init_hidden(B, H, W, device) for l in self.layers]
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
        skip = self.input_skip_conv(x[:, -1:, :, :])
        out = out + skip
        return out


# ---------------- helpers ----------------
def pretty_month_xticks(ax, times):
    """Set monthly tick labels along x axis given numpy datetime64 array or indices fallback."""
    try:
        import pandas as pd

        times_pd = pd.to_datetime(times)
        # set ticks at first day of each month present
        months = pd.DatetimeIndex(times_pd)
        mons = months.month
        uniq_mons = sorted(set(mons))
        ticks = []
        labels = []
        for m in uniq_mons:
            # choose first index where month==m
            idx = int(np.where(mons == m)[0][0])
            ticks.append(idx)
            labels.append(pd.Timestamp(times_pd[idx]).strftime("%b"))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    except Exception:
        # fallback - set 12 equally spaced ticks
        N = len(times)
        ticks = np.linspace(0, N - 1, 12).astype(int)
        labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def compute_mean_ssim(preds, actuals, sea_mask):
    if ssim_fn is None:
        return None
    # preds, actuals: (N,1,H,W) numpy
    preds_np = np.asarray(preds)
    acts_np = np.asarray(actuals)
    N = preds_np.shape[0]
    H = preds_np.shape[-2]
    W = preds_np.shape[-1]
    mask = np.asarray(sea_mask)
    vals = []
    for i in range(N):
        a = acts_np[i, 0].astype(np.float64).copy()
        p = preds_np[i, 0].astype(np.float64).copy()
        # replace sea pixels in prediction with actual to avoid affecting SSIM
        try:
            p[mask] = a[mask]
        except Exception:
            pass
        dr = float(a.max() - a.min())
        if dr == 0.0:
            dr = 1e-6
        try:
            s = ssim_fn(a, p, data_range=dr)
        except Exception:
            s = np.nan
        vals.append(s)
    vals = np.array(vals, dtype=np.float64)
    if np.all(np.isnan(vals)):
        return None
    return float(np.nanmean(vals))


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


# ---------------- Prepare datasets ----------------
print("Loading dataset and preparing train/val/test splits (chronological)...")
all_years = sorted(
    [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
)
print("Detected year folders:", all_years)
TEST_YEAR = "2024" if "2024" in all_years else all_years[-1]
trainval_years = [y for y in all_years if y != TEST_YEAR]
if not trainval_years:
    raise ValueError("No years left for train/val after excluding test year")

print(f"Using years {trainval_years} for train+val. TEST set: {TEST_YEAR}")

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
n_test = len(dataset_test)
print(f"Train+Val samples: {n_tv}   Test samples: {n_test}")

# chronologically split train/val: first 80% -> train, last 20% -> val
indices = np.arange(n_tv)
split_idx = int(np.floor((1.0 - val_split) * n_tv))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]
train_set = Subset(dataset_trainval, train_indices)
val_set = Subset(dataset_trainval, val_indices)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train/Val sizes -> train={len(train_set)}, val={len(val_set)}")

sea_mask = dataset_trainval.sea_mask
H, W = dataset_trainval.H, dataset_trainval.W

# ---------------- climatology & normalization from training targets ----------------
print("Computing climatology from training targets...")
clim_sum = np.zeros((H, W), dtype=np.float64)
count = 0
for i in train_indices:
    _, y = dataset_trainval[i]
    clim_sum += y.numpy().squeeze(0)
    count += 1
climatology = (clim_sum / max(1, count)).astype(np.float32)
np.save(os.path.join(out_dir, "climatology.npy"), climatology)
print("Saved climatology.")


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


norm_mean, norm_std = 0.0, 1.0
if USE_NORMALIZATION:
    print("Computing normalization (mean/std) from anomaly training set...")
    norm_mean, norm_std = compute_norm_from_anomalies(
        dataset_trainval, train_indices, climatology
    )
    print("norm mean, std:", norm_mean, norm_std)

# prepare broadcast tensors
clim_t = (
    torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)
)  # 1x1xHxW
sea_mask_t = torch.from_numpy(sea_mask).to(device)
land_mask_t = (~sea_mask_t).to(device).unsqueeze(0).unsqueeze(0)

# ---------------- build model ----------------
model = ResidualSwinLSTMWithRefine(
    in_channels=input_len,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    window_size=window_size,
    num_heads=num_heads,
    dropout=dropout_p,
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=6
)
criterion_map = nn.SmoothL1Loss(reduction="none")

# load checkpoint if exists
ckpt_path = os.path.join(out_dir, "best_model.pth")
if os.path.exists(ckpt_path):
    print("Loading checkpoint:", ckpt_path)
    st = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(st)
    except Exception:
        model.load_state_dict(st, strict=False)
    print("Model loaded from checkpoint. Skipping training.")
    SKIP_TRAIN = True
else:
    SKIP_TRAIN = False

# ---------------- training loop ----------------
best_val_rmse = 1e9
best_state = None
patience = 12
no_improve = 0

if not SKIP_TRAIN:
    for epoch in range(1, epochs + 1):
        model.train()
        run = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)
            X_anom = X - clim_t
            y_anom = y - clim_t
            if USE_NORMALIZATION:
                X_anom = (X_anom - norm_mean) / norm_std
                y_anom = (y_anom - norm_mean) / norm_std

            opt.zero_grad()
            out = model(X_anom)
            if out.shape != y_anom.shape:
                out = F.interpolate(
                    out, size=y_anom.shape[2:], mode="bilinear", align_corners=False
                )

            map_loss = criterion_map(out, y_anom)
            mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
            masked_vals = map_loss.masked_select(mask)
            loss_basic = (
                masked_vals.mean()
                if masked_vals.numel()
                else torch.tensor(0.0, device=device)
            )

            # gradient loss on absolute
            if USE_NORMALIZATION:
                out_abs = out * norm_std + norm_mean
            else:
                out_abs = out
            out_abs = out_abs + clim_t
            grad_loss = gradient_loss_torch(
                out_abs, y, land_mask_t.expand(out_abs.shape[0], 1, H, W)
            )

            loss = loss_basic + GRAD_WEIGHT * grad_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.item()) * X.size(0)
            seen += X.size(0)
            pbar.set_postfix(batch_loss=f"{loss.item():.6f}")
        train_loss = run / max(1, seen)

        # validation
        model.eval()
        vr = 0.0
        vseen = 0
        all_val_preds = []
        all_val_actuals = []
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
                        outv,
                        size=yv_anom.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                map_loss = criterion_map(outv, yv_anom)
                mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
                masked_vals = map_loss.masked_select(mask)
                lv = (
                    masked_vals.mean()
                    if masked_vals.numel()
                    else torch.tensor(0.0, device=device)
                )
                vr += float(lv.item()) * Xv.size(0)
                vseen += Xv.size(0)

                if USE_NORMALIZATION:
                    outv = outv * norm_std + norm_mean
                outv_abs = outv + clim_t
                all_val_preds.append(outv_abs.cpu().numpy())
                all_val_actuals.append(yv.cpu().numpy())
        val_loss = vr / max(1, vseen)

        # compute val metrics
        if len(all_val_preds):
            preds_arr = np.concatenate(all_val_preds, axis=0)
            acts_arr = np.concatenate(all_val_actuals, axis=0)
            mask_flat = (~dataset_trainval.sea_mask).ravel()
            preds_flat = preds_arr.reshape(preds_arr.shape[0], -1)
            acts_flat = acts_arr.reshape(acts_arr.shape[0], -1)
            diffs = acts_flat[:, mask_flat] - preds_flat[:, mask_flat]
            mse = float(np.nanmean(diffs**2))
            mae = float(np.nanmean(np.abs(diffs)))
            rmse = float(np.sqrt(mse))
            mean_ssim_val = (
                compute_mean_ssim(preds_arr, acts_arr, dataset_trainval.sea_mask)
                if ssim_fn is not None
                else None
            )
        else:
            mse = mae = rmse = float("nan")
            mean_ssim_val = None

        sched.step(val_loss)
        if epoch % PRINT_EVERY == 0:
            print(
                f"Epoch {epoch:03d} Train={train_loss:.6f} Val={val_loss:.6f} | VAL MSE={mse:.6f} MAE={mae:.6f} RMSE={rmse:.6f} SSIM={mean_ssim_val}"
            )

        # checkpoint by val RMSE
        if not math.isnan(rmse) and rmse < best_val_rmse - 1e-6:
            best_val_rmse = rmse
            best_state = model.state_dict().copy()
            torch.save(best_state, ckpt_path)
            no_improve = 0
            print(f"Saved best model (val_rmse={rmse:.4f}).")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping (no improvement).")
                break

# final safeguard save
if best_state is not None:
    torch.save(best_state, ckpt_path)
    print("Final: saved best model.")
else:
    torch.save(model.state_dict(), ckpt_path)
    print("Final: saved current model state.")

# ---------------- Evaluate on TEST set (per-sample time series) ----------------
print("Evaluating on test set and collecting per-sample metrics...")

# We'll iterate over dataset_test samples directly (avoid DataLoader collating datetime)
test_N = len(dataset_test)
test_preds = np.empty((test_N, 1, H, W), dtype=np.float32)
test_actuals = np.empty((test_N, 1, H, W), dtype=np.float32)
test_times = []
model.eval()
with torch.no_grad():
    for i in tqdm(range(test_N), desc="Test inference"):
        Xs, ys = dataset_test[i]  # tensors
        Xs = Xs.unsqueeze(0).to(device) if Xs.dim() == 3 else Xs.to(device)
        ys = ys.unsqueeze(0).to(device) if ys.dim() == 3 else ys.to(device)
        X_anom = Xs - clim_t
        if USE_NORMALIZATION:
            X_anom = (X_anom - norm_mean) / norm_std
        out = model(X_anom)
        if out.shape[2:] != ys.shape[2:]:
            out = F.interpolate(
                out, size=ys.shape[2:], mode="bilinear", align_corners=False
            )
        if USE_NORMALIZATION:
            out = out * norm_std + norm_mean
        out_abs = out + clim_t
        test_preds[i] = out_abs.cpu().numpy()
        test_actuals[i] = ys.cpu().numpy()
        test_times.append(dataset_test.get_sample_time(i))

# If timestamps missing or many None, create fallback uniform timestamps across 2024
n_none = sum(1 for t in test_times if t is None)
if n_none > 0:
    print(
        f"Warning: {n_none}/{len(test_times)} test samples missing time metadata. Building fallback timestamps for 2024."
    )
    # build uniform timestamps: assume first sample corresponds to 2024-01-01 00:00 (UTC)
    # time step = SAMPLE_STEP hours
    start = np.datetime64("2024-01-01T00:00")
    step_hours = SAMPLE_STEP
    fallback_times = [
        start + np.timedelta64(int(step_hours * i), "h") for i in range(test_N)
    ]
    test_times = np.array(fallback_times)
else:
    test_times = np.array(test_times)

# compute per-sample metrics (land-only)
mask_flat = (~dataset_trainval.sea_mask).ravel()
mae_ts = np.full(test_N, np.nan, dtype=np.float32)
rmse_ts = np.full(test_N, np.nan, dtype=np.float32)
ssim_ts = np.full(test_N, np.nan, dtype=np.float32) if ssim_fn is not None else None

for i in range(test_N):
    P = test_preds[i].reshape(-1)[mask_flat]
    A = test_actuals[i].reshape(-1)[mask_flat]
    if P.size == 0:
        mae_ts[i] = np.nan
        rmse_ts[i] = np.nan
    else:
        dif = A - P
        mse = float(np.nanmean(dif**2))
        mae = float(np.nanmean(np.abs(dif)))
        mae_ts[i] = mae
        rmse_ts[i] = float(np.sqrt(mse))
    if ssim_fn is not None:
        val = compute_mean_ssim(
            test_preds[i : i + 1], test_actuals[i : i + 1], dataset_trainval.sea_mask
        )
        ssim_ts[i] = val if val is not None else np.nan

# Save metrics arrays
np.save(os.path.join(out_dir, "test_times.npy"), test_times.astype("datetime64[h]"))
np.save(os.path.join(out_dir, "test_mae.npy"), mae_ts)
np.save(os.path.join(out_dir, "test_rmse.npy"), rmse_ts)
if ssim_ts is not None:
    np.save(os.path.join(out_dir, "test_ssim.npy"), ssim_ts)

print("Saved test metric arrays.")


# ---------------- Plotting: three separate figures ----------------
def make_series_plot(y, times, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.arange(len(y)), y, linewidth=1.4)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_xlabel("Months →", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(alpha=0.25)
    # x ticks: monthly labels (attempt)
    pretty_month_xticks(ax, times)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fullp = os.path.join(out_dir, fname)
    plt.savefig(fullp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fullp)


# Titles and labels
title_ssim = f"Figure: Time series of SSIM for test data (2024) of SwinLSTM"
title_mae = f"Figure: Time series of MAE (units) for test data (2024) of SwinLSTM"
title_rmse = f"Figure: Time series of RMSE (units) for test data (2024) of SwinLSTM"

# ssim may be None
if ssim_ts is not None:
    make_series_plot(ssim_ts, test_times, "SSIM", "Figure4_SSIM_2024.png", title_ssim)
make_series_plot(mae_ts, test_times, "MAE (units)", "Figure5_MAE_2024.png", title_mae)
make_series_plot(
    rmse_ts, test_times, "RMSE (units)", "Figure6_RMSE_2024.png", title_rmse
)

print("All done. Results saved to:", out_dir)
