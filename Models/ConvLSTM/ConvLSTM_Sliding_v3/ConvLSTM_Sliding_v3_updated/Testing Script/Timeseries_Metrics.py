import os
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

# Optional imports
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.family"] = "Times New Roman"

# ---------------- CONFIG ----------------
folder_path = "../../../../Dataset/2024"  # 2024 data with t2m
out_dir = "./"  # current folder
model_path = os.path.join(out_dir, "best_model.pth")
clim_path = os.path.join(out_dir, "climatology.npy")

ts_out_dir = os.path.join(out_dir, "Timeseries_Metrics")
os.makedirs(ts_out_dir, exist_ok=True)

# must match training
input_len = 8
target_offset = 4
SAMPLE_STEP = 3
val_split = 0.18
seed = 42

hidden_dim = 192
num_layers = 3
kernel_size = 5
dropout_p = 0.05
USE_NORMALIZATION = True

epochs = 60
batch_size = 8
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device} | SSIM available: {ssim_fn is not None}")

torch.manual_seed(seed)
np.random.seed(seed)


# ---------------- Model definition (same as training) ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, dropout_p=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad
        )
        self.dropout_p = dropout_p
        self.gn = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)

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
    def __init__(
        self, in_channels, hidden_dim=128, num_layers=3, kernel_size=5, dropout_p=0.05
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                ConvLSTMCell(
                    in_dim, hidden_dim, kernel_size=kernel_size, dropout_p=dropout_p
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
        hiddens = [l.init_hidden(B, (H, W), x.device) for l in self.layers]
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


# ---------------- Dataset (no timestamps here) ----------------
class SlidingMaskDataset(TorchDataset):
    """
    Sliding windows over 2024 with filling and temporal downsampling.
    """

    def __init__(self, folder_path, input_len, target_offset, sample_step):
        self.frames = []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if not files:
            raise RuntimeError(f"No .nc files in {folder_path}")

        for fn in files:
            path = os.path.join(folder_path, fn)
            try:
                ds = Dataset(path)
                if "t2m" not in ds.variables:
                    ds.close()
                    continue
                arr = np.array(ds["t2m"][:], dtype=np.float32)
                if arr.ndim == 3:
                    for t in range(arr.shape[0]):
                        self.frames.append(arr[t])
                elif arr.ndim == 2:
                    self.frames.append(arr)
                ds.close()
            except Exception as e:
                print(f"Skipping {fn}: {e}")

        if len(self.frames) == 0:
            raise RuntimeError("No frames loaded")

        # temporal downsampling
        self.frames = self.frames[::sample_step]

        self.input_len = input_len
        self.target_offset = target_offset

        stack = np.stack(self.frames, axis=0)
        self.sea_mask = np.isnan(stack).all(axis=0)
        self.H, self.W = self.frames[0].shape

        self.starts = []
        for s in range(len(self.frames) - input_len - target_offset + 1):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.input_len

        inp = np.stack(self.frames[s:e], axis=0).astype(np.float32)
        tgt = self.frames[e - 1 + self.target_offset].astype(np.float32)
        mask = self.sea_mask

        # fill NaNs with land mean
        inp_filled = np.empty_like(inp, dtype=np.float32)
        for i in range(inp.shape[0]):
            frame = inp[i]
            land_vals = frame[~mask]
            fill = float(np.nanmean(land_vals)) if land_vals.size else 0.0
            inp_filled[i] = np.where(np.isnan(frame), fill, frame)

        land_vals_tgt = tgt[~mask]
        fill_t = float(np.nanmean(land_vals_tgt)) if land_vals_tgt.size else 0.0
        tgt_filled = np.where(np.isnan(tgt), fill_t, tgt).astype(np.float32)

        return (
            torch.from_numpy(inp_filled).float(),  # C,H,W
            torch.from_numpy(tgt_filled).unsqueeze(0).float(),  # 1,H,W
        )


# ---------------- Normalization helper ----------------
def compute_norm_from_anomalies(dataset, train_idx, climatology):
    s = 0.0
    ss = 0.0
    cnt = 0
    clim = climatology.astype(np.float64)
    for i in train_idx:
        X, y = dataset[i]
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


def compute_mean_ssim(preds, actuals, sea_mask):
    if ssim_fn is None:
        return None
    preds_np = preds[:, 0].astype(np.float64)
    actuals_np = actuals[:, 0].astype(np.float64)
    mask = sea_mask
    vals = []
    for i in range(preds_np.shape[0]):
        a = actuals_np[i].copy()
        p = preds_np[i].copy()
        p[mask] = a[mask]
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


# ---------------- Plotting helper (stem-style) ----------------
def plot_stem_timeseries(times, values, title, ylabel, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(times, values, "o", markersize=3)
    for t, v in zip(times, values):
        ax.vlines(t, ymin=0.0, ymax=v, colors="C0", alpha=0.35, linewidth=0.7)

    ax.set_title(title, fontsize=18, pad=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Time (2024)", fontsize=14)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=35)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


# ---------------- MAIN ----------------
def main():
    print("📦 Loading 2024 dataset...")
    dataset = SlidingMaskDataset(folder_path, input_len, target_offset, SAMPLE_STEP)
    print(f"Total samples (windows): {len(dataset)}")

    sea_mask = dataset.sea_mask
    H, W = dataset.H, dataset.W

    # split indices for train/val (for norm + possible training)
    indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(np.floor(val_split * len(dataset)))
    val_idx = indices[:split]
    train_idx = indices[split:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(
        f"Downsampled frames: {len(dataset.frames)}, windows: train={len(train_set)}, val={len(val_set)}"
    )

    # ---------- climatology ----------
    if os.path.exists(clim_path):
        print("Loading climatology from", clim_path)
        climatology = np.load(clim_path).astype(np.float32)
        if climatology.shape != (H, W):
            raise RuntimeError(f"Climatology shape {climatology.shape} != {(H, W)}")
    else:
        print("Computing climatology from training targets...")
        clim_sum = np.zeros((H, W), dtype=np.float64)
        count = 0
        for i in train_idx:
            _, y = dataset[i]
            clim_sum += y.numpy().squeeze(0)
            count += 1
        climatology = (clim_sum / max(1, count)).astype(np.float32)
        np.save(clim_path, climatology)
        print("Saved climatology to:", clim_path)

    # ---------- normalization ----------
    if USE_NORMALIZATION:
        print("📏 Computing normalization mean/std from training anomalies...")
        norm_mean, norm_std = compute_norm_from_anomalies(
            dataset, train_idx, climatology
        )
    else:
        norm_mean, norm_std = 0.0, 1.0
    print(f"norm_mean = {norm_mean:.6f}, norm_std = {norm_std:.6f}")

    clim_t = (
        torch.from_numpy(climatology).float().to(device).unsqueeze(0).unsqueeze(0)
    )  # 1x1xHxW
    sea_mask_t = torch.from_numpy(sea_mask).to(device)
    land_mask_t = (~sea_mask_t).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW

    # ---------- model ----------
    model = ResidualConvLSTMWithRefine(
        in_channels=input_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout_p=dropout_p,
    ).to(device)

    # ---------- training if no checkpoint ----------
    if os.path.exists(model_path):
        print("🧠 Loading trained model from", model_path)
        state = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception as e:
            print("Warning: strict load failed:", e)
            model.load_state_dict(state, strict=False)
        model.to(device)
    else:
        print("No checkpoint found — training model on 2024 data...")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=6
        )
        criterion = nn.SmoothL1Loss(reduction="none")

        best_val_rmse = 1e9
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            run = 0.0
            seen = 0
            pbar = tqdm(train_loader, ncols=80, desc=f"Epoch {epoch}/{epochs}")
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
                        out,
                        size=y_anom.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                map_loss = criterion(out, y_anom)
                mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
                vals = map_loss.masked_select(mask)
                loss = (
                    vals.mean()
                    if vals.numel() > 0
                    else torch.tensor(0.0, device=device)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                run += loss.item() * X.size(0)
                seen += X.size(0)
                pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

            train_loss = run / max(1, seen)

            # validation
            model.eval()
            vr = 0.0
            vseen = 0
            val_preds = []
            val_actuals = []
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

                    map_loss = criterion(outv, yv_anom)
                    mask = land_mask_t.expand(map_loss.shape[0], 1, H, W)
                    vals = map_loss.masked_select(mask)
                    lv = (
                        vals.mean()
                        if vals.numel() > 0
                        else torch.tensor(0.0, device=device)
                    )
                    vr += lv.item() * Xv.size(0)
                    vseen += Xv.size(0)

                    if USE_NORMALIZATION:
                        outv = outv * norm_std + norm_mean
                    outv_abs = outv + clim_t
                    val_preds.append(outv_abs.cpu().numpy())
                    val_actuals.append(yv.cpu().numpy())

            val_loss = vr / max(1, vseen)
            sched.step(val_loss)

            preds_arr = (
                np.concatenate(val_preds, axis=0)
                if len(val_preds)
                else np.empty((0, 1, H, W))
            )
            actuals_arr = (
                np.concatenate(val_actuals, axis=0)
                if len(val_actuals)
                else np.empty((0, 1, H, W))
            )

            if preds_arr.size:
                mask_flat = (~dataset.sea_mask).ravel()
                pf = preds_arr.reshape(preds_arr.shape[0], -1)
                af = actuals_arr.reshape(actuals_arr.shape[0], -1)
                dif = af[:, mask_flat] - pf[:, mask_flat]
                mse = float(np.nanmean(dif**2))
                rmse = float(np.sqrt(mse))
                mae = float(np.nanmean(np.abs(dif)))
                ssim_val = compute_mean_ssim(preds_arr, actuals_arr, dataset.sea_mask)
            else:
                mse = mae = rmse = float("nan")
                ssim_val = None

            print(
                f"Epoch {epoch:03d} Train={train_loss:.6f} Val={val_loss:.6f} | "
                f"VAL RMSE={rmse:.6f} MAE={mae:.6f} SSIM={ssim_val}"
            )

            if preds_arr.size and rmse < best_val_rmse:
                best_val_rmse = rmse
                best_state = model.state_dict().copy()
                torch.save(best_state, model_path)
                print(f"✅ Saved new best model -> {model_path}")

        if best_state is None:
            best_state = model.state_dict()
            torch.save(best_state, model_path)
            print("Saved final model to", model_path)
        model.load_state_dict(best_state)

    # ---------- INFERENCE + METRICS (time series) ----------
    print("🚀 Running inference over all 2024 windows...")
    model.eval()
    times = []
    ssim_list = []
    mae_list = []
    rmse_list = []

    num_frames = len(dataset.frames)
    start_time = datetime(2024, 1, 1, 0, 0)
    end_time = datetime(2024, 12, 31, 23, 59)
    total_seconds = (end_time - start_time).total_seconds()

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), ncols=80):
            X, y = dataset[idx]

            X = X.unsqueeze(0).to(device)  # 1,C,H,W
            y = y.unsqueeze(0).to(device)  # 1,1,H,W

            X_anom = X - clim_t
            if USE_NORMALIZATION:
                X_anom = (X_anom - norm_mean) / norm_std

            pred = model(X_anom)
            if pred.shape[2:] != y.shape[2:]:
                pred = F.interpolate(
                    pred, size=y.shape[2:], mode="bilinear", align_corners=False
                )

            if USE_NORMALIZATION:
                pred = pred * norm_std + norm_mean
            pred_abs = pred + clim_t  # 1,1,H,W

            p2d = pred_abs[0, 0].detach().cpu().numpy()
            a2d = y[0, 0].detach().cpu().numpy()

            p_land = p2d[~sea_mask]
            a_land = a2d[~sea_mask]
            diff = a_land - p_land

            mse = float(np.mean(diff**2))
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(mse))

            # SSIM (force sea = actual)
            if ssim_fn is not None:
                p_for_ssim = p2d.copy()
                p_for_ssim[sea_mask] = a2d[sea_mask]
                dr = float(a2d.max() - a2d.min())
                if dr == 0.0:
                    dr = 1e-6
                try:
                    ssim_val = float(ssim_fn(a2d, p_for_ssim, data_range=dr))
                except Exception:
                    ssim_val = float("nan")
            else:
                ssim_val = float("nan")

            ssim_list.append(ssim_val)
            mae_list.append(mae)
            rmse_list.append(rmse)

            # synthetic timestamp: map target-frame index into 2024
            s = dataset.starts[idx]
            target_idx = s + input_len - 1 + target_offset
            alpha = target_idx / max(1, num_frames - 1)
            sec_off = alpha * total_seconds
            ts = start_time + timedelta(seconds=sec_off)
            if ts.year < 2024:
                ts = start_time
            elif ts.year > 2024:
                ts = end_time
            times.append(ts)

    print("✅ Finished inference & metrics (SSIM / MAE / RMSE).")

    # ---------- plots ----------
    plot_stem_timeseries(
        times,
        ssim_list,
        "Figure 1: Time series of SSIM for test data (2024) of ConvLSTM (refine)",
        "SSIM",
        os.path.join(ts_out_dir, "Figure1_SSIM_timeseries_2024.png"),
    )

    plot_stem_timeseries(
        times,
        mae_list,
        "Figure 2: Time series of MAE for test data (2024) of ConvLSTM (refine)",
        "MAE (normalized)",
        os.path.join(ts_out_dir, "Figure2_MAE_timeseries_2024.png"),
    )

    plot_stem_timeseries(
        times,
        rmse_list,
        "Figure 3: Time series of RMSE for test data (2024) of ConvLSTM (refine)",
        "RMSE (normalized)",
        os.path.join(ts_out_dir, "Figure3_RMSE_timeseries_2024.png"),
    )

    print("🎉 Done. SSIM / MAE / RMSE plots saved in:", ts_out_dir)


if __name__ == "__main__":
    main()
