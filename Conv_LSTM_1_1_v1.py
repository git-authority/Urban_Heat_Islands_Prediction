import os
import io
import sys
import numpy as np
from netCDF4 import Dataset as ncDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import torch.nn.functional as F


# ---------------------- Dataset ----------------------
class TempSequenceDataset(Dataset):
    """Loads time-ordered 2D temperature frames from netCDF files and
    produces samples of `input_len` consecutive frames and a single target
    frame `target_offset` steps after the last input frame.

    Important: This dataset DOES NOT normalize temperatures (per user request).
    """

    def __init__(self, folder_path, input_len=8, target_offset=4):
        self.frames = []  # list of 2D arrays (H, W)
        self.input_len = input_len
        self.target_offset = target_offset

        # sort files to preserve temporal order as best as possible
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
        if len(files) == 0:
            raise ValueError(f"No .nc files found in {folder_path}")

        for file in files:
            path = os.path.join(folder_path, file)
            try:
                ds = ncDataset(path)
                if "t2m" not in ds.variables:
                    print(f"Skipping {file}: variable 't2m' not found")
                    continue

                arr = np.array(ds.variables["t2m"][:])

                # arr could be (time, H, W) or (H, W) if single timestep
                if arr.ndim == 3:
                    for t in range(arr.shape[0]):
                        frame = np.array(arr[t], dtype=np.float32)
                        # replace NaN in each frame with that frame's mean
                        if np.isnan(frame).any():
                            meanv = np.nanmean(frame)
                            frame = np.nan_to_num(frame, nan=meanv)
                        self.frames.append(frame)

                elif arr.ndim == 2:
                    frame = np.array(arr, dtype=np.float32)
                    if np.isnan(frame).any():
                        meanv = np.nanmean(frame)
                        frame = np.nan_to_num(frame, nan=meanv)
                    self.frames.append(frame)

                else:
                    print(f"Skipping {file}: unexpected t2m dims {arr.shape}")

            except Exception as e:
                print(f"Skipping {file}: {e}")

        if len(self.frames) == 0:
            raise ValueError("No valid frames found in dataset files.")

        # verify shapes are consistent
        shapes = {f.shape for f in self.frames}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent frame shapes found: {shapes}")

        self.H, self.W = self.frames[0].shape

        # Number of samples available
        self.num_samples = len(self.frames) - self.input_len - self.target_offset + 1
        if self.num_samples <= 0:
            raise ValueError(
                "Not enough frames for the given input_len and target_offset."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # inputs: frames idx .. idx+input_len-1, shape (C=input_len, H, W)
        start = idx
        end = idx + self.input_len
        input_frames = np.stack(self.frames[start:end], axis=0)  # (C,H,W)
        target_idx = end - 1 + self.target_offset
        target_frame = self.frames[target_idx]  # (H,W)

        # convert to torch tensors
        inputs = torch.from_numpy(input_frames).float()
        target = torch.from_numpy(target_frame).unsqueeze(0).float()  # (1,H,W)
        return inputs, target


# ---------------------- ConvLSTM model (drop-in) ----------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, hidden):
        # x: (B, input_dim, H, W)
        # hidden: (h_cur, c_cur) each (B, hidden_dim, H, W)
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        # split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        H, W = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
        )


class SameSizeConvLSTM(nn.Module):
    """
    Drop-in replacement model. Accepts input as (B, C=input_len, H, W)
    Internally treats C as time-steps T and runs ConvLSTM over T frames
    (with channel dim = 1 per time-step). Outputs (B,1,H,W) matching your original code.
    """

    def __init__(self, in_channels, hidden_dim=64, num_layers=2):
        """
        in_channels: number of input frames (this will be treated as T)
        hidden_dim: number of ConvLSTM feature maps
        num_layers: number of stacked ConvLSTM layers
        """
        super().__init__()
        self.input_len = in_channels  # number of time steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # build ConvLSTM layers; first layer input_dim=1 (each timestep is single-channel t2m)
        layers = []
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            layers.append(
                ConvLSTMCell(input_dim=in_dim, hidden_dim=hidden_dim, kernel_size=3)
            )
        self.layers = nn.ModuleList(layers)

        # final 1x1 conv to produce single-channel output
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C=input_len, H, W)
        B, C, H, W = x.size()
        device = x.device

        # treat channel axis as time -> create (B, T=C, 1, H, W)
        x_time = x.unsqueeze(2)  # (B, C, 1, H, W)
        # iterate over time steps
        hiddens = [layer.init_hidden(B, (H, W), device) for layer in self.layers]

        last_output = None
        for t in range(C):
            frame = x_time[:, t, :, :, :]  # (B, 1, H, W)
            inp = frame
            for li, layer in enumerate(self.layers):
                h_cur, c_cur = hiddens[li]
                h_next, c_next = layer(inp, (h_cur, c_cur))
                hiddens[li] = (h_next, c_next)
                inp = h_next  # input to next layer is hidden state
            last_output = inp  # last layer's hidden at time t

        out = self.final_conv(last_output)  # (B,1,H,W)
        return out


# ---------------------- Utilities ----------------------
class EarlyStoppingNoSave:
    def __init__(self, patience=30, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            # store best weights in-memory (no disk save)
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

    def load_best(self, model, device):
        if self.best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})


# ---------------------- Helper: verify plot saving ----------------------
def verify_plots_dir_can_write(path):
    """Create directory if needed and attempt to save a tiny test PNG there.
    Returns True if OK, otherwise raises an exception with a helpful message.
    """
    os.makedirs(path, exist_ok=True)
    test_path = os.path.join(path, "__plot_write_test.png")
    try:
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        fig.savefig(test_path)
        plt.close(fig)
        # remove test file
        try:
            os.remove(test_path)
        except Exception:
            pass
        return True
    except Exception as e:
        raise RuntimeError(f"Cannot write to plots directory '{path}': {e}")


# ---------------------- Training script ----------------------
if __name__ == "__main__":
    # ---------------- config ----------------
    folder = "Dataset/2024"  # change if needed
    output_dir = "Outputs_v1"
    plots_dir = os.path.join(output_dir, "plots")

    # Ensure output directories exist and are writable BEFORE any long work
    try:
        verify_plots_dir_can_write(plots_dir)
    except Exception as e:
        print("\nERROR: plot directory not writable - aborting before training.")
        print(e)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # For your use-case: 8 days * 8 slots/day = 64 input frames (3-hour spacing)
    input_len = 64
    target_offset = 4  # predict 4 steps after last input (21->next day 09 in 3-hour spacing example)
    batch_size = 8
    lr = 1e-4
    epochs = 200
    val_split = 0.2
    patience = 30
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device, "\n")

    # ---------------- dataset ----------------
    dataset = TempSequenceDataset(
        folder, input_len=input_len, target_offset=target_offset
    )
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(np.floor(val_split * n))
    val_idx = indices[:split]
    train_idx = indices[split:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # On Windows it's safer to use num_workers=0
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ---------------- model ----------------
    # Use the same interface: pass in_channels=input_len so the class treats channels as timesteps
    model = SameSizeConvLSTM(in_channels=input_len, hidden_dim=64, num_layers=2).to(
        device
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    early_stopper = EarlyStoppingNoSave(patience=patience)

    train_losses = []
    val_losses = []

    # ---------------- training ----------------
    print("Starting training...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        it = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train", leave=False)
        for X, y in pbar:
            # X: (B, C=input_len, H, W)
            X = X.to(device)
            y = y.to(device)  # (B,1,H,W)

            optimizer.zero_grad()
            outputs = model(X)

            # If shapes mismatch for any reason, resize outputs to match target
            if outputs.shape != y.shape:
                outputs = F.interpolate(
                    outputs, size=y.shape[2:], mode="bilinear", align_corners=False
                )

            loss = criterion(outputs, y)

            if torch.isnan(loss):
                raise ValueError("Loss became NaN. Check your data for invalid values.")

            loss.backward()

            # gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * X.size(0)
            it += X.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss = running_loss / it
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_running = 0.0
        val_it = 0

        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                outv = model(Xv)
                if outv.shape != yv.shape:
                    outv = F.interpolate(
                        outv, size=yv.shape[2:], mode="bilinear", align_corners=False
                    )
                lv = criterion(outv, yv)
                val_running += lv.item() * Xv.size(0)
                val_it += Xv.size(0)

        val_loss = val_running / max(1, val_it)
        val_losses.append(val_loss)

        # scheduler step
        scheduler.step(val_loss)

        # clear, spaced terminal output
        print("\n" + "=" * 60)
        print(
            f"Epoch {epoch:03d}:\n    Train Loss = {train_loss:.6f}\n    Val   Loss = {val_loss:.6f}\n"
        )

        # early stopping (no disk save)
        improved = early_stopper.step(val_loss, model)
        if improved:
            print(
                f"*** Validation loss improved to {early_stopper.best_loss:.6f} (in-memory best state saved) ***\n"
            )
        if early_stopper.should_stop():
            print(
                f"No improvement for {early_stopper.patience} epochs. Early stopping.\n"
            )
            break

    # restore best weights (in-memory) if available
    early_stopper.load_best(model, device)
    print("Restored in-memory best model weights (no disk save).\n")

    # ---------------- save train/val loss plot ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    loss_plot_path = os.path.join(plots_dir, "train_val_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to {loss_plot_path}\n")

    # ---------------- Predictions & Heatmaps (single combined figure) ----------------
    model.eval()
    sample_dir = os.path.join(plots_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # pick one batch from val_loader for visualization
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            preds = model(Xb)  # (B,1,H,W)
            if preds.shape != yb.shape:
                preds = F.interpolate(
                    preds, size=yb.shape[2:], mode="bilinear", align_corners=False
                )
            preds = preds.cpu().numpy()
            actuals = yb.cpu().numpy()
            inputs_np = Xb.cpu().numpy()  # (B,C,H,W)
            break

    sample_idx = 0
    actual = actuals[sample_idx, 0]
    predicted = preds[sample_idx, 0]
    error = actual - predicted

    # Determine temp vmin/vmax from actual & predicted for consistent gray scale
    combined_temp = np.concatenate([actual.flatten(), predicted.flatten()])
    tmin = np.min(combined_temp)
    tmax = np.max(combined_temp)

    # Error range symmetric
    max_abs = np.max(np.abs(error)) if np.max(np.abs(error)) > 0 else 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Use nearest interpolation to keep pixel blocks visible, and gray_r so that
    # higher temperature values map to white (hot) and lower to black (cool).
    im0 = axes[0].imshow(
        actual, cmap="gray", vmin=tmin, vmax=tmax, interpolation="nearest"
    )
    axes[0].set_title("Actual Heatmap")
    axes[0].axis("off")

    im1 = axes[1].imshow(
        predicted, cmap="gray_r", vmin=tmin, vmax=tmax, interpolation="nearest"
    )
    axes[1].set_title("Predicted Heatmap")
    axes[1].axis("off")

    # For error, user requested same black-white scheme. We map negative->dark, positive->bright.
    im2 = axes[2].imshow(
        error, cmap="gray_r", vmin=-max_abs, vmax=max_abs, interpolation="nearest"
    )
    axes[2].set_title("Error Heatmap (Actual - Predicted)")
    axes[2].axis("off")

    # colorbars: one for temp (shared for first two), one for error
    cbar_ax_temp = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    temp_cb = fig.colorbar(im1, cax=cbar_ax_temp)
    temp_cb.set_label("Temperature (raw units)")
    temp_cb.ax.invert_yaxis()  # keep white at top if preferred visually

    cbar_ax_err = fig.add_axes([0.92, 0.12, 0.02, 0.3])
    err_cb = fig.colorbar(im2, cax=cbar_ax_err)
    err_cb.set_label("Error (raw units)")

    plt.suptitle("Actual | Predicted | Error (grayscale: white=hot, black=cool)")
    combined_path = os.path.join(sample_dir, "heatmaps_combined.png")
    plt.savefig(combined_path, bbox_inches="tight")
    plt.close()

    print("Saved combined heatmap image:")
    print(f" - {combined_path}\n")

    print("All done.\n")
