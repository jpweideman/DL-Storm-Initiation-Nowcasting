import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import wandb
import os
import random
from tqdm import tqdm
import ast

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Dataset definitions (copied from train_conv_lstm.py)
class RadarWindowDataset(Dataset):
    def __init__(self, cube_norm, seq_in, seq_out, maxv=85.0):
        self.cube = cube_norm
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.maxv = maxv
        self.last = cube_norm.shape[0] - seq_in - seq_out + 1
    def __len__(self):
        return self.last
    def __getitem__(self, i):
        X = np.maximum(self.cube[i:i+self.seq_in], 0) / (self.maxv + 1e-6)
        Y = np.maximum(self.cube[i+self.seq_in:i+self.seq_in+self.seq_out], 0) / (self.maxv + 1e-6)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32).squeeze(0)
        return torch.from_numpy(X), torch.from_numpy(Y)

class PatchRadarWindowDataset(Dataset):
    def __init__(self, cube_norm, seq_in, seq_out, patch_size=64, patch_stride=64, patch_thresh=0.4, patch_frac=0.15, patch_index_path=None, maxv=85.0):
        self.cube = cube_norm
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_thresh = patch_thresh
        self.patch_frac = patch_frac
        self.maxv = maxv
        self.patches = []  # List of (t, y, x)
        T, C, H, W = cube_norm.shape
        last = T - seq_in - seq_out + 1
        if patch_index_path is not None and os.path.exists(patch_index_path):
            print(f"Loading patch indices from {patch_index_path}")
            self.patches = np.load(patch_index_path, allow_pickle=True).tolist()
        else:
            for t in tqdm(range(last), desc="Extracting patches"):
                for y in range(0, H - patch_size + 1, patch_stride):
                    for x in range(0, W - patch_size + 1, patch_stride):
                        if y + patch_size <= H and x + patch_size <= W:
                            Y_patch = np.maximum(self.cube[t+seq_in:t+seq_in+seq_out, :, y:y+patch_size, x:x+patch_size], 0) / (self.maxv + 1e-6)
                            total_pix = Y_patch.size
                            n_above = (Y_patch > patch_thresh).sum()
                            if n_above / total_pix >= patch_frac:
                                self.patches.append((t, y, x))
            if patch_index_path is not None:
                np.save(patch_index_path, np.array(self.patches, dtype=object))
                print(f"Saved patch indices to {patch_index_path}")
    def __len__(self):
        return len(self.patches)
    def __getitem__(self, i):
        t, y, x = self.patches[i]
        X_patch = np.maximum(self.cube[t:t+self.seq_in, :, y:y+self.patch_size, x:x+self.patch_size], 0) / (self.maxv + 1e-6)
        Y_patch = np.maximum(self.cube[t+self.seq_in:t+self.seq_in+self.seq_out, :, y:y+self.patch_size, x:x+self.patch_size], 0) / (self.maxv + 1e-6)
        X_patch = X_patch.astype(np.float32)
        Y_patch = Y_patch.astype(np.float32).squeeze(0)
        return torch.from_numpy(X_patch), torch.from_numpy(Y_patch), t, y, x

# Weighted MSE loss (copied)


# TrajGRU cell and network implementation
class TrajGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, L=5, zoneout=0.0):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.L = L  # number of allowed flow fields
        self.zoneout = zoneout
        padding = kernel_size // 2
        self.conv_x = nn.Conv2d(input_channels, hidden_channels * 3, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size, padding=padding)
        self.conv_flow = nn.Conv2d(input_channels + hidden_channels, 2 * L, kernel_size, padding=padding)
        # Single 1x1 conv for concatenated warped states (paper style)
        self.conv_warp = nn.Conv2d(hidden_channels * L, hidden_channels, 1)

    def forward(self, x, h_prev):
        # Compute flow fields
        flow_input = torch.cat([x, h_prev], dim=1)
        flows = self.conv_flow(flow_input)  # (B, 2*L, H, W)
        flows = flows.chunk(self.L, dim=1)  # L x (B, 2, H, W)
        warped_h = []
        for l in range(self.L):
            flow = flows[l]
            h_warped = self._warp(h_prev, flow)
            warped_h.append(h_warped)
        # Concatenate along channel dimension (B, hidden*L, H, W)
        h_cat = torch.cat(warped_h, dim=1)
        h_agg = self.conv_warp(h_cat)  # (B, hidden, H, W)

        x_conv = self.conv_x(x)
        h_conv = self.conv_h(h_agg)
        i_x, f_x, o_x = x_conv.chunk(3, dim=1)
        i_h, f_h, o_h = h_conv.chunk(3, dim=1)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        o = torch.tanh(o_x + o_h)
        h_new = f * h_agg + i * o

        # Optional: Zoneout (paper style)
        if self.zoneout > 0.0 and self.training:
            mask = torch.empty_like(h_new).bernoulli_(1 - self.zoneout)
            h_new = mask * h_new + (1 - mask) * h_prev

        return h_new

    def _warp(self, x, flow):
        # x: (B, C, H, W), flow: (B, 2, H, W)
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=x.device),
            torch.arange(0, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        flow = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        new_grid = grid + flow
        new_grid_x = 2.0 * new_grid[..., 0] / max(W - 1, 1) - 1.0
        new_grid_y = 2.0 * new_grid[..., 1] / max(H - 1, 1) - 1.0
        new_grid = torch.stack((new_grid_x, new_grid_y), dim=-1)
        return F.grid_sample(x, new_grid, align_corners=True, padding_mode='border')

class TrajGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, L=5, seq_len_in=10, seq_len_out=1):
        super().__init__()
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.n_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        # Support per-layer kernel_size and L
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * self.n_layers
        if isinstance(L, int):
            L = [L] * self.n_layers
        assert len(kernel_size) == self.n_layers, f"kernel_size must have {self.n_layers} elements"
        assert len(L) == self.n_layers, f"L must have {self.n_layers} elements"
        self.kernel_size = kernel_size
        self.L = L
        self.cells = nn.ModuleList([
            TrajGRUCell(
                input_channels if i == 0 else hidden_channels[i-1],
                hidden_channels[i],
                kernel_size=kernel_size[i],
                L=L[i]
            ) for i in range(self.n_layers)
        ])
        self.out_conv = nn.Conv2d(hidden_channels[-1], input_channels, 1)

    def forward(self, x):
        # x: (B, C, D, H, W) where D = seq_len_in
        B, C, D, H, W = x.size()
        h = [torch.zeros(B, cell.hidden_channels, H, W, device=x.device, dtype=x.dtype) for cell in self.cells]
        outputs = []
        # Encode input sequence
        for t in range(self.seq_len_in):
            # For each layer, input is data for first cell, previous hidden for others
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h[i] = cell(x[:, :, t, :, :], h[i])
                else:
                    h[i] = cell(h[i-1], h[i])
        # Predict future sequence
        # Start with the last input frame as the first prediction input
        pred_inp = x[:, :, -1, :, :]
        for t in range(self.seq_len_out):
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h[i] = cell(pred_inp, h[i])
                else:
                    h[i] = cell(h[i-1], h[i])
            pred_inp = self.out_conv(h[-1])  # output of the last cell, to be used as input for next step
            outputs.append(pred_inp.unsqueeze(2))
        return torch.cat(outputs, dim=2)  # (B, C, seq_len_out, H, W)

# Training function
def train_radar_model(
    npy_path: str,
    save_dir: str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 4,
    lr: float = 2e-4,
    hidden_channels: list = None,
    kernel_size: list = None,
    L: list = None,
    epochs: int = 15,
    device: str = "cuda" ,
    loss_name: str = "mse",
    loss_weight_thresh: float = 30.0,
    loss_weight_high: float = 10.0,
    patch_size: int = 64,
    patch_stride: int = 64,
    patch_thresh: float = 0.4,
    patch_frac: float = 0.15,
    use_patches: bool = False,
    wandb_project: str = "radar-forecasting",
    early_stopping_patience: int = 10,
):
    """
    Train a TrajGRU radar forecasting model.

    Parameters
    ----------
    npy_path : str
        Path to the input NumPy file containing radar reflectivity data with shape (T, C, H, W).
    save_dir : str
        Directory to save model checkpoints and statistics.
    seq_len_in : int, optional
        Number of input time steps (default: 10).
    seq_len_out : int, optional
        Number of output time steps to predict (default: 1).
    train_val_test_split : tuple, optional
        Tuple/list of three floats (train, val, test) that sum to 1.0 (default: (0.7, 0.15, 0.15)).
    batch_size : int, optional
        Batch size for training (default: 4).
    lr : float, optional
        Learning rate for the optimizer (default: 2e-4).
    hidden_channels : list, optional
        List of hidden channels for each TrajGRU layer. The length determines the number of layers.
        Example: [64, 128, 128] for a 3-layer model with 64, 128, and 128 channels respectively.
        If None, defaults to [64] (single layer).
    kernel_size : list, optional
        List of kernel sizes for each TrajGRU layer. Must have same length as hidden_channels.
        Example: [3, 5, 3] for different kernel sizes per layer.
        If None, defaults to [3] * len(hidden_channels).
    L : list, optional
        List of L values (number of flow fields) for each TrajGRU layer. Must have same length as hidden_channels.
        Example: [5, 13, 9] for different L values per layer.
        If None, defaults to [5] * len(hidden_channels).
    epochs : int, optional
        Number of training epochs (default: 15).
    device : str, optional
        Device to run training on ('cuda' or 'cpu'); defaults to 'cuda' if available.
    loss_name : str, optional
        Loss function to use; options: 'mse', 'weighted_mse', 'b_mse'.
        'b_mse' uses the weighted MSE described in the paper, with bins: <2, 2-5, 5-10, 10-30, 30-45, >=45.
    loss_weight_thresh : float, optional
        Threshold for weighted MSE (default: 30.0 dBZ).
    loss_weight_high : float, optional
        Weight for high-reflectivity pixels (default: 10.0).
    patch_size : int, optional
        Size of spatial patches to extract (default: 64).
    patch_stride : int, optional
        Stride for patch extraction (default: 64).
    patch_thresh : float, optional
        Threshold for extracting patches (default: 0.4).
    patch_frac : float, optional
        Minimum fraction of pixels in patch above threshold (default: 0.15).
    use_patches : bool, optional
        Whether to use patch-based training (default: False).
    wandb_project : str, optional
        wandb project name (default: "radar-forecasting").
    early_stopping_patience : int, optional
        Number of epochs with no improvement before early stopping (default: 10).

    Returns
    -------
    None
    """
    # Set default values if None
    if hidden_channels is None:
        hidden_channels = [64]
    if kernel_size is None:
        kernel_size = [3] * len(hidden_channels)
    if L is None:
        L = [5] * len(hidden_channels)
    
    # Validate parameter lengths
    n_layers = len(hidden_channels)
    if len(kernel_size) != n_layers:
        raise ValueError(f"kernel_size must have {n_layers} elements, got {len(kernel_size)}")
    if len(L) != n_layers:
        raise ValueError(f"L must have {n_layers} elements, got {len(L)}")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load & sanitize (memory-mapped)
    cube = np.load(npy_path, mmap_mode='r')
    T,C,H,W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # chronological split & min-max
    if not (isinstance(train_val_test_split, (tuple, list)) and len(train_val_test_split) == 3):
        raise ValueError("train_val_test_split must be a tuple/list of three floats (train, val, test)")
    if not abs(sum(train_val_test_split) - 1.0) < 1e-6:
        raise ValueError(f"train_val_test_split must sum to 1.0, got {train_val_test_split} (sum={sum(train_val_test_split)})")
    train_frac, val_frac, _ = train_val_test_split
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    idx_train = list(range(0, n_train))
    idx_val = list(range(n_train, n_train + n_val))
    maxv = 85.0
    print(f"Normalization maxv (fixed): {maxv}")
    np.savez(save_dir/"minmax_stats.npz", maxv=maxv)
    eps = 1e-6

    # DataLoaders
    if use_patches:
        patch_index_path = str(save_dir / "patch_indices.npy")
        full_ds  = PatchRadarWindowDataset(cube, seq_len_in, seq_len_out, patch_size, patch_stride, patch_thresh, patch_frac, patch_index_path=patch_index_path, maxv=maxv)
        train_idx = []
        val_idx = []
        for i, (t, y, x) in enumerate(full_ds.patches):
            if t in idx_train:
                train_idx.append(i)
            elif t in idx_val:
                val_idx.append(i)
        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size, shuffle=False)
        print(f"Patch-based: train={len(train_ds)}  val={len(val_ds)}")
    else:
        full_ds  = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv=maxv)
        train_ds = Subset(full_ds, idx_train)
        val_ds   = Subset(full_ds, idx_val)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds, batch_size, shuffle=False)
        print(f"Samples  train={len(train_ds)}  val={len(val_ds)}")

    # model, optimizer, loss
    # Check input channels
    if C <= 0:
        raise ValueError(f"Invalid number of channels: {C}")
    model = TrajGRU(input_channels=C, hidden_channels=hidden_channels, kernel_size=kernel_size, L=L, seq_len_in=seq_len_in, seq_len_out=seq_len_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_name == "mse":
        criterion = lambda pred, tgt: mse_loss(pred, tgt, maxv=maxv, eps=eps)
    elif loss_name == "weighted_mse":
        criterion = lambda pred, tgt: weighted_mse_loss(
            pred, tgt,
            threshold=loss_weight_thresh,  # now in dBZ
            weight_high=loss_weight_high,
            maxv=maxv,
            eps=eps
        )
    elif loss_name == "b_mse":
        criterion = lambda pred, tgt: b_mse_loss(pred, tgt, maxv=maxv, eps=eps)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    # checkpoints
    ckpt_latest = save_dir/"latest.pt"
    ckpt_best   = save_dir/"best_val.pt"
    best_val    = float('inf')
    start_ep = 1
    epochs_since_improvement = 0
    if ckpt_latest.exists():
        st = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(st['model'])
        optimizer.load_state_dict(st['optim'])
        best_val = st['best_val']
        start_ep = st['epoch'] + 1
        print(f"✔ Resumed epoch {st['epoch']} (best_val={best_val:.4f})")

    end_epoch = start_ep + epochs - 1

    # wandb
    run_id = save_dir.name
    wandb.init(
        project=wandb_project,
        name=run_id,
        id=run_id,
        resume="allow",
        config={
            'seq_len_in': seq_len_in,
            'train_frac': train_frac,
            'batch_size': batch_size,
            'lr': lr,
            'hidden_channels': hidden_channels,
            'kernel_size': kernel_size,
            'L': L,
            'epochs': epochs,
            'device': device,
            'loss_name': loss_name,
            'loss_weight_thresh': loss_weight_thresh,
            'loss_weight_high': loss_weight_high,
            'patch_size': patch_size,
            'patch_stride': patch_stride,
            'patch_thresh': patch_thresh,
            'patch_frac': patch_frac,
            'use_patches': use_patches,
            'early_stopping_patience': early_stopping_patience
        }
    )
    wandb.watch(model)

    # training loop
    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        tot=0.0
        with torch.set_grad_enabled(train):
            for batch in tqdm(dl, desc=("Train" if train else "Val"), leave=False):
                if use_patches:
                    xb, yb = batch[0], batch[1]
                else:
                    xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
                # Ensure yb is (B, C, seq_len_out, H, W)
                if yb.ndim == 4:
                    yb = yb.unsqueeze(2)
                pred  = model(xb)
                # Squeeze singleton temporal dimension if present
                if pred.shape[2] == 1:
                    pred = pred.squeeze(2)
                if yb.shape[2] == 1:
                    yb = yb.squeeze(2)
                loss  = criterion(pred, yb)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                tot += loss.item()*xb.size(0)
        return tot/len(dl.dataset)

    for ep in range(start_ep, end_epoch+1):
        tr = run_epoch(train_dl, True)
        vl = run_epoch(val_dl,   False)
        print(f"[{ep:02d}/{end_epoch}] train {tr:.4f} | val {vl:.4f}")
        wandb.log({'epoch':ep,'train_loss':tr,'val_loss':vl})
        atomic_save({'epoch':ep,'model':model.state_dict(),
                    'optim':optimizer.state_dict(),'best_val':best_val},
                   ckpt_latest)
        if vl < best_val:
            best_val = vl
            atomic_save(model.state_dict(), ckpt_best)
            print("New best saved")
            wandb.log({'best_val_loss':best_val})
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        # Only apply early stopping if patience > 0
        if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping: validation loss did not improve for {epochs_since_improvement} epochs.")
            break

    print("Done. Checkpoints in", save_dir.resolve())
    wandb.finish()

def compute_mse_by_ranges(pred, target, ranges):
    """
    Compute MSE for different reflectivity ranges.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    target : np.ndarray
        Ground truth values.
    ranges : list of tuple
        List of (min, max) ranges to compute MSE for.

    Returns
    -------
    dict
        Dictionary with MSE values for each range.
    """
    mse_by_range = {}
    for r_min, r_max in ranges:
        mask = (target >= r_min) & (target < r_max)
        if np.any(mask):
            mse = np.mean((pred[mask] - target[mask]) ** 2)
            mse_by_range[f"mse_{r_min}_{r_max}"] = mse
        else:
            mse_by_range[f"mse_{r_min}_{r_max}"] = np.nan
    return mse_by_range


def predict_test_set(
    npy_path: str,
    run_dir:  str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 4,
    hidden_channels: list = None,
    kernel_size: list = None,
    L: list = None,
    which: str = "best",
    device: str = None,
    save_arrays: bool = True,
):
    """
    Run inference on the test set using a TrajGRU model.

    Parameters
    ----------
    npy_path : str
        Path to the input NumPy file containing radar reflectivity data with shape (T, C, H, W).
    run_dir : str
        Directory containing model checkpoints and statistics from training.
    seq_len_in : int, optional
        Number of input radar frames to use for prediction (default: 10).
    seq_len_out : int, optional
        Number of future radar frames to predict (default: 1).
    train_val_test_split : tuple, optional
        Tuple/list of three floats (train, val, test) that sum to 1.0 (default: (0.7, 0.15, 0.15)).
    batch_size : int, optional
        Batch size for inference (default: 4).
    hidden_channels : list, optional
        List of hidden channels for each TrajGRU layer. The length determines the number of layers.
        Example: [64, 128, 128] for a 3-layer model with 64, 128, and 128 channels respectively.
        If None, defaults to [64] (single layer).
    kernel_size : list, optional
        List of kernel sizes for each TrajGRU layer. Must have same length as hidden_channels.
        Example: [3, 5, 3] for different kernel sizes per layer.
        If None, defaults to [3] * len(hidden_channels).
    L : list, optional
        List of L values (number of flow fields) for each TrajGRU layer. Must have same length as hidden_channels.
        Example: [5, 13, 9] for different L values per layer.
        If None, defaults to [5] * len(hidden_channels).
    which : str, optional
        Which checkpoint to load - 'best' for best validation checkpoint or 'latest' (default: 'best').
    device : str, optional
        Device to run inference on (default: 'cpu').
    save_arrays : bool, optional
        Whether to save predictions and targets as memory-mapped .npy files in run_dir (default: True).
        Files will be named 'test_preds_dBZ.npy' and 'test_targets_dBZ.npy'.

    Returns
    -------
    None
        The function saves predictions and targets to disk if save_arrays=True, and prints MSE metrics
        for different reflectivity ranges (0-20, 20-35, 35-45, 45-100 dBZ).
    """
    # Set default values if None
    if hidden_channels is None:
        hidden_channels = [64]
    if kernel_size is None:
        kernel_size = [3] * len(hidden_channels)
    if L is None:
        L = [5] * len(hidden_channels)
    
    # # Validate parameter lengths

    
    device = device or "cpu"
    run_dir = Path(run_dir)
    ckpt    = run_dir / ("best_val.pt" if which=="best" else "latest.pt")
    stats   = np.load(run_dir/"minmax_stats.npz")
    maxv    = float(stats['maxv']); eps=1e-6

    # Use memory-mapped loading for large datasets
    cube = np.load(npy_path, mmap_mode='r')
    T, C, H, W = cube.shape
    if not (isinstance(train_val_test_split, (tuple, list)) and len(train_val_test_split) == 3):
        raise ValueError("train_val_test_split must be a tuple/list of three floats (train, val, test)")
    if not abs(sum(train_val_test_split) - 1.0) < 1e-6:
        raise ValueError(f"train_val_test_split must sum to 1.0, got {train_val_test_split} (sum={sum(train_val_test_split)})")
    train_frac, val_frac, test_frac = train_val_test_split
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    idx_test = list(range(n_train + n_val, n_total))
    ds      = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv=maxv)
    test_ds  = Subset(ds, idx_test)
    dl      = DataLoader(test_ds, batch_size, shuffle=False)

    model = TrajGRU(input_channels=C, hidden_channels=hidden_channels, kernel_size=kernel_size, L=L, seq_len_in=seq_len_in, seq_len_out=seq_len_out)
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and 'model' in st:
        st=st['model']
    model.load_state_dict(st)
    model.to(device).eval()

    N = len(test_ds)
    if save_arrays:
        preds_memmap = np.memmap(run_dir/"test_preds_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
        gts_memmap   = np.memmap(run_dir/"test_targets_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
    else:
        preds_memmap = None
        gts_memmap = None

    # For MSE by range, accumulate sum of squared errors and counts for each range
    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_sums = {f"mse_{r_min}_{r_max}": 0.0 for r_min, r_max in ranges}
    mse_counts = {f"mse_{r_min}_{r_max}": 0 for r_min, r_max in ranges}

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc='Validating', total=len(dl)):
            xb = xb.to(device)
            xb = xb.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
            if yb.ndim == 4:
                yb = yb.unsqueeze(2)
            out_n = model(xb)
            if out_n.shape[2] == 1:
                out_n = out_n.squeeze(2)
            if yb.shape[2] == 1:
                yb = yb.squeeze(2)
            out_n = out_n.cpu().numpy()
            yb = yb.cpu().numpy()
            out_n_dBZ = out_n * (maxv+eps)
            yb_dBZ = yb * (maxv+eps)
            batch_size = out_n.shape[0]
            if save_arrays:
                preds_memmap[idx:idx+batch_size] = out_n_dBZ
                gts_memmap[idx:idx+batch_size] = yb_dBZ
            # Compute MSE by range for this batch
            for r_min, r_max in ranges:
                mask = (yb_dBZ >= r_min) & (yb_dBZ < r_max)
                n_pix = np.sum(mask)
                if n_pix > 0:
                    mse = np.sum((out_n_dBZ[mask] - yb_dBZ[mask]) ** 2)
                    mse_sums[f"mse_{r_min}_{r_max}"] += mse
                    mse_counts[f"mse_{r_min}_{r_max}"] += n_pix
            idx += batch_size

    if save_arrays:
        preds_memmap.flush()
        gts_memmap.flush()
        # Save shape and dtype metadata for memmap arrays
        meta = {
            'shape': (N, C, H, W),
            'dtype': 'float32'
        }
        np.savez(run_dir/"test_preds_dBZ_meta.npz", **meta)
        np.savez(run_dir/"test_targets_dBZ_meta.npz", **meta)

    # Finalize MSE by range
    mse_by_range = {}
    for r_min, r_max in ranges:
        key = f"mse_{r_min}_{r_max}"
        if mse_counts[key] > 0:
            mse_by_range[key] = mse_sums[key] / mse_counts[key]
        else:
            mse_by_range[key] = np.nan
    # Save MSE metrics
    np.savez(run_dir/"mse_by_range.npz", **mse_by_range)
    print("MSE by reflectivity range:")
    for range_name, mse in mse_by_range.items():
        print(f"{range_name}: {mse:.4f}")
    if save_arrays:
        print("Saved test_preds_dBZ.npy + test_targets_dBZ.npy →", run_dir)
    print("Validation complete.")
    return None

def atomic_save(obj, path):
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def b_mse_loss(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute the B-MSE as described in the paper, using the weighting scheme:
    w(x) = 1 if x < 2
           2 if 2 <= x < 5
           5 if 5 <= x < 10
           10 if 10 <= x < 30
           30 if 30 <= x < 45
           45 if x >= 45
    pred, target: normalized (0-1), so convert to dBZ first.
    """
    # Convert to dBZ
    pred_dBZ = pred * (maxv + eps)
    target_dBZ = target * (maxv + eps)
    # Compute weights
    w = torch.ones_like(target_dBZ)
    w = torch.where(target_dBZ < 2, torch.tensor(1.0, device=target.device), w)
    w = torch.where((target_dBZ >= 2) & (target_dBZ < 5), torch.tensor(2.0, device=target.device), w)
    w = torch.where((target_dBZ >= 5) & (target_dBZ < 10), torch.tensor(5.0, device=target.device), w)
    w = torch.where((target_dBZ >= 10) & (target_dBZ < 30), torch.tensor(10.0, device=target.device), w)
    w = torch.where((target_dBZ >= 30) & (target_dBZ < 45), torch.tensor(30.0, device=target.device), w)
    w = torch.where(target_dBZ >= 45, torch.tensor(45.0, device=target.device), w)
    # B-MSE
    b_mse = (w * (pred_dBZ - target_dBZ) ** 2).sum() / w.sum()
    return b_mse

def mse_loss(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute MSE in dBZ units.
    pred, target: normalized (0-1), so convert to dBZ first.
    """
    pred_dBZ = pred * (maxv + eps)
    target_dBZ = target * (maxv + eps)
    return ((pred_dBZ - target_dBZ) ** 2).mean()

def weighted_mse_loss(pred, target, threshold=30.0, weight_high=10.0, maxv=85.0, eps=1e-6):
    """
    Weighted MSE loss in dBZ units, emphasizing high-reflectivity areas.
    pred, target: normalized (0-1), so convert to dBZ first.
    threshold: dBZ value above which to apply weight_high (e.g., 30.0)
    weight_high: weight for pixels above threshold
    """
    pred_dBZ = pred * (maxv + eps)
    target_dBZ = target * (maxv + eps)
    weight = torch.ones_like(target_dBZ)
    weight[target_dBZ > threshold] = weight_high
    return ((pred_dBZ - target_dBZ) ** 2 * weight).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or validate a TrajGRU radar forecasting model.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    train_parser.add_argument("--hidden_channels", type=str, default="64", help="Comma-separated list of hidden channels for TrajGRU layers. Length determines number of layers. Examples: '64' (1 layer), '64,128' (2 layers), '64,128,128' (3 layers)")
    train_parser.add_argument("--kernel_size", type=str, default="3", help="Comma-separated list of kernel sizes for each layer. Must have same length as hidden_channels. Examples: '3' (same for all), '5,3' (different per layer)")
    train_parser.add_argument("--L", type=str, default="5", help="Comma-separated list of L values (flow fields) for each layer. Must have same length as hidden_channels. Examples: '5' (same for all), '13,9' (different per layer)")
    train_parser.add_argument("--npy_path", type=str, default="Data/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    train_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    train_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    train_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 4)")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    train_parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda' or 'cpu')")
    train_parser.add_argument("--loss_name", type=str, default="mse", help="Loss function: mse, weighted_mse, or b_mse")
    train_parser.add_argument("--loss_weight_thresh", type=float, default=30.0, help="Threshold in normalized space to apply higher loss weighting or masking (default: 30.0 dBZ)")
    train_parser.add_argument("--loss_weight_high", type=float, default=10.0, help="Weight multiplier for pixels above threshold (default: 10.0)")
    train_parser.add_argument("--patch_size", type=int, default=64, help="Size of spatial patches to extract (default: 64)")
    train_parser.add_argument("--patch_stride", type=int, default=32, help="Stride for patch extraction (default: 32)")
    train_parser.add_argument("--patch_thresh", type=float, default=0.35, help="Threshold for extracting patches (default: 0.4)")
    train_parser.add_argument("--patch_frac", type=float, default=0.05, help="Minimum fraction of pixels in patch above threshold (default: 0.05)")
    train_parser.add_argument("--use_patches", type=str, default="True", help="Whether to use patch-based training: True or False (default: True)")
    train_parser.add_argument("--wandb_project", type=str, default="radar-forecasting", help="wandb project name")
    train_parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs with no improvement before early stopping (default: 10). Set to 0 or negative to disable early stopping.")

    # Subparser for validation
    val_parser = subparsers.add_parser("test", help="Run inference on the test set")
    val_parser.add_argument("--npy_path", type=str, required=True, help="Path to input .npy radar file")
    val_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    val_parser.add_argument("--hidden_channels", type=str, default="64", help="Comma-separated list of hidden channels for TrajGRU layers. Length determines number of layers. Examples: '64' (1 layer), '64,128' (2 layers), '64,128,128' (3 layers)")
    val_parser.add_argument("--kernel_size", type=str, default="3", help="Comma-separated list of kernel sizes for each layer. Must have same length as hidden_channels. Examples: '3' (same for all), '5,3' (different per layer)")
    val_parser.add_argument("--L", type=str, default="5", help="Comma-separated list of L values (flow fields) for each layer. Must have same length as hidden_channels. Examples: '5' (same for all), '13,9' (different per layer)")
    val_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    val_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    val_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    val_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    val_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    val_parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: 'cpu')")
    val_parser.add_argument("--save_arrays", type=bool, default=True, help="Whether to save predictions and targets as .npy files")

    # Subparser for testing
    test_parser = subparsers.add_parser("test", help="Run inference on the test set")
    test_parser.add_argument("--npy_path", type=str, required=True, help="Path to input .npy radar file")
    test_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    test_parser.add_argument("--hidden_channels", type=str, default="64", help="Comma-separated list of hidden channels for TrajGRU layers. Length determines number of layers. Examples: '64' (1 layer), '64,128' (2 layers), '64,128,128' (3 layers)")
    test_parser.add_argument("--kernel_size", type=str, default="3", help="Comma-separated list of kernel sizes for each layer. Must have same length as hidden_channels. Examples: '3' (same for all), '5,3' (different per layer)")
    test_parser.add_argument("--L", type=str, default="5", help="Comma-separated list of L values (flow fields) for each layer. Must have same length as hidden_channels. Examples: '5' (same for all), '13,9' (different per layer)")
    test_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    test_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    test_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    test_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    test_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    test_parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: 'cpu')")
    test_parser.add_argument("--save_arrays", type=bool, default=True, help="Whether to save predictions and targets as .npy files")

    args = parser.parse_args()

    # Parse comma-separated lists for hidden_channels, kernel_size, and L
    def parse_int_list(val):
        if "," in val:
            items = [int(x) for x in val.split(",")]
            return items
        else:
            return [int(val)]

    if args.command == "train":
        import ast
        train_val_test_split = ast.literal_eval(args.train_val_test_split)
        # Save arguments to save_dir/args.json
        import json, os
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        # Convert use_patches string to boolean
        if isinstance(args.use_patches, str):
            if args.use_patches.lower() in ["true", "1", "yes"]:
                args.use_patches = True
            elif args.use_patches.lower() in ["false", "0", "no"]:
                args.use_patches = False
            else:
                raise ValueError("--use_patches must be True or False")
        hidden_channels = parse_int_list(args.hidden_channels)
        kernel_size = parse_int_list(args.kernel_size)
        L = parse_int_list(args.L)
        
        # Expand single values to match hidden_channels length
        if len(kernel_size) == 1:
            kernel_size = kernel_size * len(hidden_channels)
        elif len(kernel_size) != len(hidden_channels):
            raise ValueError(f"kernel_size must have 1 or {len(hidden_channels)} values, got {len(kernel_size)}")
        
        if len(L) == 1:
            L = L * len(hidden_channels)
        elif len(L) != len(hidden_channels):
            raise ValueError(f"L must have 1 or {len(hidden_channels)} values, got {len(L)}")
        
        train_radar_model(
            npy_path=args.npy_path,
            save_dir=args.save_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            L=L,
            epochs=args.epochs,
            device=args.device,
            loss_name=args.loss_name,
            loss_weight_thresh=args.loss_weight_thresh,
            loss_weight_high=args.loss_weight_high,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_thresh=args.patch_thresh,
            patch_frac=args.patch_frac,
            use_patches=args.use_patches,
            wandb_project=args.wandb_project,
            early_stopping_patience=args.early_stopping_patience,
        )
    elif args.command == "test":
        import ast
        train_val_test_split = ast.literal_eval(args.train_val_test_split)
        # Save arguments to run_dir/args.json
        import json, os
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        hidden_channels = parse_int_list(args.hidden_channels)
        kernel_size = parse_int_list(args.kernel_size)
        L = parse_int_list(args.L)
        
        # Expand single values to match hidden_channels length
        if len(kernel_size) == 1:
            kernel_size = kernel_size * len(hidden_channels)
        elif len(kernel_size) != len(hidden_channels):
            raise ValueError(f"kernel_size must have 1 or {len(hidden_channels)} values, got {len(kernel_size)}")
        
        if len(L) == 1:
            L = L * len(hidden_channels)
        elif len(L) != len(hidden_channels):
            raise ValueError(f"L must have 1 or {len(hidden_channels)} values, got {len(L)}")
        
        predict_test_set(
            npy_path=args.npy_path,
            run_dir=args.run_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            L=L,
            which=args.which,
            device=args.device,
            save_arrays=args.save_arrays,
        )

