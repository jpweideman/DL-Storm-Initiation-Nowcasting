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

# Flexible Vision Transformer with Conv Tokenizer and Relative Positional Encoding
class ConvTokenizer(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x, (H, W)

class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads, max_rel_dist=32):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel_dist = max_rel_dist
        self.rel_bias = nn.Parameter(torch.zeros((2*max_rel_dist-1, 2*max_rel_dist-1, num_heads)))
    def forward(self, H, W):
        device = self.rel_bias.device
        needed_rel_dist = max(H, W)
        if needed_rel_dist > self.max_rel_dist:
            # Dynamically expand rel_bias if needed
            new_size = 2*needed_rel_dist-1
            new_rel_bias = torch.zeros((new_size, new_size, self.num_heads), device=device, dtype=self.rel_bias.dtype)
            old_size = self.rel_bias.shape[0]
            # Copy old values into the center of the new tensor
            start = (new_size - old_size) // 2
            new_rel_bias[start:start+old_size, start:start+old_size, :] = self.rel_bias.data
            self.rel_bias = nn.Parameter(new_rel_bias)
            self.max_rel_dist = needed_rel_dist
        coords = torch.stack(torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'), dim=-1)  # (H, W, 2)
        coords_flat = coords.reshape(-1, 2)
        rel_coords = coords_flat[:, None, :] - coords_flat[None, :, :]  # (HW, HW, 2)
        rel_coords += self.max_rel_dist - 1
        bias = self.rel_bias[rel_coords[..., 0], rel_coords[..., 1]]  # (HW, HW, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, HW, HW)

class FlexibleViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.rel_pos = RelativePositionBias2D(num_heads)
    def forward(self, x, H, W):
        # x: (B, HW, C)
        B, N, C = x.shape
        attn_bias = self.rel_pos(H, W)  # (num_heads, HW, HW)
        # MultiheadAttention does not support attn_bias directly, so we use attn_mask workaround
        # We'll use attn_mask=None for now, as attn_bias is not natively supported in PyTorch's MultiheadAttention
        # For a more advanced implementation, use xformers or timm
        x2, _ = self.attn(x, x, x, need_weights=False)
        x = x + x2
        x = x + self.mlp(self.norm2(x))
        return x

class FlexibleRadarTransformer(nn.Module):
    def __init__(self, input_channels, seq_len_in, seq_len_out, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1, conv_kernel=7, conv_stride=4, conv_padding=3):
        super().__init__()
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.input_channels = input_channels
        self.d_model = d_model
        self.tokenizer = ConvTokenizer(input_channels * seq_len_in, d_model, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding)
        self.blocks = nn.ModuleList([
            FlexibleViTBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, input_channels * seq_len_out)
        self.input_channels = input_channels
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
    def forward(self, x):
        # x: (B, seq_len_in, C, H, W)
        B, S, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C * S, H, W)  # (B, C*S, H, W)
        x, (Hf, Wf) = self.tokenizer(x)  # (B, Hf*Wf, d_model)
        for i, blk in enumerate(self.blocks):
            x = blk(x, Hf, Wf)
        x = self.head(x)  # (B, Hf*Wf, C*seq_len_out)
        x = x.view(B, Hf, Wf, C, self.seq_len_out).permute(0, 3, 4, 1, 2)  # (B, C, seq_len_out, Hf, Wf)
        # Upsample to match input H, W if needed
        if (Hf, Wf) != (H, W):
            B, C, T, _, _ = x.shape
            x = x.reshape(B * C * T, 1, Hf, Wf)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x.reshape(B, C, T, H, W)
        return x

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
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
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
    conv_kernel: int = 7,
    conv_stride: int = 4,
    conv_padding: int = 3,
):
    """
    Train a Transformer-based radar forecasting model.

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
    d_model : int, optional
        Dimension of the model's input and output (default: 128).
    nhead : int, optional
        Number of attention heads in the Transformer encoder (default: 4).
    num_layers : int, optional
        Number of layers in the Transformer encoder (default: 4).
    dim_feedforward : int, optional
        Dimension of the feedforward network in the Transformer encoder (default: 256).
    dropout : float, optional
        Dropout probability in the Transformer encoder (default: 0.1).
    epochs : int, optional
        Number of training epochs (default: 15).
    device : str, optional
        Device to run training on ('cuda' or 'cpu'); defaults to 'cuda' if available.
    loss_name : str, optional
        Loss function to use; either 'mse', 'weighted_mse'.
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
    conv_kernel : int, optional
        Convolutional stem kernel size (default: 7).
    conv_stride : int, optional
        Convolutional stem stride (default: 4).
    conv_padding : int, optional
        Convolutional stem padding (default: 3).

    Returns
    -------
    None
    """
    if not (isinstance(train_val_test_split, (tuple, list)) and len(train_val_test_split) == 3):
        raise ValueError("train_val_test_split must be a tuple/list of three floats (train, val, test)")
    if not abs(sum(train_val_test_split) - 1.0) < 1e-6:
        raise ValueError(f"train_val_test_split must sum to 1.0, got {train_val_test_split} (sum={sum(train_val_test_split)})")
    train_frac, val_frac, _ = train_val_test_split
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load & sanitize (memory-mapped)
    cube = np.load(npy_path, mmap_mode='r')
    T,C,H,W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # chronological split & min-max
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    idx_train = list(range(0, n_train))
    idx_val = list(range(n_train, n_train + n_val))
    idx_test = list(range(n_train + n_val, n_total))

    maxv = 85.0
    print(f"Normalization maxv (fixed): {maxv}")
    np.savez(save_dir/"minmax_stats.npz", maxv=maxv)
    eps = 1e-6

    # DataLoaders
    if use_patches:
        patch_index_path = str(save_dir / "patch_indices.npy")
        full_ds  = PatchRadarWindowDataset(cube, seq_len_in, seq_len_out, patch_size, patch_stride, patch_thresh, patch_frac, patch_index_path=patch_index_path, maxv=maxv)
        train_ds = Subset(full_ds, idx_train)
        val_ds   = Subset(full_ds, idx_val)
        test_ds  = Subset(full_ds, idx_test)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size, shuffle=False)
        print(f"Patch-based: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    else:
        full_ds  = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv=maxv)
        train_ds = Subset(full_ds, idx_train)
        val_ds   = Subset(full_ds, idx_val)
        test_ds  = Subset(full_ds, idx_test)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds, batch_size, shuffle=False)
        print(f"Samples  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # model, optimizer, loss
    model     = FlexibleRadarTransformer(input_channels=C, seq_len_in=seq_len_in, seq_len_out=seq_len_out, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, conv_kernel=conv_kernel, conv_stride=conv_stride, conv_padding=conv_padding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_name == "mse":
        criterion = lambda pred, tgt: mse_loss(pred, tgt, maxv=maxv, eps=eps)
    elif loss_name == "weighted_mse":
        criterion = lambda pred, tgt: weighted_mse_loss(
            pred, tgt,
            threshold=loss_weight_thresh,  # dBZ
            weight_high=loss_weight_high,
            maxv=maxv,
            eps=eps
        )
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
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
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
            'early_stopping_patience': early_stopping_patience,
            'conv_kernel': conv_kernel,
            'conv_stride': conv_stride,
            'conv_padding': conv_padding,
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
                # Do NOT permute xb; model expects (B, seq_len_in, C, H, W)
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
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    which: str = "best",
    device: str = None,
    save_arrays: bool = True,
    conv_kernel: int = 7,
    conv_stride: int = 4,
    conv_padding: int = 3,
):
    """
    Run inference on the test set using a Transformer-based radar forecasting model from train_radar_model.

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
    d_model : int, optional
        Dimension of the model's input and output (default: 128).
    nhead : int, optional
        Number of attention heads in the Transformer encoder (default: 4).
    num_layers : int, optional
        Number of layers in the Transformer encoder (default: 4).
    dim_feedforward : int, optional
        Feedforward network dimension (default: 256).
    dropout : float, optional
        Dropout probability in the Transformer encoder (default: 0.1).
    which : str, optional
        Which checkpoint to load - 'best' for best validation checkpoint or 'latest' (default: 'best').
    device : str, optional
        Device to run inference on (default: 'cpu').
    save_arrays : bool, optional
        Whether to save predictions and targets as memory-mapped .npy files in run_dir (default: True).
        Files will be named 'test_preds_dBZ.npy' and 'test_targets_dBZ.npy'.
    conv_kernel : int, optional
        Convolutional stem kernel size (default: 7).
    conv_stride : int, optional
        Convolutional stem stride (default: 4).
    conv_padding : int, optional
        Convolutional stem padding (default: 3).

    Returns
    -------
    None
        The function saves predictions and targets to disk if save_arrays=True, and prints MSE metrics
        for different reflectivity ranges (0-20, 20-35, 35-45, 45-100 dBZ).
    """
    import numpy as np
    from tqdm import tqdm

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

    model = FlexibleRadarTransformer(input_channels=C, seq_len_in=seq_len_in, seq_len_out=seq_len_out, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, conv_kernel=conv_kernel, conv_stride=conv_stride, conv_padding=conv_padding)
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and 'model' in st:
        st=st['model']
    model.load_state_dict(st)
    model.to(device).eval()

    N = len(test_ds)
    # We'll determine C, H, W from the first batch
    preds_memmap = None
    gts_memmap = None
    sample_count = 0
    idx = 0

    # --- MSE by reflectivity range setup (as in other scripts) ---
    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_sums = {f"mse_{r_min}_{r_max}": 0.0 for r_min, r_max in ranges}
    mse_counts = {f"mse_{r_min}_{r_max}": 0 for r_min, r_max in ranges}
    # -----------------------------------------------------------

    with torch.no_grad():
        for batch in tqdm(dl, desc='Testing', total=len(dl)):
            # Support both patch and non-patch datasets
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
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
            if out_n_dBZ.ndim == 3:
                out_n_dBZ = out_n_dBZ[:, None, :, :]
            if yb_dBZ.ndim == 3:
                yb_dBZ = yb_dBZ[:, None, :, :]
            batch_size, C, H, W = out_n_dBZ.shape
            slots_left = N - idx
            write_count = min(batch_size, slots_left)
            # Initialize memmaps on first batch
            if preds_memmap is None and save_arrays:
                preds_memmap = np.memmap(run_dir/"test_preds_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
                gts_memmap   = np.memmap(run_dir/"test_targets_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
                assert preds_memmap.shape[0] == N, f"preds_memmap.shape[0] ({preds_memmap.shape[0]}) != N ({N})"
                assert gts_memmap.shape[0] == N, f"gts_memmap.shape[0] ({gts_memmap.shape[0]}) != N ({N})"
            # Defensive shape check and safe writing
            if save_arrays and write_count > 0:
                expected_shape = (write_count, C, H, W)
                if out_n_dBZ[:write_count].shape != expected_shape:
                    raise ValueError(f"Shape mismatch: memmap expects {expected_shape}, got {out_n_dBZ[:write_count].shape}")
                if yb_dBZ[:write_count].shape != expected_shape:
                    raise ValueError(f"Shape mismatch: memmap expects {expected_shape}, got {yb_dBZ[:write_count].shape}")
                end_idx = min(idx + write_count, N)
                actual_write = end_idx - idx
                preds_memmap[idx:end_idx] = out_n_dBZ[:actual_write]
                gts_memmap[idx:end_idx] = yb_dBZ[:actual_write]
                idx = end_idx
                sample_count += actual_write
                if idx >= N:
                    break
            # --- MSE by range accumulation (as in other scripts) ---
            for r_min, r_max in ranges:
                mask = (yb_dBZ[:write_count] >= r_min) & (yb_dBZ[:write_count] < r_max)
                n_pix = np.sum(mask)
                if n_pix > 0:
                    mse = np.sum((out_n_dBZ[:write_count][mask] - yb_dBZ[:write_count][mask]) ** 2)
                    mse_sums[f"mse_{r_min}_{r_max}"] += mse
                    mse_counts[f"mse_{r_min}_{r_max}"] += n_pix
            # -----------------------------------------------------
    print(f"Total samples written: {sample_count}, N: {N}")
    if save_arrays and preds_memmap is not None:
        preds_memmap.flush()
        gts_memmap.flush()
        # Save shape and dtype metadata for memmap arrays
        meta = {
            'shape': preds_memmap.shape,
            'dtype': str(preds_memmap.dtype)
        }
        np.savez(run_dir/"test_preds_dBZ_meta.npz", **meta)
        np.savez(run_dir/"test_targets_dBZ_meta.npz", **meta)

    # Finalize MSE by range (as in other scripts)
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
    return None

def atomic_save(obj, path):
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or validate a Transformer-based radar forecasting model.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    train_parser.add_argument("--npy_path", type=str, default="Data/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    train_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    train_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    train_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 4)")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    train_parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda' or 'cpu')")
    train_parser.add_argument("--loss_name", type=str, default="mse", help="Loss function: mse or weighted_mse (all in dBZ units)")
    train_parser.add_argument("--loss_weight_thresh", type=float, default=30.0, help="Threshold in dBZ to apply higher loss weighting (default: 30.0 dBZ)")
    train_parser.add_argument("--loss_weight_high", type=float, default=10.0, help="Weight multiplier for pixels above threshold (default: 10.0)")
    train_parser.add_argument("--patch_size", type=int, default=64, help="Size of spatial patches to extract (default: 64)")
    train_parser.add_argument("--patch_stride", type=int, default=32, help="Stride for patch extraction (default: 32)")
    train_parser.add_argument("--patch_thresh", type=float, default=0.35, help="Threshold for extracting patches (default: 0.4)")
    train_parser.add_argument("--patch_frac", type=float, default=0.05, help="Minimum fraction of pixels in patch above threshold (default: 0.05)")
    train_parser.add_argument("--use_patches", type=str, default="False", help="Whether to use patch-based training: True or False (default: False)")
    train_parser.add_argument("--wandb_project", type=str, default="radar-forecasting", help="wandb project name")
    train_parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs with no improvement before early stopping (default: 10). Set to 0 or negative to disable early stopping.")
    train_parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension (default: 128)")
    train_parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    train_parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer encoder layers (default: 4)")
    train_parser.add_argument("--dim_feedforward", type=int, default=256, help="Feedforward network dimension (default: 256)")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    train_parser.add_argument("--conv_kernel", type=int, default=7, help="Convolutional stem kernel size (default: 7)")
    train_parser.add_argument("--conv_stride", type=int, default=4, help="Convolutional stem stride (default: 4)")
    train_parser.add_argument("--conv_padding", type=int, default=3, help="Convolutional stem padding (default: 3)")

    # Subparser for test
    test_parser = subparsers.add_parser("test", help="Run test and compute MSE by reflectivity range")
    test_parser.add_argument("--npy_path", type=str, required=True, help="Path to input .npy radar file")
    test_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    test_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    test_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    test_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    test_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    test_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    test_parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: 'cpu')")
    test_parser.add_argument("--save_arrays", type=bool, default=True, help="Whether to save predictions and targets as .npy files")
    test_parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension (default: 128)")
    test_parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    test_parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer encoder layers (default: 4)")
    test_parser.add_argument("--dim_feedforward", type=int, default=256, help="Feedforward network dimension (default: 256)")
    test_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    test_parser.add_argument("--conv_kernel", type=int, default=7, help="Convolutional stem kernel size (default: 7)")
    test_parser.add_argument("--conv_stride", type=int, default=4, help="Convolutional stem stride (default: 4)")
    test_parser.add_argument("--conv_padding", type=int, default=3, help="Convolutional stem padding (default: 3)")

    args = parser.parse_args()

    if args.command == "train":
        # Convert use_patches string to boolean
        if isinstance(args.use_patches, str):
            if args.use_patches.lower() in ["true", "1", "yes"]:
                args.use_patches = True
            elif args.use_patches.lower() in ["false", "0", "no"]:
                args.use_patches = False
            else:
                raise ValueError("--use_patches must be True or False")
        import ast
        train_val_test_split = ast.literal_eval(args.train_val_test_split)
        try:
            train_radar_model(
                npy_path=args.npy_path,
                save_dir=args.save_dir,
                seq_len_in=args.seq_len_in,
                seq_len_out=args.seq_len_out,
                train_val_test_split=train_val_test_split,
                batch_size=args.batch_size,
                lr=args.lr,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
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
                conv_kernel=args.conv_kernel,
                conv_stride=args.conv_stride,
                conv_padding=args.conv_padding,
            )
        except Exception as e:
            print(f"Error during training: {e}")
    elif args.command == "test":
        try:
            predict_test_set(
                npy_path=args.npy_path,
                run_dir=args.run_dir,
                seq_len_in=args.seq_len_in,
                seq_len_out=args.seq_len_out,
                train_val_test_split=args.train_val_test_split,
                batch_size=args.batch_size,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                which=args.which,
                device=args.device,
                save_arrays=args.save_arrays,
                conv_kernel=args.conv_kernel,
                conv_stride=args.conv_stride,
                conv_padding=args.conv_padding,
            )
        except Exception as e:
            print(f"Error during validation: {e}")

