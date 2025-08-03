import sys
from pathlib import Path
# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import wandb
import os
from tqdm import tqdm
import ast
import json

from src.models.traj_gru_baseline import TrajGRUEncoderDecoder
# Import utilities
from src.training.utils import set_seed, atomic_save, mse_loss, weighted_mse_loss, b_mse_loss
# Import dataloaders
from src.training.utils import RadarWindowDataset, PatchRadarWindowDataset

set_seed(123)



def parse_int_list(val):
    if "," in val:
        return [int(x) for x in val.split(",")]
    else:
        return [int(val)]

# Training function 
def train_radar_model(
    npy_path: str,
    save_dir: str,
    args,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 4,
    lr: float = 2e-4,
    epochs: int = 15,
    device: str = "cuda",
    loss_name: str = "mse",
    loss_weight_thresh: float = 30.0,
    loss_weight_high: float = 10.0,
    patch_size: int = 64,
    patch_stride: int = 64,
    patch_thresh: float = 35.0,
    patch_frac: float = 0.01,
    use_patches: bool = False,
    wandb_project: str = "radar-forecasting",
    early_stopping_patience: int = 10,
    hidden_channels=None,
    kernel_size=None,
    L=None,
    conv_kernels=None,
    conv_strides=None,
):
    """
    Train a symmetric TrajGRU radar forecasting model.

    All architecture lists (hidden_channels, kernel_size, L, conv_kernels, conv_strides) must be the same length (number of layers).
    The encoder alternates Conv2d and TrajGRU layers; the decoder alternates TrajGRU and ConvTranspose2d layers, using the reversed lists for all parameters.

    Parameters
    ----------
    npy_path : str
        Path to the input NumPy file containing radar reflectivity data with shape (T, C, H, W).
    save_dir : str
        Directory to save model checkpoints and statistics.
    args : argparse.Namespace
        Parsed command-line arguments (for wandb and reproducibility).
    seq_len_in : int
        Number of input time steps.
    seq_len_out : int
        Number of output time steps to predict.
    train_val_test_split : tuple
        Tuple/list of three floats (train, val, test) that sum to 1.0.
    batch_size : int
        Batch size for training.
    lr : float
        Learning rate for the optimizer.
    epochs : int
        Number of training epochs.
    device : str
        Device to run training on ('cuda' or 'cpu').
    loss_name : str
        Loss function to use; options: 'mse', 'weighted_mse', 'b_mse'.
    loss_weight_thresh : float
        Threshold for weighted MSE (in dBZ).
    loss_weight_high : float
        Weight for high-reflectivity pixels.
    patch_size, patch_stride, patch_thresh, patch_frac : various (patch_frac default: 0.01)
        Patch extraction parameters (if use_patches=True).
    use_patches : bool
        Whether to use patch-based training.
    wandb_project : str
        wandb project name.
    early_stopping_patience : int
        Number of epochs with no improvement before early stopping.
    hidden_channels : list
        List of hidden channels for each layer (encoder/decoder, symmetric).
    kernel_size : list
        List of kernel sizes for each layer (encoder/decoder, symmetric).
    L : list
        List of L values (number of flow fields) for each layer (encoder/decoder, symmetric).
    conv_kernels : list
        List of kernel sizes for encoder Conv2d/decoder ConvTranspose2d (symmetric).
    conv_strides : list
        List of strides for encoder Conv2d/decoder ConvTranspose2d (symmetric).
    """
    # Set default values if None
    if hidden_channels is None:
        hidden_channels = [64]
    if kernel_size is None:
        kernel_size = [3] * len(hidden_channels)
    if L is None:
        L = [5] * len(hidden_channels)
    if conv_kernels is None:
        conv_kernels = [3] * len(hidden_channels)
    if conv_strides is None:
        conv_strides = [2] * len(hidden_channels)
    # Validate parameter lengths
    n_layers = len(hidden_channels)
    if not (len(kernel_size) == n_layers and len(L) == n_layers and len(conv_kernels) == n_layers and len(conv_strides) == n_layers):
        raise ValueError("All architecture lists (hidden_channels, kernel_size, L, conv_kernels, conv_strides) must have the same length.")
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
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
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
    
    # Model construction
    model = TrajGRUEncoderDecoder(
        input_channels=C,  # Use actual number of channels from data
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        L=L,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        seq_len_in=seq_len_in,
        seq_len_out=seq_len_out
    ).to(device)

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
    if not args.no_wandb:
        run_id = save_dir.name
        wandb.init(
            project=wandb_project,
            name=run_id,
            id=run_id,
            resume="allow",
            dir="experiments/wandb",
            config={
                'seq_len_in': seq_len_in,
                'seq_len_out': seq_len_out,
                'train_val_test_split': train_val_test_split,
                'batch_size': batch_size,
                'lr': lr,
                'hidden_channels': hidden_channels,
                'kernel_size': kernel_size,
                'L': L,
                'conv_kernels': conv_kernels,
                'conv_strides': conv_strides,
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
                'wandb_project': wandb_project,
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
        if not args.no_wandb:
            wandb.log({'epoch':ep,'train_loss':tr,'val_loss':vl})
        atomic_save({'epoch':ep,'model':model.state_dict(),
                    'optim':optimizer.state_dict(),'best_val':best_val},
                   ckpt_latest)
        if vl < best_val:
            best_val = vl
            atomic_save(model.state_dict(), ckpt_best)
            print("New best saved")
            if not args.no_wandb:
                wandb.log({'best_val_loss':best_val})
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        # Only apply early stopping if patience > 0
        if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping: validation loss did not improve for {epochs_since_improvement} epochs.")
            break

    print("Done. Checkpoints in", save_dir.resolve())
    if not args.no_wandb:
        wandb.finish()


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
    conv_kernels: list = None,
    conv_strides: list = None,
    which: str = "best",
    device: str = None,
    save_arrays: bool = True,
    predictions_dir: str = None,
):
    """
    Run inference on the test set using a symmetric TrajGRU model.

    All architecture lists (hidden_channels, kernel_size, L, conv_kernels, conv_strides) must be the same length (number of layers).
    The encoder alternates Conv2d and TrajGRU layers; the decoder alternates TrajGRU and ConvTranspose2d layers, using the reversed lists for all parameters.

    Parameters
    ----------
    npy_path : str
        Path to the input NumPy file containing radar reflectivity data with shape (T, C, H, W).
    run_dir : str
        Directory containing model checkpoints and statistics from training.
    seq_len_in : int
        Number of input radar frames to use for prediction.
    seq_len_out : int
        Number of future radar frames to predict.
    train_val_test_split : tuple
        Tuple/list of three floats (train, val, test) that sum to 1.0.
    batch_size : int
        Batch size for inference.
    hidden_channels : list
        List of hidden channels for each layer (encoder/decoder, symmetric).
    kernel_size : list
        List of kernel sizes for each layer (encoder/decoder, symmetric).
    L : list
        List of L values (number of flow fields) for each layer (encoder/decoder, symmetric).
    conv_kernels : list
        List of kernel sizes for encoder Conv2d/decoder ConvTranspose2d (symmetric).
    conv_strides : list
        List of strides for encoder Conv2d/decoder ConvTranspose2d (symmetric).
    which : str
        Which checkpoint to load - 'best' for best validation checkpoint or 'latest'.
    device : str
        Device to run inference on.
    save_arrays : bool
        Whether to save predictions and targets as memory-mapped .npy files.
    predictions_dir : str
        Directory to save large prediction/target files (default: same as run_dir).
    """
    # Set default values if None
    if hidden_channels is None:
        hidden_channels = [64]
    if kernel_size is None:
        kernel_size = [3] * len(hidden_channels)
    if L is None:
        L = [5] * len(hidden_channels)
    if conv_kernels is None:
        conv_kernels = [3] * len(hidden_channels)
    if conv_strides is None:
        conv_strides = [2] * len(hidden_channels)
    n_layers = len(hidden_channels)
    if not (len(kernel_size) == n_layers and len(L) == n_layers and len(conv_kernels) == n_layers and len(conv_strides) == n_layers):
        raise ValueError("All architecture lists (hidden_channels, kernel_size, L, conv_kernels, conv_strides) must have the same length.")
    device = device or "cpu"
    run_dir = Path(run_dir)
    ckpt    = run_dir / ("best_val.pt" if which=="best" else "latest.pt")
    stats   = np.load(run_dir/"minmax_stats.npz")
    maxv    = float(stats['maxv']); eps=1e-6
    
    # Determine where to save predictions
    if predictions_dir is None:
        predictions_dir = run_dir
    else:
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)

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

    # Model construction
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and 'model' in st:
        st=st['model']
    
    # Use the same input channels as the data (consistent with other scripts)
    model = TrajGRUEncoderDecoder(
        input_channels=C,  # Use current data channels
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        L=L,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        seq_len_in=seq_len_in,
        seq_len_out=seq_len_out
    ).to(device)
    
    model.load_state_dict(st)
    model.to(device).eval()

    N = len(test_ds)
    if save_arrays:
        preds_memmap = np.memmap(predictions_dir/"test_preds_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
        gts_memmap   = np.memmap(predictions_dir/"test_targets_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
    else:
        preds_memmap = None
        gts_memmap = None

    # For MSE by range, accumulate sum of squared errors and counts for each range
    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_sums = {f"mse_{r_min}_{r_max}": 0.0 for r_min, r_max in ranges}
    mse_counts = {f"mse_{r_min}_{r_max}": 0 for r_min, r_max in ranges}

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc='Testing', total=len(dl)):
            xb = xb.to(device)
            xb = xb.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
            
            # Handle target shape: yb should be (B, C, H, W) but might be (B, C, seq_len_out, H, W)
            if yb.ndim == 5:  # (B, C, seq_len_out, H, W)
                if yb.shape[2] == 1:  # seq_len_out == 1
                    yb = yb.squeeze(2)  # Remove sequence dimension
                else:
                    raise ValueError(f"Expected seq_len_out=1, got {yb.shape[2]}")
            
            out_n = model(xb)
            # Handle shape: model outputs (B, seq_len_out, C, H, W)
            # We want (B, C, H, W) for single timestep prediction
            if out_n.shape[1] == 1:  # seq_len_out == 1
                out_n = out_n.squeeze(1)  # Remove sequence dimension
            else:
                raise ValueError(f"Expected seq_len_out=1, got {out_n.shape[1]}")
                
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
        np.savez(predictions_dir/"test_preds_dBZ_meta.npz", **meta)
        np.savez(predictions_dir/"test_targets_dBZ_meta.npz", **meta)

    # Finalize MSE by range
    mse_by_range = {}
    for r_min, r_max in ranges:
        key = f"mse_{r_min}_{r_max}"
        if mse_counts[key] > 0:
            mse_by_range[key] = mse_sums[key] / mse_counts[key]
        else:
            mse_by_range[key] = np.nan
    # Create results directory in run_dir and save MSE metrics there
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save MSE metrics as JSON 
    import json
    with open(results_dir / "mse_by_ranges.json", "w") as f:
        json.dump(mse_by_range, f, indent=2)
    
    print("MSE by reflectivity range:")
    for range_name, mse in mse_by_range.items():
        print(f"{range_name}: {mse:.4f}")
    if save_arrays:
        print(f"Saved test_preds_dBZ.npy + test_targets_dBZ.npy → {predictions_dir}")
        print(f"Saved mse_by_ranges.json → {results_dir}")
    print("Validation complete.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a symmetric TrajGRU encoder-forecaster model (no U-Net, no skip connections).\n\nSpecify architecture using comma-separated lists for each argument. Example:\n\n--hidden_channels 64,192,192 --kernel_size 3,3,3 --L 13,13,9 --conv_kernels 5,5,3 --conv_strides 3,2,1\n\nThis will create a 3-layer encoder and 3-layer decoder, with decoder using reversed parameters.\n\n⚠️  IMPORTANT: Large strides (>3) can cause blank predictions due to excessive information loss. Use smaller strides for better results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    train_parser.add_argument("--npy_path", type=str, default="data/processed/ZH_radar_dataset.npy", help="Path to input .npy radar file")
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
    train_parser.add_argument("--patch_thresh", type=float, default=35.0, help="Threshold for extracting patches (default: 35.0 dBZ)")
    train_parser.add_argument("--patch_frac", type=float, default=0.01, help="Minimum fraction of pixels in patch above threshold (default: 0.01)")
    train_parser.add_argument("--use_patches", type=str, default="True", help="Whether to use patch-based training: True or False (default: True)")
    train_parser.add_argument("--wandb_project", type=str, default="radar-forecasting", help="wandb project name")
    train_parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    train_parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs with no improvement before early stopping (default: 10). Set to 0 or negative to disable early stopping.")
    train_parser.add_argument("--hidden_channels", type=str, required=True, help="Comma-separated list of hidden channels for each layer (encoder+decoder, symmetric)")
    train_parser.add_argument("--kernel_size", type=str, required=True, help="Comma-separated list of kernel sizes for each layer (encoder+decoder, symmetric)")
    train_parser.add_argument("--L", type=str, required=True, help="Comma-separated list of L values for each layer (encoder+decoder, symmetric)")
    train_parser.add_argument("--conv_kernels", type=str, required=True, help="Comma-separated list of kernel sizes for encoder Conv2d/decoder ConvTranspose2d (symmetric)")
    train_parser.add_argument("--conv_strides", type=str, required=True, help="Comma-separated list of strides for encoder Conv2d/decoder ConvTranspose2d (symmetric). WARNING: Be careful, large strides can cause blank predictions")

    # Test subparser
    test_parser = subparsers.add_parser("test", help="Test the model (inference)")
    test_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    test_parser.add_argument("--npy_path", type=str, default="data/processed/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    test_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    test_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    test_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    test_parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 4)")
    test_parser.add_argument("--hidden_channels", type=str, required=True, help="Comma-separated list of hidden channels for each layer (encoder+decoder, symmetric)")
    test_parser.add_argument("--kernel_size", type=str, required=True, help="Comma-separated list of kernel sizes for each layer (encoder+decoder, symmetric)")
    test_parser.add_argument("--L", type=str, required=True, help="Comma-separated list of L values for each layer (encoder+decoder, symmetric)")
    test_parser.add_argument("--conv_kernels", type=str, required=True, help="Comma-separated list of kernel sizes for encoder Conv2d/decoder ConvTranspose2d (symmetric)")
    test_parser.add_argument("--conv_strides", type=str, required=True, help="Comma-separated list of strides for encoder Conv2d/decoder ConvTranspose2d (symmetric)")
    test_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    test_parser.add_argument("--device", type=str, default='cpu', help="Device to run inference on (default: 'cpu')")
    test_parser.add_argument("--save_arrays", type=lambda x: (str(x).lower() in ['true','1','yes']), default=True, help="Whether to save predictions and targets as .npy files (default: True)")
    test_parser.add_argument("--predictions_dir", type=str, default=None, help="Directory to save prediction arrays (default: run_dir)")

    args = parser.parse_args()

    def expand_to_length(lst, n):
        if len(lst) == 1:
            return lst * n
        elif len(lst) == n:
            return lst
        else:
            raise ValueError(f"List must have 1 or {n} values, got {len(lst)}")

    if args.command == "train":
        import json, os
        train_val_test_split = ast.literal_eval(args.train_val_test_split)
        # Create save directory and save arguments
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "train_args.json"), "w") as f:
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
        conv_kernels = parse_int_list(args.conv_kernels)
        conv_strides = parse_int_list(args.conv_strides)
        n_layers = len(hidden_channels)
        kernel_size = expand_to_length(kernel_size, n_layers)
        L = expand_to_length(L, n_layers)
        conv_kernels = expand_to_length(conv_kernels, n_layers)
        conv_strides = expand_to_length(conv_strides, n_layers)
        train_radar_model(
            npy_path=args.npy_path,
            save_dir=args.save_dir,
            args=args,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            lr=args.lr,
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
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            L=L,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
        )
    elif args.command == "test":
        import json, os
        train_val_test_split = ast.literal_eval(args.train_val_test_split)
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, "test_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        hidden_channels = parse_int_list(args.hidden_channels)
        kernel_size = parse_int_list(args.kernel_size)
        L = parse_int_list(args.L)
        conv_kernels = parse_int_list(args.conv_kernels)
        conv_strides = parse_int_list(args.conv_strides)
        n_layers = len(hidden_channels)
        kernel_size = expand_to_length(kernel_size, n_layers)
        L = expand_to_length(L, n_layers)
        conv_kernels = expand_to_length(conv_kernels, n_layers)
        conv_strides = expand_to_length(conv_strides, n_layers)
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
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            which=args.which,
            device=args.device,
            save_arrays=args.save_arrays,
            predictions_dir=args.predictions_dir,
        )