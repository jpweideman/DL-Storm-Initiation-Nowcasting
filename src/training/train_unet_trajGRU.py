import sys
from pathlib import Path
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

from src.models.unet_traj_gru import UNetTrajGRU
from src.training.utils import set_seed, atomic_save, mse_loss, weighted_mse_loss, b_mse_loss
from src.training.utils import RadarWindowDataset, PatchRadarWindowDataset
from src.training.utils.training_utils import (
    init_forecasting_metrics_accumulator,
    accumulate_forecasting_metrics_batch,
    compute_final_forecasting_metrics,
)

set_seed(123)

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
    base_ch: int = 32,
    bottleneck_dims: list = None,
    kernel: int = 3,
    L: int = 5,
    epochs: int = 15,
    device: str = "cuda" ,
    loss_name: str = "mse",
    loss_weight_thresh: float = 30.0,
    loss_weight_high: float = 10.0,
    patch_size: int = 64,
    patch_stride: int = 32,
    patch_thresh: float = 35.0,
    patch_frac: float = 0.01,
    use_patches: bool = False,
    wandb_project: str = "radar-forecasting",
    early_stopping_patience: int = 10,
):
    """
    Train a UNet TrajGRU radar forecasting model.

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
    base_ch : int, optional
        Base number of channels for U-Net encoder/decoder (default: 32).
    bottleneck_dims : list, optional
        Sequence of widths for the bottleneck TrajGRU layers (e.g., (64, 32)).
        Each entry corresponds to a DoubleTrajGRUBlock (i.e., two TrajGRU cells per entry).
        The number of entries determines the depth of the bottleneck.
        If None, defaults to (base_ch*4,) for a single bottleneck stage.
    kernel : int, optional
        Kernel size for all convolutions in the U-Net (default: 3).
    L : int, optional
        Number of flow fields for all TrajGRU layers across encoder, bottleneck, and decoder (default: 5).
    epochs : int, optional
        Number of training epochs (default: 15).
    device : str, optional
        Device to run training on ('cuda' or 'cpu'); defaults to 'cuda' if available.
    loss_name : str, optional
        Loss function to use; options: 'mse', 'weighted_mse', 'b_mse'.
        'b_mse' uses the weighted MSE described in the paper, with bins: <2, 2-5, 5-10, 10-30, 30-45, >=45.
    loss_weight_thresh : float, optional
        Reflectivity threshold in dBZ for weighted_mse loss (e.g., 30.0). Only used when loss_name='weighted_mse'.
    loss_weight_high : float, optional
        Weight multiplier for pixels above threshold in weighted_mse loss (e.g., 10.0). Only used when loss_name='weighted_mse'.
    patch_size : int, optional
        Size of spatial patches to extract (default: 64).
    patch_stride : int, optional
        Stride for patch extraction (default: 32).
    patch_thresh : float, optional
        Threshold for extracting patches (default: 35.0 dBZ).
    patch_frac : float, optional
        Minimum fraction of pixels in patch above threshold (default: 0.01).
    use_patches : bool, optional
        Whether to use patch-based training (default: False).
    wandb_project : str, optional
        wandb project name (default: "radar-forecasting").
    early_stopping_patience : int, optional
        Number of epochs with no improvement before early stopping (default: 10).
    """
    if bottleneck_dims is None:
        bottleneck_dims = [base_ch*4]
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # memory-mapped loading
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
        patch_ds = PatchRadarWindowDataset(cube, seq_len_in, seq_len_out, patch_size, patch_stride, patch_thresh, patch_frac, patch_index_path=patch_index_path, maxv=maxv)
        train_idx = []
        for i, (t, y, x) in enumerate(patch_ds.patches):
            if t in idx_train:
                train_idx.append(i)
        train_ds = Subset(patch_ds, train_idx)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
        
        # Validation always use full frames 
        full_ds = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv=maxv)
        val_ds = Subset(full_ds, idx_val)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False)
        print(f"Patch-based training: train_patches={len(train_ds)}, val_fullframes={len(val_ds)}")
    else:
        full_ds  = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv=maxv)
        train_ds = Subset(full_ds, idx_train)
        val_ds   = Subset(full_ds, idx_val)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds, batch_size, shuffle=False)
        print(f"Full-frame training: train={len(train_ds)}, val={len(val_ds)}")

    # model, optimizer, loss
    # Check input channels
    if C <= 0:
        raise ValueError(f"Invalid number of channels: {C}")
    model = UNetTrajGRU(in_ch=C, out_ch=C, base_ch=base_ch, bottleneck_dims=bottleneck_dims, seq_len=seq_len_in, kernel=kernel, L=L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_name == "mse":
        criterion = lambda pred, tgt: mse_loss(pred, tgt, maxv=maxv, eps=eps)
    elif loss_name == "weighted_mse":
        criterion = lambda pred, tgt: weighted_mse_loss(
            pred, tgt,
            threshold=loss_weight_thresh, 
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
            dir="experiments",
            config={
                'seq_len_in': seq_len_in,
                'seq_len_out': seq_len_out,
                'train_val_test_split': train_val_test_split,
                'batch_size': batch_size,
                'lr': lr,
                'base_ch': base_ch,
                'bottleneck_dims': bottleneck_dims,
                'kernel': kernel,
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
                'wandb_project': wandb_project,
                'early_stopping_patience': early_stopping_patience
            }
        )
        wandb.watch(model)

    # training loop
    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        tot=0.0
        
        if not train:
            # Initialize running statistics for validation (global metrics accumulator + MSE by ranges)
            import numpy as np
            ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
            mse_by_range = {f"mse_{r_min}_{r_max}": {"sum": 0.0, "count": 0} for r_min, r_max in ranges}
            metrics_accumulator = init_forecasting_metrics_accumulator()
        
        with torch.set_grad_enabled(train):
            for batch in tqdm(dl, desc=("Train" if train else "Val"), leave=False):
                if use_patches and train:  # Only training uses patches 
                    xb, yb = batch[0], batch[1]
                else:
                    xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                if yb.ndim == 4:
                    yb = yb.unsqueeze(2)
                pred  = model(xb)
                if pred.shape[2] == 1:
                    pred = pred.squeeze(2)
                if yb.shape[2] == 1:
                    yb = yb.squeeze(2)
                loss  = criterion(pred, yb)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                tot += loss.item()*xb.size(0)
                
                if not train:
                    metrics_accumulator = accumulate_forecasting_metrics_batch(
                        metrics_accumulator, pred.detach(), yb.detach(), maxv=maxv, eps=eps
                    )
                    maxv_loc = maxv + eps
                    pred_dBZ = pred.detach() * maxv_loc
                    target_dBZ = yb.detach() * maxv_loc
                    if pred_dBZ.shape != target_dBZ.shape:
                        pred_dBZ = pred_dBZ.squeeze()
                        target_dBZ = target_dBZ.squeeze()
                    for r_min, r_max in ranges:
                        mask = (target_dBZ >= r_min) & (target_dBZ < r_max)
                        n_pix = torch.sum(mask).item()
                        if n_pix > 0:
                            mse_bin = torch.sum(((pred_dBZ[mask] - target_dBZ[mask]) ** 2)).item()
                            mse_by_range[f"mse_{r_min}_{r_max}"]["sum"] += mse_bin
                            mse_by_range[f"mse_{r_min}_{r_max}"]["count"] += n_pix
        
        if not train:
            import numpy as np
            final_mse_by_range = {}
            for range_name, stats in mse_by_range.items():
                if stats["count"] > 0:
                    final_mse_by_range[range_name] = stats["sum"] / stats["count"]
                else:
                    final_mse_by_range[range_name] = np.nan
            final_storm_metrics = compute_final_forecasting_metrics(metrics_accumulator)
            
            print("Validation metrics:")
            print(f"  B-MSE: {final_storm_metrics['b_mse']:.4f}")
            for th, csi in final_storm_metrics['csi_by_threshold'].items():
                print(f"  CSI {th}: {csi:.4f}")
            for th, hss in final_storm_metrics['hss_by_threshold'].items():
                print(f"  HSS {th}: {hss:.4f}")
            print(f"  MSE: {final_storm_metrics['mse']:.4f}")
            for range_name, mse_val in final_mse_by_range.items():
                print(f"  {range_name}: {mse_val:.4f}")
            
            if not args.no_wandb:
                wandb.log({**{f"val_{k}": v for k, v in final_storm_metrics['csi_by_threshold'].items()},
                           **{f"val_{k}": v for k, v in final_storm_metrics['hss_by_threshold'].items()},
                           "val_b_mse": final_storm_metrics['b_mse'],
                           "val_mse": final_storm_metrics['mse'],
                           **{f"val_{k}": v for k, v in final_mse_by_range.items()}})
            
            run_epoch.validation_metrics = {
                "b_mse": float(final_storm_metrics['b_mse']),
                "mse": float(final_storm_metrics['mse']),
                "csi_by_threshold": final_storm_metrics['csi_by_threshold'],
                "hss_by_threshold": final_storm_metrics['hss_by_threshold'],
                "confusion_by_threshold": final_storm_metrics.get('confusion_by_threshold', {}),
                "mse_by_range": final_mse_by_range
            }
        
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
            

            if hasattr(run_epoch, 'validation_metrics'):
                import json
                results_dir = save_dir / "results"
                results_dir.mkdir(exist_ok=True)
                metrics_to_save = {
                    "epoch": ep,
                    "val_loss": vl,
                    **run_epoch.validation_metrics
                }
                with open(results_dir / "best_validation_metrics.json", "w") as f:
                    json.dump(metrics_to_save, f, indent=2)
                print(f"Validation metrics saved to {results_dir}/best_validation_metrics.json")
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
    base_ch: int = 32,
    bottleneck_dims: list = None,
    kernel: int = 3,
    L: int = 5,
    which: str = "best",
    device: str = None,
    save_arrays: bool = True,
    predictions_dir: str = None,
):
    """
    Run testing on a trained UNet TrajGRU model: generate predictions, save arrays, and compute metrics.

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
    base_ch : int, optional
        Base number of channels for U-Net encoder/decoder (default: 32).
    bottleneck_dims : list, optional
        Sequence of widths for the bottleneck TrajGRU layers (e.g., (64, 32)).
        Each entry corresponds to a DoubleTrajGRUBlock (i.e., two TrajGRU cells per entry).
        The number of entries determines the depth of the bottleneck.
        If None, defaults to (base_ch*4,) for a single bottleneck stage.
    kernel : int, optional
        Kernel size for all convolutions in the U-Net (default: 3).
    L : int, optional
        Number of flow fields for all TrajGRU layers across encoder, bottleneck, and decoder (default: 5).
    which : str, optional
        Which checkpoint to load - 'best' for best validation checkpoint or 'latest' (default: 'best').
    device : str, optional
        Device to run inference on (default: 'cpu').
    save_arrays : bool, optional
        Whether to save predictions and targets as memory-mapped .npy files (default: True).
        Files will be named 'test_preds_dBZ.npy' and 'test_targets_dBZ.npy'.
    predictions_dir : str, optional
        Directory to save large prediction/target files (default: same as run_dir).
        If None, files are saved in run_dir. If specified, creates the directory if it doesn't exist.
    """
    if bottleneck_dims is None:
        bottleneck_dims = [base_ch*4]

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

    # Use mmap loading for large datasets
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

    model = UNetTrajGRU(in_ch=C, out_ch=C, base_ch=base_ch, bottleneck_dims=bottleneck_dims, seq_len=seq_len_in, kernel=kernel, L=L)
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and 'model' in st:
        st=st['model']
    model.load_state_dict(st)
    model.to(device).eval()

    N = len(test_ds)
    if save_arrays:
        preds_memmap = np.memmap(predictions_dir/"test_preds_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
        gts_memmap   = np.memmap(predictions_dir/"test_targets_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
    else:
        preds_memmap = None
        gts_memmap = None

    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_sums = {f"mse_{r_min}_{r_max}": 0.0 for r_min, r_max in ranges}
    mse_counts = {f"mse_{r_min}_{r_max}": 0 for r_min, r_max in ranges}
    metrics_accumulator = init_forecasting_metrics_accumulator()

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc='Testing', total=len(dl)):
            xb = xb.to(device)

            if yb.ndim == 4:
                yb = yb.unsqueeze(2)
            out_n = model(xb)
            if out_n.shape[2] == 1:
                out_n = out_n.squeeze(2)
            if yb.shape[2] == 1:
                yb = yb.squeeze(2)
            
            metrics_accumulator = accumulate_forecasting_metrics_batch(
                metrics_accumulator, out_n, yb, maxv=maxv, eps=eps
            )
            
            out_n = out_n.cpu().numpy()
            yb = yb.cpu().numpy()
            out_n_dBZ = out_n * (maxv+eps)
            yb_dBZ = yb * (maxv+eps)
            batch_size = out_n.shape[0]
            if save_arrays:
                preds_memmap[idx:idx+batch_size] = out_n_dBZ
                gts_memmap[idx:idx+batch_size] = yb_dBZ

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
        meta = {
            'shape': (N, C, H, W),
            'dtype': 'float32'
        }
        np.savez(predictions_dir/"test_preds_dBZ_meta.npz", **meta)
        np.savez(predictions_dir/"test_targets_dBZ_meta.npz", **meta)

    final_metrics = compute_final_forecasting_metrics(metrics_accumulator)
    
    mse_by_range = {}
    for r_min, r_max in ranges:
        key = f"mse_{r_min}_{r_max}"
        if mse_counts[key] > 0:
            mse_by_range[key] = mse_sums[key] / mse_counts[key]
        else:
            mse_by_range[key] = np.nan

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    import json
    test_metrics = {
        "b_mse": float(final_metrics['b_mse']),
        "mse": float(final_metrics['mse']),
        "csi_by_threshold": {k: float(v) for k, v in final_metrics['csi_by_threshold'].items()},
        "hss_by_threshold": {k: float(v) for k, v in final_metrics['hss_by_threshold'].items()},
        "confusion_by_threshold": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in final_metrics.get('confusion_by_threshold', {}).items()},
        "mse_by_range": {k: float(v) if not np.isnan(v) else None for k, v in mse_by_range.items()}
    }
    
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    with open(results_dir / "test_mse_by_ranges.json", "w") as f:
        json.dump({k: float(v) if not np.isnan(v) else None for k, v in mse_by_range.items()}, f, indent=2)
    
    print("MSE by reflectivity range:")
    for range_name, mse in mse_by_range.items():
        print(f"  {range_name}: {mse:.4f}")
    
    print("\nTest metrics:")
    print(f"  B-MSE: {final_metrics['b_mse']:.4f}")
    print(f"  MSE: {final_metrics['mse']:.4f}")
    for th, csi in final_metrics['csi_by_threshold'].items():
        print(f"  CSI {th}: {csi:.4f}")
    for th, hss in final_metrics['hss_by_threshold'].items():
        print(f"  HSS {th}: {hss:.4f}")
    
    if save_arrays:
        print(f"\nSaved test_preds_dBZ.npy + test_targets_dBZ.npy → {predictions_dir}")
    print(f"Saved test_metrics.json → {results_dir}")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or validate a UNet TrajGRU radar forecasting model.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    train_parser.add_argument("--base_ch", type=int, default=32, help="Base number of channels for U-Net encoder/decoder (default: 32)")
    train_parser.add_argument("--bottleneck_dims", type=str, default="128", help="Comma-separated list of bottleneck channel dimensions. Examples: '128' (1 stage), '64,32' (2 stages), '128,64,32' (3 stages)")
    train_parser.add_argument("--kernel", type=int, default=3, help="Kernel size for all convolutions (default: 3)")
    train_parser.add_argument("--L", type=int, default=5, help="Number of flow fields for all TrajGRU layers (default: 5)")
    train_parser.add_argument("--npy_path", type=str, default="data/processed/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    train_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    train_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    train_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    train_parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda' or 'cpu')")
    train_parser.add_argument("--loss_name", type=str, default="mse", help="Loss function: mse, weighted_mse, or b_mse")
    train_parser.add_argument("--loss_weight_thresh", type=float, default=30.0, help="Threshold in dBZ to apply higher loss weighting for weighted_mse loss (default: 30.0 dBZ). Only used when --loss_name=weighted_mse")
    train_parser.add_argument("--loss_weight_high", type=float, default=10.0, help="Weight multiplier for pixels above threshold in weighted_mse loss (default: 10.0). Only used when --loss_name=weighted_mse")
    train_parser.add_argument("--patch_size", type=int, default=64, help="Size of spatial patches to extract (default: 64)")
    train_parser.add_argument("--patch_stride", type=int, default=32, help="Stride for patch extraction (default: 32)")
    train_parser.add_argument("--patch_thresh", type=float, default=35.0, help="Threshold in dBZ for extracting patches (default: 35.0 dBZ)")
    train_parser.add_argument("--patch_frac", type=float, default=0.01, help="Minimum fraction of pixels in patch above threshold (default: 0.01)")
    train_parser.add_argument("--use_patches", type=str, default="False", help="Whether to use patch-based training: True or False (default: False)")
    train_parser.add_argument("--wandb_project", type=str, default="radar-forecasting", help="wandb project name")
    train_parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    train_parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs with no improvement before early stopping (default: 10). Set to 0 or negative to disable early stopping.")

    # Subparser for test
    test_parser = subparsers.add_parser("test", help="Test model: generate predictions and compute metrics")
    test_parser.add_argument("--npy_path", type=str, required=True, help="Path to input .npy radar file")
    test_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    test_parser.add_argument("--base_ch", type=int, default=32, help="Base number of channels for U-Net encoder/decoder (default: 32)")
    test_parser.add_argument("--bottleneck_dims", type=str, default="128", help="Comma-separated list of bottleneck channel dimensions. Examples: '128' (1 stage), '64,32' (2 stages), '128,64,32' (3 stages)")
    test_parser.add_argument("--kernel", type=int, default=3, help="Kernel size for all convolutions (default: 3)")
    test_parser.add_argument("--L", type=int, default=5, help="Number of flow fields for all TrajGRU layers (default: 5)")
    test_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    test_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    test_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    test_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    test_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    test_parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: 'cpu')")
    test_parser.add_argument("--save_arrays", type=str, default="True", help="Whether to save predictions and targets as .npy files (True/False)")
    test_parser.add_argument("--predictions_dir", type=str, default=None, help="Directory to save large prediction/target files (default: same as run_dir)")

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
        # Save arguments to save_dir/train_args.json
        import json, os
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
        bottleneck_dims = parse_int_list(args.bottleneck_dims)
        L = args.L
        
        train_radar_model(
            npy_path=args.npy_path,
            save_dir=args.save_dir,
            args=args,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            lr=args.lr,
            base_ch=args.base_ch,
            bottleneck_dims=bottleneck_dims,
            kernel=args.kernel,
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
        # Convert save_arrays string to boolean
        if isinstance(args.save_arrays, str):
            if args.save_arrays.lower() in ["true", "1", "yes"]:
                args.save_arrays = True
            elif args.save_arrays.lower() in ["false", "0", "no"]:
                args.save_arrays = False
            else:
                raise ValueError("--save_arrays must be True or False")
        # Save arguments to run_dir/test_args.json
        import json, os
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, "test_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        bottleneck_dims = parse_int_list(args.bottleneck_dims)
        L = args.L
        
        predict_test_set(
            npy_path=args.npy_path,
            run_dir=args.run_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            base_ch=args.base_ch,
            bottleneck_dims=bottleneck_dims,
            kernel=args.kernel,
            L=L,
            which=args.which,
            device=args.device,
            save_arrays=args.save_arrays,
            predictions_dir=args.predictions_dir,
        )

