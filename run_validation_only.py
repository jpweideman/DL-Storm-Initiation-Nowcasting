#!/usr/bin/env python3
"""
Script to run validation on pre-trained models using the exact same validation code from training scripts.
This ensures we get identical results to what would be computed during training.
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import ast
from tqdm import tqdm

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.conv_lstm import ConvLSTM
from src.models.unet_conv_lstm import UNetConvLSTM
from src.models.cnn_3d import CNN3D
from src.models.unet_3d_cnn import UNet3DCNN
from src.models.traj_gru import TrajGRU
from src.models.traj_gru_baseline import TrajGRUEncoderDecoder
from src.training.utils import RadarWindowDataset, set_seed
from src.utils.storm_utils import compute_forecasting_metrics

set_seed(123)

def run_validation_epoch_exact(model, val_loader, device, maxv=85.0, eps=1e-6):
    """
    Run validation epoch using the EXACT same code as in the training scripts.
    This ensures we get identical results.
    """
    model.eval()
    
    # Initialize running statistics for batch-by-batch metric computation
    # For MSE by dBZ bins
    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_by_range = {f"mse_{r_min}_{r_max}": {"sum": 0.0, "count": 0} for r_min, r_max in ranges}
    
    # For overall MSE
    total_mse_sum = 0.0
    total_pixels = 0
    
    # For batch-by-batch metrics (B-MSE, CSI, HSS)
    total_b_mse = 0.0
    total_samples = 0
    total_csi = {2: 0.0, 5: 0.0, 10: 0.0, 30: 0.0, 45: 0.0}
    total_hss = {2: 0.0, 5: 0.0, 10: 0.0, 30: 0.0, 45: 0.0}
    total_csi_count = 0
    total_hss_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            # Convert to dBZ for metric computation
            pred_dBZ = pred.detach() * maxv
            target_dBZ = yb.detach() * maxv
            
            # Ensure both tensors have the same shape for metric computation
            if pred_dBZ.shape != target_dBZ.shape:
                # Remove singleton dimensions to match shapes
                pred_dBZ = pred_dBZ.squeeze()
                target_dBZ = target_dBZ.squeeze()
            
            # Compute MSE by dBZ bins batch by batch
            for r_min, r_max in ranges:
                mask = (target_dBZ >= r_min) & (target_dBZ < r_max)
                n_pix = torch.sum(mask).item()
                if n_pix > 0:
                    mse_bin = torch.sum(((pred_dBZ[mask] - target_dBZ[mask]) ** 2)).item()
                    mse_by_range[f"mse_{r_min}_{r_max}"]["sum"] += mse_bin
                    mse_by_range[f"mse_{r_min}_{r_max}"]["count"] += n_pix
            
            # Compute overall MSE
            batch_mse = torch.sum(((pred_dBZ - target_dBZ) ** 2)).item()
            total_mse_sum += batch_mse
            total_pixels += pred_dBZ.numel()
            
            # Compute storm metrics for this batch using normalized data for consistency
            # Pass maxv and eps to ensure same normalization as training loss
            batch_metrics = compute_forecasting_metrics(
                pred.detach().cpu().numpy(), 
                yb.detach().cpu().numpy(),
                maxv=maxv, 
                eps=eps
            )
            
            # Accumulate B-MSE from compute_forecasting_metrics
            total_b_mse += batch_metrics['b_mse'] * xb.size(0)
            total_samples += xb.size(0)
            
            # Accumulate CSI and HSS for this batch
            for th in total_csi:
                csi_key = f"csi_{th}"
                hss_key = f"hss_{th}"
                if csi_key in batch_metrics['csi_by_threshold']:
                    total_csi[th] += batch_metrics['csi_by_threshold'][csi_key]
                    total_hss[th] += batch_metrics['hss_by_threshold'][hss_key]
            
            total_csi_count += 1
            total_hss_count += 1
    
    # Finalize MSE by dBZ bins
    final_mse_by_range = {}
    for range_name, stats in mse_by_range.items():
        if stats["count"] > 0:
            final_mse_by_range[range_name] = stats["sum"] / stats["count"]
        else:
            final_mse_by_range[range_name] = np.nan
    
    # Finalize overall MSE
    final_mse = total_mse_sum / total_pixels if total_pixels > 0 else np.nan
    
    # Finalize storm metrics
    final_storm_metrics = {}
    if total_samples > 0:
        final_storm_metrics['b_mse'] = total_b_mse / total_samples
        final_storm_metrics['csi_by_threshold'] = {}
        final_storm_metrics['hss_by_threshold'] = {}
        for th in total_csi:
            final_storm_metrics['csi_by_threshold'][f"csi_{th}"] = total_csi[th] / total_csi_count
            final_storm_metrics['hss_by_threshold'][f"hss_{th}"] = total_hss[th] / total_hss_count
    
    # Print results exactly as in training
    print("Validation metrics:")
    print(f"  B-MSE: {final_storm_metrics['b_mse']:.4f}")
    for th, csi in final_storm_metrics['csi_by_threshold'].items():
        print(f"  CSI {th}: {csi:.4f}")
    for th, hss in final_storm_metrics['hss_by_threshold'].items():
        print(f"  HSS {th}: {hss:.4f}")
    print(f"  MSE: {final_mse:.4f}")
    for range_name, mse_val in final_mse_by_range.items():
        print(f"  {range_name}: {mse_val:.4f}")
    
    return {
        "b_mse": final_storm_metrics['b_mse'],
        "mse": final_mse,
        "csi_by_threshold": final_storm_metrics['csi_by_threshold'],
        "hss_by_threshold": final_storm_metrics['hss_by_threshold'],
        "mse_by_range": final_mse_by_range
    }

def validate_model(model_type, checkpoint_path, save_dir, npy_path, **model_params):
    """Generic validation function that works for all model types."""
    
    # Initialize model based on type
    if model_type == "conv_lstm":
        model = ConvLSTM(in_ch=14, hidden_dims=model_params['hidden_dims'], kernel=model_params['kernel'])
    elif model_type == "unet_conv_lstm":
        model = UNetConvLSTM(in_ch=14, base_ch=model_params['base_ch'], lstm_hid=model_params['lstm_hid'], kernel=model_params['kernel'])
    elif model_type == "cnn_3d":
        model = CNN3D(in_ch=14, hidden_dims=model_params['hidden_dims'], kernel=model_params['kernel'])
    elif model_type == "unet_3d_cnn":
        model = UNet3DCNN(in_ch=14, base_ch=model_params['base_ch'], bottleneck_ch=model_params['bottleneck_ch'], kernel=model_params['kernel'])
    elif model_type == "traj_gru":
        model = TrajGRU(in_ch=14, hidden_dims=model_params['hidden_dims'], kernel_size=model_params['kernel_size'])
    elif model_type == "traj_gru_baseline":
        # TrajGRU baseline expects lists for all parameters
        hidden_channels = list(model_params['hidden_dims'])  # Convert tuple to list
        kernel_size = [model_params['kernel_size']] * len(hidden_channels)  # Repeat kernel_size for each layer
        L = [13, 13, 9]  # From training args
        conv_kernels = [5, 5, 3]  # From training args  
        conv_strides = [3, 2, 1]  # From training args
        model = TrajGRUEncoderDecoder(input_channels=14, hidden_channels=hidden_channels, kernel_size=kernel_size, L=L, conv_kernels=conv_kernels, conv_strides=conv_strides)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Loading {model_type} model from {checkpoint_path}")
    
    # Load checkpoint
    device = model_params.get('device', 'cuda')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # Create validation dataset (same as in training scripts)
    cube = np.load(npy_path, mmap_mode='r')
    T, C, H, W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")
    
    # Use same parameters as training
    seq_len_in = model_params.get('seq_len_in', 10)
    seq_len_out = model_params.get('seq_len_out', 1)
    train_val_test_split = model_params.get('train_val_test_split', (0.5, 0.5, 0))
    batch_size = model_params.get('batch_size', 2)
    maxv = 85.0
    eps = 1e-6
    
    train_frac, val_frac, _ = train_val_test_split
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    
    # Create validation dataset exactly as in training
    full_ds = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv=maxv)
    val_ds = Subset(full_ds, list(range(n_train, n_train + n_val)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Validation: {len(val_ds)} samples")
    
    # Run validation using exact same code
    validation_metrics = run_validation_epoch_exact(model, val_loader, device, maxv=maxv, eps=eps)
    
    # Save results
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(exist_ok=True)
    
    metrics_to_save = {
        "model_type": model_type,
        "model_params": model_params,
        "validation_metrics": validation_metrics
    }
    
    output_file = results_dir / "updated_validation_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Validation metrics saved to {output_file}")
    return validation_metrics

def main():
    parser = argparse.ArgumentParser(description="Run validation on pre-trained models using exact training code")
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["conv_lstm", "unet_conv_lstm", "cnn_3d", "unet_3d_cnn", "traj_gru", "traj_gru_baseline"], 
                       help="Type of model to validate")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save validation results")
    parser.add_argument("--npy_path", type=str, default="data/processed/ZH_radar_dataset_20_40.npy",
                       help="Path to input .npy radar file")
    parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length")
    parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length")
    parser.add_argument("--train_val_test_split", type=str, default="(0.5,0.5,0)",
                       help="Train/val/test split as tuple string")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Model-specific arguments
    parser.add_argument("--hidden_dims", type=str, help="Hidden dimensions for ConvLSTM/3D-CNN/TrajGRU (e.g., '(256,256)')")
    parser.add_argument("--kernel_size", type=int, help="Kernel size for ConvLSTM/3D-CNN/TrajGRU")
    parser.add_argument("--base_ch", type=int, help="Base channels for UNet models")
    parser.add_argument("--lstm_hid", type=str, help="LSTM hidden dims for UNet-ConvLSTM (e.g., '64' or '(64,128)')")
    parser.add_argument("--bottleneck_ch", type=int, help="Bottleneck channels for UNet-3D-CNN")
    parser.add_argument("--kernel", type=int, help="Kernel size for UNet models")
    
    args = parser.parse_args()
    
    # Parse train_val_test_split
    try:
        train_val_test_split = ast.literal_eval(args.train_val_test_split)
    except:
        raise ValueError("train_val_test_split must be a valid tuple string like '(0.5,0.5,0)'")
    
    # Build model parameters dict
    model_params = {
        'seq_len_in': args.seq_len_in,
        'seq_len_out': args.seq_len_out,
        'train_val_test_split': train_val_test_split,
        'batch_size': args.batch_size,
        'device': args.device
    }
    
    # Add model-specific parameters
    if args.model_type in ["conv_lstm", "cnn_3d"]:
        if not args.hidden_dims or not args.kernel_size:
            raise ValueError(f"--hidden_dims and --kernel_size are required for {args.model_type}")
        try:
            model_params['hidden_dims'] = ast.literal_eval(args.hidden_dims)
        except:
            raise ValueError("hidden_dims must be a valid tuple string like '(256,256)'")
        model_params['kernel'] = args.kernel_size  # CNN3D and ConvLSTM use 'kernel'
    
    elif args.model_type in ["traj_gru", "traj_gru_baseline"]:
        if not args.hidden_dims or not args.kernel_size:
            raise ValueError(f"--hidden_dims and --kernel_size are required for {args.model_type}")
        try:
            model_params['hidden_dims'] = ast.literal_eval(args.hidden_dims)
        except:
            raise ValueError("hidden_dims must be a valid tuple string like '(256,256)'")
        model_params['kernel_size'] = args.kernel_size  # TrajGRU uses 'kernel_size'
    
    elif args.model_type == "unet_conv_lstm":
        if not args.base_ch or not args.lstm_hid or not args.kernel:
            raise ValueError("--base_ch, --lstm_hid, and --kernel are required for UNet-ConvLSTM")
        try:
            model_params['lstm_hid'] = ast.literal_eval(args.lstm_hid)
        except:
            raise ValueError("lstm_hid must be a valid int or tuple string like '64' or '(64,128)'")
        model_params['base_ch'] = args.base_ch
        model_params['kernel'] = args.kernel
    
    elif args.model_type == "unet_3d_cnn":
        if not args.base_ch or not args.bottleneck_ch or not args.kernel:
            raise ValueError("--base_ch, --bottleneck_ch, and --kernel are required for UNet-3D-CNN")
        model_params['base_ch'] = args.base_ch
        model_params['bottleneck_ch'] = args.bottleneck_ch
        model_params['kernel'] = args.kernel
    
    # Run validation
    validate_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir,
        npy_path=args.npy_path,
        **model_params
    )

if __name__ == "__main__":
    main()
