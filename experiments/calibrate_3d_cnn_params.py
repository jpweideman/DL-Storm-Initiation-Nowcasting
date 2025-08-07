#!/usr/bin/env python3
"""
Script to calibrate 3D CNN parameters for fair comparison with UNet 3D CNN.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.cnn_3d import CNN3D
from src.models.unet_3d_cnn import UNet3DCNN
from tqdm import tqdm

def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(name, model, params):
    """Print model information."""
    print(f"{name}: {params:,} parameters")

def main():
    print("=== 3D CNN Parameter Calibration for Fair Comparisons ===\n")
    
    in_ch = 14
    seq_len = 10
    
    # Target UNet 3D CNN configurations
    unet_configs = {
        "UNet 3D CNN Small": {"base_ch": 16, "bottleneck_dims": (32,)},
        "UNet 3D CNN Medium": {"base_ch": 32, "bottleneck_dims": (64,)},
        "UNet 3D CNN Large": {"base_ch": 64, "bottleneck_dims": (128,)},
    }
    
    # Calculate UNet parameters
    unet_params = {}
    for name, config in unet_configs.items():
        model = UNet3DCNN(in_ch=in_ch, out_ch=in_ch, **config)
        params = count_parameters(model)
        unet_params[name] = params
        print_model_info(name, model, params)
    
    print("\n" + "="*60)
    print("3D CNN CONFIGURATIONS TO MATCH UNET PARAMETERS")
    print("="*60)
    
    # Find 3D CNN configurations to match UNet parameters
    for unet_name, target_params in unet_params.items():
        print(f"\n--- Matching {unet_name} ({target_params:,} params) ---")
        
        # Try different hidden dimension configurations
        configs_to_try = [
            # Single layer configurations
            (target_params // 1000,),  # Rough estimate
            (target_params // 800,),
            (target_params // 600,),
            (target_params // 400,),
            # Two layer configurations
            (target_params // 1200, target_params // 1200),
            (target_params // 1000, target_params // 1000),
            (target_params // 800, target_params // 800),
            # Three layer configurations
            (target_params // 1500, target_params // 1500, target_params // 1500),
            (target_params // 1200, target_params // 1200, target_params // 1200),
        ]
        
        best_config = None
        best_diff = float('inf')
        
        for hidden_dims in configs_to_try:
            try:
                model = CNN3D(in_ch=in_ch, hidden_dims=hidden_dims, kernel=3)
                params = count_parameters(model)
                diff = abs(params - target_params)
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = (hidden_dims, params)
                
                print(f"  hidden_dims={hidden_dims}: {params:,} params (diff: {diff:,})")
                
            except Exception as e:
                print(f"  hidden_dims={hidden_dims}: Error - {e}")
        
        if best_config:
            hidden_dims, params = best_config
            print(f"\n  BEST MATCH: hidden_dims={hidden_dims}")
            print(f"  Parameters: {params:,} (target: {target_params:,}, diff: {best_diff:,})")
            print(f"  Ratio: {params/target_params:.2f}x")
    
    print("\n" + "="*60)
    print("RECOMMENDED TRAINING CONFIGURATIONS")
    print("="*60)
    
    # Generate recommended training commands
    for unet_name, target_params in unet_params.items():
        print(f"\n--- {unet_name} Comparison ---")
        
        # Find the best matching 3D CNN config
        if "Small" in unet_name:
            cnn_hidden = (64, 64)  # Approximate match for small
        elif "Medium" in unet_name:
            cnn_hidden = (128, 128)  # Approximate match for medium  
        elif "Large" in unet_name:
            cnn_hidden = (256, 256)  # Approximate match for large
        
        unet_config = unet_configs[unet_name]
        
        print(f"\n3D CNN Training Command:")
        print(f"python src/training/train_3D_cnn.py train \\")
        print(f"  --save_dir experiments/runs/3dcnn_{unet_name.lower().replace(' ', '_').replace('unet_3d_cnn_', '')} \\")
        print(f"  --npy_path data/processed/ZH_radar_dataset_20_40.npy \\")
        print(f"  --seq_len_in 10 --seq_len_out 1 \\")
        print(f"  --train_val_test_split \"(0.5,0.5,0)\" \\")
        print(f"  --batch_size 8 --lr 0.0001 --epochs 25 \\")
        print(f"  --device cuda --loss_name b_mse \\")
        print(f"  --hidden_channels \"{','.join(map(str, cnn_hidden))}\" --kernel_size 3 \\")
        print(f"  --wandb_project experiments_validation_comparison")
        
        print(f"\nUNet 3D CNN Training Command:")
        print(f"python src/training/train_unet_3D_cnn.py train \\")
        print(f"  --save_dir experiments/runs/{unet_name.lower().replace(' ', '_')} \\")
        print(f"  --npy_path data/processed/ZH_radar_dataset_20_40.npy \\")
        print(f"  --seq_len_in 10 --seq_len_out 1 \\")
        print(f"  --train_val_test_split \"(0.5,0.5,0)\" \\")
        print(f"  --batch_size 8 --lr 0.0001 --epochs 25 \\")
        print(f"  --device cuda --loss_name b_mse \\")
        print(f"  --base_ch {unet_config['base_ch']} --bottleneck_dims \"{','.join(map(str, unet_config['bottleneck_dims']))}\" --kernel 3 \\")
        print(f"  --wandb_project experiments_validation_comparison")

if __name__ == "__main__":
    main() 