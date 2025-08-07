#!/usr/bin/env python3
"""
Script to count parameters for different model configurations to design fair comparisons.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.conv_lstm import ConvLSTM
from src.models.unet_conv_lstm import UNetConvLSTM
from src.models.cnn_3d import CNN3D
from src.models.unet_3d_cnn import UNet3DCNN
from tqdm import tqdm

def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(name, model, in_ch=14, seq_len=10):
    """Print model information including parameter count."""
    total_params = count_parameters(model)
    print(f"\n{name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model: {model}")
    return total_params

def main():
    print("=== Model Parameter Comparison for Fair Experiments ===\n")
    
    # Test configurations
    in_ch = 14
    seq_len = 10
    
    # Configuration 1: Small models (~100K-200K parameters)
    print("=== SMALL MODELS (~100K-200K parameters) ===")
    
    # ConvLSTM small
    convlstm_small = ConvLSTM(in_ch=in_ch, hidden_dims=(32, 32), kernel=3)
    convlstm_small_params = print_model_info("ConvLSTM Small", convlstm_small)
    
    # UNet ConvLSTM small
    unet_convlstm_small = UNetConvLSTM(in_ch=in_ch, out_ch=in_ch, base_ch=16, lstm_hid=32, seq_len=seq_len, kernel=3)
    unet_convlstm_small_params = print_model_info("UNet ConvLSTM Small", unet_convlstm_small)
    
    # 3D CNN small
    cnn3d_small = CNN3D(in_ch=in_ch, hidden_dims=(32, 32), kernel=3)
    cnn3d_small_params = print_model_info("3D CNN Small", cnn3d_small)
    
    # UNet 3D CNN small
    unet_3dcnn_small = UNet3DCNN(in_ch=in_ch, out_ch=in_ch, base_ch=16, bottleneck_dims=(32,), kernel=3)
    unet_3dcnn_small_params = print_model_info("UNet 3D CNN Small", unet_3dcnn_small)
    
    # Configuration 2: Medium models (~500K-1M parameters)
    print("\n=== MEDIUM MODELS (~500K-1M parameters) ===")
    
    # ConvLSTM medium
    convlstm_medium = ConvLSTM(in_ch=in_ch, hidden_dims=(64, 64), kernel=3)
    convlstm_medium_params = print_model_info("ConvLSTM Medium", convlstm_medium)
    
    # UNet ConvLSTM medium
    unet_convlstm_medium = UNetConvLSTM(in_ch=in_ch, out_ch=in_ch, base_ch=32, lstm_hid=64, seq_len=seq_len, kernel=3)
    unet_convlstm_medium_params = print_model_info("UNet ConvLSTM Medium", unet_convlstm_medium)
    
    # 3D CNN medium
    cnn3d_medium = CNN3D(in_ch=in_ch, hidden_dims=(64, 64), kernel=3)
    cnn3d_medium_params = print_model_info("3D CNN Medium", cnn3d_medium)
    
    # UNet 3D CNN medium
    unet_3dcnn_medium = UNet3DCNN(in_ch=in_ch, out_ch=in_ch, base_ch=32, bottleneck_dims=(64,), kernel=3)
    unet_3dcnn_medium_params = print_model_info("UNet 3D CNN Medium", unet_3dcnn_medium)
    
    # Configuration 3: Large models (~2M-5M parameters)
    print("\n=== LARGE MODELS (~2M-5M parameters) ===")
    
    # ConvLSTM large
    convlstm_large = ConvLSTM(in_ch=in_ch, hidden_dims=(128, 128), kernel=3)
    convlstm_large_params = print_model_info("ConvLSTM Large", convlstm_large)
    
    # UNet ConvLSTM large
    unet_convlstm_large = UNetConvLSTM(in_ch=in_ch, out_ch=in_ch, base_ch=64, lstm_hid=128, seq_len=seq_len, kernel=3)
    unet_convlstm_large_params = print_model_info("UNet ConvLSTM Large", unet_convlstm_large)
    
    # 3D CNN large
    cnn3d_large = CNN3D(in_ch=in_ch, hidden_dims=(128, 128), kernel=3)
    cnn3d_large_params = print_model_info("3D CNN Large", cnn3d_large)
    
    # UNet 3D CNN large
    unet_3dcnn_large = UNet3DCNN(in_ch=in_ch, out_ch=in_ch, base_ch=64, bottleneck_dims=(128,), kernel=3)
    unet_3dcnn_large_params = print_model_info("UNet 3D CNN Large", unet_3dcnn_large)
    
    # Summary
    print("\n=== PARAMETER SUMMARY ===")
    print(f"{'Model':<20} {'Small':<10} {'Medium':<10} {'Large':<10}")
    print("-" * 50)
    print(f"{'ConvLSTM':<20} {convlstm_small_params:<10,} {convlstm_medium_params:<10,} {convlstm_large_params:<10,}")
    print(f"{'UNet ConvLSTM':<20} {unet_convlstm_small_params:<10,} {unet_convlstm_medium_params:<10,} {unet_convlstm_large_params:<10,}")
    print(f"{'3D CNN':<20} {cnn3d_small_params:<10,} {cnn3d_medium_params:<10,} {cnn3d_large_params:<10,}")
    print(f"{'UNet 3D CNN':<20} {unet_3dcnn_small_params:<10,} {unet_3dcnn_medium_params:<10,} {unet_3dcnn_large_params:<10,}")
    
    # Fair comparison recommendations
    print("\n=== FAIR COMPARISON RECOMMENDATIONS ===")
    print("For fair comparisons, use these parameter-matched configurations:")
    print()
    print("1. SMALL SCALE (~150K parameters):")
    print("   - ConvLSTM: hidden_dims=(32, 32)")
    print("   - UNet ConvLSTM: base_ch=16, lstm_hid=32")
    print("   - 3D CNN: hidden_dims=(32, 32)")
    print("   - UNet 3D CNN: base_ch=16, bottleneck_dims=(32,)")
    print()
    print("2. MEDIUM SCALE (~600K parameters):")
    print("   - ConvLSTM: hidden_dims=(64, 64)")
    print("   - UNet ConvLSTM: base_ch=32, lstm_hid=64")
    print("   - 3D CNN: hidden_dims=(64, 64)")
    print("   - UNet 3D CNN: base_ch=32, bottleneck_dims=(64,)")
    print()
    print("3. LARGE SCALE (~3M parameters):")
    print("   - ConvLSTM: hidden_dims=(128, 128)")
    print("   - UNet ConvLSTM: base_ch=64, lstm_hid=128")
    print("   - 3D CNN: hidden_dims=(128, 128)")
    print("   - UNet 3D CNN: base_ch=64, bottleneck_dims=(128,)")

if __name__ == "__main__":
    main() 