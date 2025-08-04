"""
Radar Data Loading Classes

This module contains PyTorch Dataset classes for loading radar data for training
spatiotemporal prediction models. Supports both full-frame and patch-based training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RadarWindowDataset(Dataset):
    """
    Dataset for loading radar data in sliding window format.
    
    This dataset loads full radar frames and creates input-output pairs
    for spatiotemporal prediction models.
    
    Parameters
    ----------
    cube : np.ndarray
        Radar data cube of shape (T, C, H, W) in original scale.
    seq_in : int
        Number of input time steps.
    seq_out : int
        Number of output time steps.
    maxv : float, optional
        Maximum value for normalization (default: 85.0).
    """
    
    def __init__(self, cube, seq_in, seq_out, maxv=85.0):
        self.cube = cube
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.maxv = maxv
        self.last = cube.shape[0] - seq_in - seq_out + 1

    def __len__(self):
        return self.last

    def __getitem__(self, i):
        # Load the required slice, clip negatives, and normalize
        X = np.maximum(self.cube[i:i+self.seq_in], 0) / (self.maxv + 1e-6)
        Y = np.maximum(self.cube[i+self.seq_in:i+self.seq_in+self.seq_out], 0) / (self.maxv + 1e-6)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32).squeeze(0)
        return torch.from_numpy(X), torch.from_numpy(Y)


class PatchRadarWindowDataset(Dataset):
    """
    Dataset for loading radar data in patch-based format.
    
    This dataset extracts patches from radar frames and creates input-output pairs
    for training models on smaller spatial regions. Only patches with sufficient
    signal (above threshold) are included.
    
    Parameters
    ----------
    cube : np.ndarray
        Radar data cube of shape (T, C, H, W) in original scale.
    seq_in : int
        Number of input time steps.
    seq_out : int
        Number of output time steps.
    patch_size : int, optional
        Size of spatial patches (default: 64).
    patch_stride : int, optional
        Stride for patch extraction (default: 64).
    patch_thresh : float, optional
        Threshold for patch selection in dBZ (default: 35).
    patch_frac : float, optional
        Minimum fraction of pixels above threshold (default: 0.01).
    patch_index_path : str, optional
        Path to save/load patch indices for caching (default: None).
    maxv : float, optional
        Maximum value for normalization (default: 85.0).
    """
    
    def __init__(self, cube, seq_in, seq_out, patch_size=64, patch_stride=64, 
                 patch_thresh=35, patch_frac=0.01, patch_index_path=None, maxv=85.0):
        self.cube = cube
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_thresh = patch_thresh  # dBZ value
        self.patch_frac = patch_frac
        self.maxv = maxv
        self.patches = []  # List of (t, y, x)
        
        T, C, H, W = cube.shape
        last = T - seq_in - seq_out + 1
        
        # Normalize patch_thresh for internal use
        patch_thresh_normalized = self.patch_thresh / (self.maxv + 1e-6)
        
        # Patch index caching
        if patch_index_path is not None and os.path.exists(patch_index_path):
            print(f"Loading patch indices from {patch_index_path}")
            self.patches = np.load(patch_index_path, allow_pickle=True).tolist()
        else:
            total_patches_checked = 0
            patches_found = 0
            for t in tqdm(range(last), desc='Extracting patches'):
                for y in range(0, H - patch_size + 1, patch_stride):
                    for x in range(0, W - patch_size + 1, patch_stride):
                        if y + patch_size <= H and x + patch_size <= W:
                            total_patches_checked += 1
                            Y_patch = np.maximum(
                                cube[t+seq_in:t+seq_in+seq_out, :, y:y+patch_size, x:x+patch_size], 0
                            ) / (maxv + 1e-6)
                            total_pix = Y_patch.size
                            n_above = (Y_patch > patch_thresh_normalized).sum()
                            if n_above / total_pix >= patch_frac:
                                self.patches.append((t, y, x))
                                patches_found += 1
            
            print(f"Patch extraction summary:")
            print(f"  Total patches checked: {total_patches_checked}")
            print(f"  Patches found: {patches_found}")
            
            if patches_found == 0:
                print(f"WARNING: No patches found! Consider:")
                print(f"  - Lowering patch_thresh (currently {patch_thresh})")
                print(f"  - Lowering patch_frac (currently {patch_frac})")
                print(f"  - Checking if data has enough high-intensity regions")
            
            if patch_index_path is not None:
                np.save(patch_index_path, np.array(self.patches, dtype=object))
                print(f"Saved patch indices to {patch_index_path}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        t, y, x = self.patches[i]
        X_patch = np.maximum(
            self.cube[t:t+self.seq_in, :, y:y+self.patch_size, x:x+self.patch_size], 0
        ) / (self.maxv + 1e-6)
        Y_patch = np.maximum(
            self.cube[t+self.seq_in:t+self.seq_in+self.seq_out, :, y:y+self.patch_size, x:x+self.patch_size], 0
        ) / (self.maxv + 1e-6)
        X_patch = X_patch.astype(np.float32)
        Y_patch = Y_patch.astype(np.float32).squeeze(0)
        return torch.from_numpy(X_patch), torch.from_numpy(Y_patch), t, y, x


class NonNormalizedRadarWindowDataset(Dataset):
    """
    Dataset for loading radar data without normalization.
    
    This dataset loads full radar frames without applying normalization,
    useful for models that handle raw dBZ values directly.
    
    Parameters
    ----------
    cube : np.ndarray
        Radar data cube of shape (T, C, H, W) in original dBZ scale.
    seq_in : int
        Number of input time steps.
    seq_out : int
        Number of output time steps.
    """
    
    def __init__(self, cube, seq_in, seq_out):
        X, Y = [], []
        last = cube.shape[0] - seq_in - seq_out + 1
        for t in range(last):
            X.append(cube[t:t+seq_in])
            Y.append(cube[t+seq_in:t+seq_in+seq_out].squeeze(0))
        self.X = np.stack(X).astype(np.float32)  # (N,seq_in,C,H,W)
        self.Y = np.stack(Y).astype(np.float32)  # (N,C,H,W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]) 