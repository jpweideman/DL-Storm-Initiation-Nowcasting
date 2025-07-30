import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def atomic_save(obj, path):
    """
    Atomically save a PyTorch object to disk to avoid partial writes.
    """
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def mse_loss(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute MSE in dBZ units.
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

def b_mse_loss(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute the B-MSE, using the weighting scheme:
    w(x) = 1 if x < 2
           2 if 2 <= x < 5
           5 if 5 <= x < 10
           10 if 10 <= x < 30
           30 if 30 <= x < 45
           45 if x >= 45
    """
    pred_dBZ = pred * (maxv + eps)
    target_dBZ = target * (maxv + eps)
    w = torch.ones_like(target_dBZ)
    w = torch.where(target_dBZ < 2, torch.tensor(1.0, device=target.device), w)
    w = torch.where((target_dBZ >= 2) & (target_dBZ < 5), torch.tensor(2.0, device=target.device), w)
    w = torch.where((target_dBZ >= 5) & (target_dBZ < 10), torch.tensor(5.0, device=target.device), w)
    w = torch.where((target_dBZ >= 10) & (target_dBZ < 30), torch.tensor(10.0, device=target.device), w)
    w = torch.where((target_dBZ >= 30) & (target_dBZ < 45), torch.tensor(30.0, device=target.device), w)
    w = torch.where(target_dBZ >= 45, torch.tensor(45.0, device=target.device), w)
    b_mse = (w * (pred_dBZ - target_dBZ) ** 2).sum() / w.sum()
    return b_mse 