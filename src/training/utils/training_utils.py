import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Parameters
    ----------
    seed : int, optional
        Random seed value (default: 42).
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

    Parameters
    ----------
    obj : torch.Tensor or dict
        Object to save.
    path : str or pathlib.Path
        Path where to save the object.
    """
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def mse_loss(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute MSE in dBZ units.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values in normalized scale (0-1).
    target : torch.Tensor
        Target values in normalized scale (0-1).
    maxv : float, optional
        Maximum value for denormalization in dBZ (default: 85.0).
    eps : float, optional
        Small epsilon to avoid division by zero (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Mean squared error in dBZ units.
    """
    pred_dBZ = pred * (maxv + eps)
    target_dBZ = target * (maxv + eps)
    return ((pred_dBZ - target_dBZ) ** 2).mean()

def weighted_mse_loss(pred, target, threshold=30.0, weight_high=10.0, maxv=85.0, eps=1e-6):
    """
    Weighted MSE loss in dBZ units, emphasizing high-reflectivity areas.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values in normalized scale (0-1).
    target : torch.Tensor
        Target values in normalized scale (0-1).
    threshold : float, optional
        dBZ value above which to apply weight_high (default: 30.0).
    weight_high : float, optional
        Weight for pixels above threshold (default: 10.0).
    maxv : float, optional
        Maximum value for denormalization in dBZ (default: 85.0).
    eps : float, optional
        Small epsilon to avoid division by zero (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Weighted mean squared error in dBZ units.
    """
    pred_dBZ = pred * (maxv + eps)
    target_dBZ = target * (maxv + eps)
    weight = torch.ones_like(target_dBZ)
    weight[target_dBZ > threshold] = weight_high
    return ((pred_dBZ - target_dBZ) ** 2 * weight).mean()

def b_mse_loss(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute the B-MSE (Balanced Mean Squared Error).

    Uses the weighting scheme:
    - w(x) = 1 if x < 2
    - w(x) = 2 if 2 <= x < 5
    - w(x) = 5 if 5 <= x < 10
    - w(x) = 10 if 10 <= x < 30
    - w(x) = 30 if 30 <= x < 45
    - w(x) = 45 if x >= 45

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values in normalized scale (0-1).
    target : torch.Tensor
        Target values in normalized scale (0-1).
    maxv : float, optional
        Maximum value for denormalization in dBZ (default: 85.0).
    eps : float, optional
        Small epsilon to avoid division by zero (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Balanced mean squared error in dBZ units.
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
    b_mse = (w * (pred_dBZ - target_dBZ) ** 2).mean()
    return b_mse


def init_forecasting_metrics_accumulator(thresholds=[2, 5, 10, 30, 45]):
    """
    Initialize accumulator for global forecasting metrics.
    
    Parameters
    ----------
    thresholds : list, optional
        List of thresholds for CSI/HSS calculation in dBZ (default: [2, 5, 10, 30, 45]).
    
    Returns
    -------
    dict
        Initialized accumulator dictionary.
    """
    return {
        'total_b_mse_sum': 0.0,
        'total_mse_sum': 0.0,
        'total_pixels': 0,
        'total_samples': 0,
        'counts': {th: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for th in thresholds}
    }

def accumulate_forecasting_metrics_batch(accumulator, pred_batch, target_batch, maxv=85.0, eps=1e-6, thresholds=[2, 5, 10, 30, 45]):
    """
    Accumulate forecasting metrics for a single batch.
    
    Parameters
    ----------
    accumulator : dict
        Accumulator dictionary from init_forecasting_metrics_accumulator().
    pred_batch : np.ndarray or torch.Tensor
        Predicted batch of shape (B, C, H, W) or (B, H, W).
    target_batch : np.ndarray or torch.Tensor
        Target batch of shape (B, C, H, W) or (B, H, W).
    maxv : float, optional
        Maximum value for denormalization in dBZ (default: 85.0).
    eps : float, optional
        Small epsilon to avoid division by zero (default: 1e-6).
    thresholds : list, optional
        List of thresholds for CSI/HSS calculation in dBZ (default: [2, 5, 10, 30, 45]).
    
    Returns
    -------
    dict
        Updated accumulator dictionary.
    """
    import numpy as np
    
    if hasattr(pred_batch, 'detach'):
        pred_batch = pred_batch.detach().cpu().numpy()
    if hasattr(target_batch, 'detach'):
        target_batch = target_batch.detach().cpu().numpy()
        
    pred_dBZ = pred_batch * (maxv + eps)
    target_dBZ = target_batch * (maxv + eps)
    
    # B-MSE for this batch
    w = np.ones_like(target_dBZ, dtype=np.float32)
    w = np.where(target_dBZ < 2, 1.0, w)
    w = np.where((target_dBZ >= 2) & (target_dBZ < 5), 2.0, w)
    w = np.where((target_dBZ >= 5) & (target_dBZ < 10), 5.0, w)
    w = np.where((target_dBZ >= 10) & (target_dBZ < 30), 10.0, w)
    w = np.where((target_dBZ >= 30) & (target_dBZ < 45), 30.0, w)
    w = np.where(target_dBZ >= 45, 45.0, w)
    
    batch_b_mse = np.mean(w * (pred_dBZ - target_dBZ) ** 2)
    
    # Accumulate metrics
    accumulator['total_b_mse_sum'] += batch_b_mse * pred_batch.shape[0]
    accumulator['total_mse_sum'] += np.sum((pred_dBZ - target_dBZ) ** 2)
    accumulator['total_pixels'] += pred_dBZ.size
    accumulator['total_samples'] += pred_batch.shape[0]
    
    for th in thresholds:
        pred_bin = (pred_dBZ >= th)
        true_bin = (target_dBZ >= th)
        
        TP = np.logical_and(pred_bin == 1, true_bin == 1).sum()
        FP = np.logical_and(pred_bin == 1, true_bin == 0).sum()
        TN = np.logical_and(pred_bin == 0, true_bin == 0).sum()
        FN = np.logical_and(pred_bin == 0, true_bin == 1).sum()
        
        accumulator['counts'][th]["TP"] += int(TP)
        accumulator['counts'][th]["FP"] += int(FP)
        accumulator['counts'][th]["TN"] += int(TN)
        accumulator['counts'][th]["FN"] += int(FN)
    
    return accumulator

def compute_final_forecasting_metrics(accumulator, thresholds=[2, 5, 10, 30, 45]):
    """
    Compute final forecasting metrics from accumulated counts.
    
    Parameters
    ----------
    accumulator : dict
        Accumulator dictionary from accumulate_forecasting_metrics_batch().
    thresholds : list, optional
        List of thresholds for CSI/HSS calculation in dBZ (default: [2, 5, 10, 30, 45]).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'b_mse': Balanced Mean Squared Error
        - 'mse': Mean Squared Error
        - 'csi_by_threshold': CSI scores for each threshold
        - 'hss_by_threshold': HSS scores for each threshold
        - 'confusion_by_threshold': TP/FP/TN/FN counts for each threshold
    """
    final_b_mse = accumulator['total_b_mse_sum'] / accumulator['total_samples'] if accumulator['total_samples'] > 0 else float('nan')
    final_mse = accumulator['total_mse_sum'] / accumulator['total_pixels'] if accumulator['total_pixels'] > 0 else float('nan')
    
    # CSI and HSS from global confusion matrix
    def compute_csi_hss_from_counts(tp, fp, fn, tn):
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = ((tp * tn) - (fn * fp)) / denom if denom > 0 else 0.0
        return float(csi), float(hss)
    
    csi_by_threshold = {}
    hss_by_threshold = {}
    confusion_by_threshold = {}
    
    for th in thresholds:
        tp = accumulator['counts'][th]["TP"]
        fp = accumulator['counts'][th]["FP"]
        tn = accumulator['counts'][th]["TN"]
        fn = accumulator['counts'][th]["FN"]
        
        csi, hss = compute_csi_hss_from_counts(tp, fp, fn, tn)
        csi_by_threshold[f"csi_{th}"] = csi
        hss_by_threshold[f"hss_{th}"] = hss
        confusion_by_threshold[str(th)] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    
    return {
        "b_mse": final_b_mse,
        "mse": final_mse,
        "csi_by_threshold": csi_by_threshold,
        "hss_by_threshold": hss_by_threshold,
        "confusion_by_threshold": confusion_by_threshold,
    } 