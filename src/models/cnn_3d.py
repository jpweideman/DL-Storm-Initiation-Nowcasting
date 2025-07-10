import torch
import torch.nn as nn


def conv3d_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    """
    Create a 3D convolutional block with BatchNorm and ReLU.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    kernel_size : int, optional
        Kernel size (default: 3).
    stride : int, optional
        Stride (default: 1).
    padding : int, optional
        Padding (default: 1).

    Returns
    -------
    nn.Sequential
        3D convolutional block.
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )


class Radar3DCNN(nn.Module):
    """
    Simple 3D CNN for spatiotemporal radar forecasting.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    hidden_dims : tuple
        Hidden channels for each layer.
    kernel : int, optional
        Kernel size (default: 3).
    seq_len_in : int, optional
        Input sequence length (default: 10).
    """
    def __init__(self, in_ch, hidden_dims=(64, 64), kernel=3, seq_len_in=10):
        super().__init__()
        layers = []
        last_ch = in_ch
        for h in hidden_dims:
            layers.append(conv3d_block(last_ch, h, kernel_size=kernel, padding=kernel//2))
            last_ch = h
        self.encoder = nn.Sequential(*layers)
        # Output: (B, hidden_dims[-1], seq_in, H, W)
        # Reduce temporal dimension (seq_in) to 1 by pooling, then output to in_ch
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.to_out = nn.Conv2d(hidden_dims[-1], in_ch, 1)
        
    def forward(self, x):
        # x: (B, seq_in, C, H, W) → (B, C, seq_in, H, W)
        B, S, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, S, H, W)
        x = self.encoder(x)            # (B, hidden, S, H, W)
        x = self.temporal_pool(x)      # (B, hidden, 1, H, W)
        x = x.squeeze(2)               # (B, hidden, H, W)
        x = self.to_out(x)             # (B, in_ch, H, W)
        return x 