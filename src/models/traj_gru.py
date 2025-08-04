import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajGRUCell(nn.Module):
    """
    TrajGRU Cell for spatiotemporal processing.
    
    A single TrajGRU cell that generates flow fields for warping and applies GRU operations.
    Combines spatial warping with temporal gating for effective spatiotemporal modeling.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    kernel_size : int, optional
        Kernel size for convolutions (default: 3).
    L : int, optional
        Number of flow fields for warping (default: 5).
    zoneout : float, optional
        Zoneout probability for regularization (default: 0.0).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, L=5, zoneout=0.0):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.L = L  # number of allowed flow fields
        self.zoneout = zoneout
        padding = kernel_size // 2
        
        # Input to hidden (i2h)
        self.i2h = nn.Conv2d(input_channels, hidden_channels * 3, kernel_size, padding=padding)
        
        # Hidden to hidden (h2h) after warping, 1x1 conv
        self.ret = nn.Conv2d(hidden_channels * L, hidden_channels * 3, 1)
        
        # Flow generation, separate layers 
        self.i2f_conv1 = nn.Conv2d(input_channels, 32, kernel_size=(5, 5), padding=(2, 2))
        self.h2f_conv1 = nn.Conv2d(hidden_channels, 32, kernel_size=(5, 5), padding=(2, 2))
        self.flows_conv = nn.Conv2d(32, L * 2, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, x, h_prev):
        # Flow generation 
        i2f_conv1 = self.i2f_conv1(x)
        h2f_conv1 = self.h2f_conv1(h_prev)
        f_conv1 = i2f_conv1 + h2f_conv1
        f_conv1 = torch.tanh(f_conv1)
        
        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        
        # Warping
        warped_data = []
        for flow in flows:
            h_warped = self._warp(h_prev, -flow)  # negative flow like original implementation
            warped_data.append(h_warped)
        wrapped_data = torch.cat(warped_data, dim=1)
        
        # GRU gates 
        i2h = self.i2h(x)
        h2h = self.ret(wrapped_data)
        
        i2h_slice = torch.split(i2h, self.hidden_channels, dim=1)
        h2h_slice = torch.split(h2h, self.hidden_channels, dim=1)
        
        reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
        update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
        new_mem = torch.tanh(i2h_slice[2] + reset_gate * h2h_slice[2])
        
        h_new = update_gate * h_prev + (1 - update_gate) * new_mem
        
        if self.zoneout > 0.0 and self.training:
            mask = torch.empty_like(h_new).bernoulli_(1 - self.zoneout)
            h_new = torch.where(mask, h_new, h_prev)

        return h_new

    def _warp(self, x, flow):
        B, C, H, W = x.size()
        
        # Create meshgrid 
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        
        vgrid = grid + flow
        
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        return F.grid_sample(x, vgrid, align_corners=True, padding_mode='border')


class TrajGRU(nn.Module):
    """
    TrajGRU model for spatiotemporal prediction.
    
    A multi-layer TrajGRU model that processes spatiotemporal data using trajectory gated recurrent units.
    Each layer contains a TrajGRUCell that generates flow fields for warping and applies GRU operations.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    hidden_channels : list of int
        List of hidden channels for each TrajGRU layer.
    kernel_size : int or list of int, optional
        Kernel size(s) for TrajGRU cells (default: 3).
        If int, same kernel size used for all layers.
        If list, must match length of hidden_channels.
    L : int or list of int, optional
        Number of flow fields for each TrajGRU layer (default: 5).
        If int, same L used for all layers.
        If list, must match length of hidden_channels.
    seq_len_in : int, optional
        Input sequence length (default: 10).
    seq_len_out : int, optional
        Output sequence length (default: 1).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, L=5, seq_len_in=10, seq_len_out=1):
        super().__init__()
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.n_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        # Support per-layer kernel_size and L
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * self.n_layers
        if isinstance(L, int):
            L = [L] * self.n_layers
        assert len(kernel_size) == self.n_layers, f"kernel_size must have {self.n_layers} elements"
        assert len(L) == self.n_layers, f"L must have {self.n_layers} elements"
        self.kernel_size = kernel_size
        self.L = L
        self.cells = nn.ModuleList([
            TrajGRUCell(
                input_channels if i == 0 else hidden_channels[i-1],
                hidden_channels[i],
                kernel_size=kernel_size[i],
                L=L[i]
            ) for i in range(self.n_layers)
        ])
        self.out_conv = nn.Conv2d(hidden_channels[-1], input_channels, 1)

    def forward(self, x):
        # x: (B, C, D, H, W) where D = seq_len_in
        B, C, D, H, W = x.size()
        h = [torch.zeros(B, cell.hidden_channels, H, W, device=x.device, dtype=x.dtype) for cell in self.cells]
        outputs = []
        # Encode input sequence
        for t in range(self.seq_len_in):
            # For each layer, input is data for first cell, previous hidden for others
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h[i] = cell(x[:, :, t, :, :], h[i])
                else:
                    h[i] = cell(h[i-1], h[i])
        # Predict future sequence
        # Start with the last input frame as the first prediction input
        pred_inp = x[:, :, -1, :, :]
        for t in range(self.seq_len_out):
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h[i] = cell(pred_inp, h[i])
                else:
                    h[i] = cell(h[i-1], h[i])
            pred_inp = self.out_conv(h[-1])  # output of the last cell, to be used as input for next step
            outputs.append(pred_inp.unsqueeze(2))
        return torch.cat(outputs, dim=2)  # (B, C, seq_len_out, H, W) 