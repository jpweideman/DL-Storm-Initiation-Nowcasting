import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.traj_gru_enc_dec import TrajGRUCell



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
                input_channel=(input_channels if i == 0 else hidden_channels[i-1]),
                num_filter=hidden_channels[i],
                L=L[i],
                i2h_kernel=kernel_size[i],
                i2h_stride=1,
                i2h_pad=kernel_size[i] // 2,
            ) for i in range(self.n_layers)
        ])
        self.out_conv = nn.Conv2d(hidden_channels[-1], input_channels, 1)

    def forward(self, x):
        # x: (B, C, D, H, W) where D = seq_len_in
        B, C, D, H, W = x.size()
        # Prepare sequence (S,B,C,H,W)
        seq = x.permute(2, 0, 1, 3, 4)
        # Encode through stacked HZTrajGRU cells
        states = []
        for i, cell in enumerate(self.cells):
            outputs_i, state_i = cell(seq if i == 0 else seq_i, None, seq_len=self.seq_len_in)
            seq_i = outputs_i
            states.append(state_i)
        # Autoregressive forecasting using stored states
        outputs = []
        for t in range(self.seq_len_out):
            inp_t = None
            for i, cell in enumerate(self.cells):
                out_step, new_state = cell(inp_t, states[i], seq_len=1)
                states[i] = new_state
                inp_t = out_step  # (1,B,C,H,W)
            y_t = self.out_conv(inp_t[0])
            outputs.append(y_t.unsqueeze(2))
        return torch.cat(outputs, dim=2)  # (B, C, seq_len_out, H, W)