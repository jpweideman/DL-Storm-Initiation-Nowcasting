import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, L=5, zoneout=0.0):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.L = L  # number of allowed flow fields
        self.zoneout = zoneout
        padding = kernel_size // 2
        self.conv_x = nn.Conv2d(input_channels, hidden_channels * 3, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size, padding=padding)
        self.conv_flow = nn.Conv2d(input_channels + hidden_channels, 2 * L, kernel_size, padding=padding)
        # Single 1x1 conv for concatenated warped states (paper style)
        self.conv_warp = nn.Conv2d(hidden_channels * L, hidden_channels, 1)

    def forward(self, x, h_prev):
        # Compute flow fields
        flow_input = torch.cat([x, h_prev], dim=1)
        flows = self.conv_flow(flow_input)  # (B, 2*L, H, W)
        flows = flows.chunk(self.L, dim=1)  # L x (B, 2, H, W)
        warped_h = []
        for l in range(self.L):
            flow = flows[l]
            h_warped = self._warp(h_prev, flow)
            warped_h.append(h_warped)
        # Concatenate along channel dimension (B, hidden*L, H, W)
        h_cat = torch.cat(warped_h, dim=1)
        h_agg = self.conv_warp(h_cat)  # (B, hidden, H, W)

        x_conv = self.conv_x(x)
        h_conv = self.conv_h(h_agg)
        i_x, f_x, o_x = x_conv.chunk(3, dim=1)
        i_h, f_h, o_h = h_conv.chunk(3, dim=1)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        o = torch.tanh(o_x + o_h)
        h_new = f * h_agg + i * o

        # Optional: Zoneout (paper style)
        if self.zoneout > 0.0 and self.training:
            mask = torch.empty_like(h_new).bernoulli_(1 - self.zoneout)
            h_new = mask * h_new + (1 - mask) * h_prev

        return h_new

    def _warp(self, x, flow):
        # x: (B, C, H, W), flow: (B, 2, H, W)
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=x.device),
            torch.arange(0, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        flow = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        new_grid = grid + flow
        new_grid_x = 2.0 * new_grid[..., 0] / max(W - 1, 1) - 1.0
        new_grid_y = 2.0 * new_grid[..., 1] / max(H - 1, 1) - 1.0
        new_grid = torch.stack((new_grid_x, new_grid_y), dim=-1)
        return F.grid_sample(x, new_grid, align_corners=True, padding_mode='border')


class TrajGRU(nn.Module):
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