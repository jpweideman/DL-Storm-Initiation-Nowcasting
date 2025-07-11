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
        self.conv_warp = nn.Conv2d(hidden_channels * L, hidden_channels, 1)

    def forward(self, x, h_prev):
        flow_input = torch.cat([x, h_prev], dim=1)
        flows = self.conv_flow(flow_input)
        flows = flows.chunk(self.L, dim=1)
        warped_h = []
        for l in range(self.L):
            flow = flows[l]
            h_warped = self._warp(h_prev, flow)
            warped_h.append(h_warped)
        h_cat = torch.cat(warped_h, dim=1)
        h_agg = self.conv_warp(h_cat)
        x_conv = self.conv_x(x)
        h_conv = self.conv_h(h_agg)
        i_x, f_x, o_x = x_conv.chunk(3, dim=1)
        i_h, f_h, o_h = h_conv.chunk(3, dim=1)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        o = torch.tanh(o_x + o_h)
        h_new = f * h_agg + i * o
        if self.zoneout > 0.0 and self.training:
            mask = torch.empty_like(h_new).bernoulli_(1 - self.zoneout)
            h_new = mask * h_new + (1 - mask) * h_prev
        return h_new

    def _warp(self, x, flow):
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=x.device),
            torch.arange(0, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        flow = flow.permute(0, 2, 3, 1)
        new_grid = grid + flow
        new_grid_x = 2.0 * new_grid[..., 0] / max(W - 1, 1) - 1.0
        new_grid_y = 2.0 * new_grid[..., 1] / max(H - 1, 1) - 1.0
        new_grid = torch.stack((new_grid_x, new_grid_y), dim=-1)
        return F.grid_sample(x, new_grid, align_corners=True, padding_mode='border')

class TrajGRUEncoderDecoder(nn.Module):
    """
    Symmetric TrajGRU Encoder-Decoder model (no U-Net, no skip connections).
    - Encoder: Alternates Conv2d (with stride for downsampling) and TrajGRU layers.
    - Decoder: Alternates TrajGRU and ConvTranspose2d (for upsampling) layers, using reversed parameters from encoder.
    - All layer parameters (kernel, stride, channels, L, etc.) are configurable as lists (see below).
    - No skip connections between encoder and decoder.
    - Input: (B, C, S, H, W) or (B, S, C, H, W) (see forward docstring).
    Args:
        input_channels: Number of input channels.
        hidden_channels: List of hidden channels for each encoder/decoder layer.
        kernel_size: List of kernel sizes for each encoder/decoder layer.
        L: List of L values (number of flow fields) for each encoder/decoder layer.
        conv_kernels: List of kernel sizes for encoder Conv2d/decoder ConvTranspose2d.
        conv_strides: List of strides for encoder Conv2d/decoder ConvTranspose2d.
        seq_len_in: Input sequence length.
        seq_len_out: Output sequence length.
    """
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 L,
                 conv_kernels,
                 conv_strides,
                 seq_len_in=10,
                 seq_len_out=1):
        super().__init__()
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        n_layers = len(hidden_channels)
        assert len(kernel_size) == n_layers, "kernel_size should match hidden_channels"
        assert len(L) == n_layers, "L should match hidden_channels"
        assert len(conv_kernels) == n_layers, "conv_kernels should match hidden_channels"
        assert len(conv_strides) == n_layers, "conv_strides should match hidden_channels"
        # Encoder
        self.encoder_convs = nn.ModuleList()
        self.encoder_trajgru = nn.ModuleList()
        encoder_in_channels = [input_channels] + hidden_channels[:-1]
        for i in range(n_layers):
            self.encoder_convs.append(
                nn.Conv2d(encoder_in_channels[i], hidden_channels[i], conv_kernels[i], stride=conv_strides[i], padding=conv_kernels[i]//2)
            )
            self.encoder_trajgru.append(
                TrajGRUCell(hidden_channels[i], hidden_channels[i], kernel_size=kernel_size[i], L=L[i])
            )
        # Decoder (reverse order)
        self.decoder_trajgru = nn.ModuleList()
        self.decoder_deconvs = nn.ModuleList()
        rev_hidden = list(reversed(hidden_channels))
        rev_kernel = list(reversed(kernel_size))
        rev_L = list(reversed(L))
        rev_conv_kernels = list(reversed(conv_kernels))
        rev_conv_strides = list(reversed(conv_strides))
        decoder_in_channels = rev_hidden.copy()
        decoder_out_channels = rev_hidden.copy()
        # For deconv, output channels should match the next decoder's input channels, except for the last layer
        deconv_out_channels = rev_hidden[1:] + [rev_hidden[-1]]
        for i in range(n_layers):
            self.decoder_trajgru.append(
                TrajGRUCell(decoder_in_channels[i], decoder_out_channels[i], kernel_size=rev_kernel[i], L=rev_L[i])
            )
            output_padding = rev_conv_strides[i] - 1
            self.decoder_deconvs.append(
                nn.ConvTranspose2d(decoder_out_channels[i], deconv_out_channels[i], rev_conv_kernels[i], stride=rev_conv_strides[i], padding=rev_conv_kernels[i]//2, output_padding=output_padding)
            )
        # Final output conv
        self.out_conv = nn.Conv2d(rev_hidden[-1], input_channels, 1)

    def forward(self, x):
        """
        x: (B, S, C, H, W) or (B, C, S, H, W)
        Returns: (B, seq_len_out, C, H, W)
        """
        if x.ndim == 5 and x.shape[1] == self.seq_len_in:
            # (B, S, C, H, W) -> (B, C, S, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        B, C, S, H, W = x.shape
        feats = x
        # Encoder
        for i, (conv, trajgru) in enumerate(zip(self.encoder_convs, self.encoder_trajgru)):
            # Collapse time for conv
            feats = feats.permute(0, 2, 1, 3, 4)  # (B, S, C, H, W)
            B, S, C_curr, H_curr, W_curr = feats.shape
            feats = feats.reshape(B * S, C_curr, H_curr, W_curr)
            feats = conv(feats)
            # Reshape back to (B, S, C_new, H_new, W_new)
            C_new, H_new, W_new = feats.shape[1], feats.shape[2], feats.shape[3]
            feats = feats.reshape(B, S, C_new, H_new, W_new)
            feats = feats.permute(0, 2, 1, 3, 4)  # (B, C_new, S, H_new, W_new)
            # TrajGRU over time
            h = torch.zeros(B, C_new, H_new, W_new, device=feats.device, dtype=feats.dtype)
            for t in range(S):
                h = trajgru(feats[:, :, t, :, :], h)
            feats = h.unsqueeze(2)  # (B, C_new, 1, H_new, W_new)
        feats = feats.squeeze(2)  # (B, C_new, H_new, W_new)
        # Decoder/Forecaster
        outputs = []
        h_list = [None] * len(self.decoder_trajgru)
        for t in range(self.seq_len_out):
            x_dec = feats
            for i in range(len(self.decoder_trajgru)):
                # Ensure h_list[i] has correct shape
                if h_list[i] is None or h_list[i].shape[1:] != (self.decoder_trajgru[i].hidden_channels, x_dec.shape[2], x_dec.shape[3]):
                    h_list[i] = torch.zeros(x_dec.shape[0], self.decoder_trajgru[i].hidden_channels, x_dec.shape[2], x_dec.shape[3], device=x_dec.device, dtype=x_dec.dtype)
                h_list[i] = self.decoder_trajgru[i](x_dec, h_list[i])
                x_dec = h_list[i]
                x_dec = self.decoder_deconvs[i](x_dec)
            out = self.out_conv(x_dec)
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, seq_len_out, C, H, W) 