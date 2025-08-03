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
        
        # Input to hidden (i2h) - reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(input_channels, hidden_channels * 3, kernel_size, padding=padding)
        
        # Hidden to hidden (h2h) - after warping, 1x1 conv
        self.ret = nn.Conv2d(hidden_channels * L, hidden_channels * 3, 1)
        
        # Flow generation - separate layers 
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
        
        # Scale grid to [-1,1] 
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        return F.grid_sample(x, vgrid, align_corners=True, padding_mode='border')

class TrajGRUEncoderDecoder(nn.Module):
    """
    Symmetric TrajGRU Encoder-Decoder model for spatiotemporal prediction.
    
    - Encoder: Alternates Conv2d (with stride for downsampling) and TrajGRU layers.
    - Decoder: Alternates TrajGRU and ConvTranspose2d (for upsampling) layers, using reversed parameters from encoder.
    - All layer parameters (kernel, stride, channels, L, etc.) are configurable as lists.
    - No skip connections between encoder and decoder.
    - Input: (B, C, S, H, W) or (B, S, C, H, W) (see forward docstring).

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    hidden_channels : list of int
        List of hidden channels for each encoder/decoder layer.
    kernel_size : list of int
        List of kernel sizes for each encoder/decoder layer.
    L : list of int
        List of L values (number of flow fields) for each encoder/decoder layer.
    conv_kernels : list of int
        List of kernel sizes for encoder Conv2d/decoder ConvTranspose2d.
    conv_strides : list of int
        List of strides for encoder Conv2d/decoder ConvTranspose2d.
    seq_len_in : int, optional
        Input sequence length (default: 10).
    seq_len_out : int, optional
        Output sequence length (default: 1).
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