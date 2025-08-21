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
            h_warped = self._warp(h_prev, flow)
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
        
        return F.grid_sample(x, vgrid, align_corners=False, padding_mode='border', mode='bilinear')

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
        # 1x1 convs to fuse skip connections in decoder
        self.skip_fuse_convs = nn.ModuleList()
        encoder_in_channels = [input_channels] + hidden_channels[:-1]
        for i in range(n_layers):
            self.encoder_convs.append(
                nn.Conv2d(encoder_in_channels[i], hidden_channels[i], conv_kernels[i], stride=conv_strides[i], padding=conv_kernels[i]//2)
            )
            self.encoder_trajgru.append(
                TrajGRUCell(hidden_channels[i], hidden_channels[i], kernel_size=kernel_size[i], L=L[i])
            )
            # placeholder; actual fuse convs created after reversing sizes
        # Decoder (reverse order)
        self.decoder_trajgru = nn.ModuleList()
        self.decoder_deconvs = nn.ModuleList()
        rev_hidden = list(reversed(hidden_channels))
        rev_kernel = list(reversed(kernel_size))
        rev_L = list(reversed(L))
        rev_conv_kernels = list(reversed(conv_kernels))
        rev_conv_strides = list(reversed(conv_strides))
        # build skip fuse convs now that we know reversed shapes
        for i in range(len(hidden_channels)):
            in_ch = rev_hidden[i] * 2  # concat decoder feat with corresponding encoder feat
            out_ch = rev_hidden[i]
            self.skip_fuse_convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
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
            # (B, S, C, H, W) to (B, C, S, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        B, C, S, H, W = x.shape
        feats = x
        enc_feats = []
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
            # store encoder feature for skip connection (before next downsample)
            enc_feats.append(h)
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
                # fuse skip connection from corresponding encoder stage (reverse order)
                if i < len(enc_feats):
                    skip = enc_feats[-(i+1)]
                    # if spatial sizes mismatch due to stride, interpolate skip
                    if skip.shape[2:] != x_dec.shape[2:]:
                        skip = F.interpolate(skip, size=x_dec.shape[2:], mode='bilinear', align_corners=False)
                    x_dec = torch.cat([x_dec, skip], dim=1)
                    x_dec = self.skip_fuse_convs[i](x_dec)
            out = self.out_conv(x_dec)
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, seq_len_out, C, H, W)



# --- HZ-style TrajGRU, Encoder, and Forecaster adapted to this codebase ---
from typing import Optional, List, Tuple


class HZTrajGRUCell(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_filter: int,
        L: int = 5,
        i2h_kernel: int = 3,
        i2h_stride: int = 1,
        i2h_pad: int = 1,
        h2h_kernel: int = 5,
        act_type=torch.tanh,
    ):
        super().__init__()
        self.num_filter = num_filter
        self.L = L
        self.act = act_type
        # gates
        self.i2h = nn.Conv2d(
            input_channel,
            num_filter * 3,
            kernel_size=i2h_kernel,
            stride=i2h_stride,
            padding=i2h_pad,
        )
        # flow gen
        self.i2f_conv1 = nn.Conv2d(input_channel, 32, kernel_size=5, padding=2)
        self.h2f_conv1 = nn.Conv2d(num_filter, 32, kernel_size=5, padding=2)
        self.flows_conv = nn.Conv2d(32, L * 2, kernel_size=5, padding=2)
        # 1x1 for warped hidden concat
        self.ret = nn.Conv2d(num_filter * L, num_filter * 3, kernel_size=1)

    @staticmethod
    def wrap(input: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input.size()
        device = input.device
        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        return F.grid_sample(input, vgrid)

    def _flow_generator(
        self, x_t: Optional[torch.Tensor], h_prev: torch.Tensor
    ) -> List[torch.Tensor]:
        i2f = self.i2f_conv1(x_t) if x_t is not None else None
        h2f = self.h2f_conv1(h_prev)
        f = h2f if i2f is None else (i2f + h2f)
        f = self.act(f)
        flows = self.flows_conv(f)
        return list(torch.split(flows, 2, dim=1))

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (S,B,C,H,W) or None; states: (B,F,H',W') or None
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h_full = self.i2h(inputs.reshape(-1, C, H, W))
            i2h_full = i2h_full.view(
                S, B, 3 * self.num_filter, i2h_full.size(2), i2h_full.size(3)
            )
            i2h_slices = torch.split(i2h_full, self.num_filter, dim=2)
            Hs, Ws = i2h_full.size(3), i2h_full.size(4)
        else:
            i2h_slices = None
            assert states is not None, "states must be provided when inputs is None"
            B, _, Hs, Ws = states.size()

        if states is None:
            device = inputs.device if inputs is not None else "cpu"
            states = torch.zeros(
                (B, self.num_filter, Hs, Ws), dtype=torch.float32, device=device
            )

        h_prev = states
        outputs = []
        for t in range(seq_len):
            x_t = inputs[t, ...] if inputs is not None else None
            flows = self._flow_generator(x_t, h_prev)
            warped = []
            for flow in flows:
                warped.append(self.wrap(h_prev, -flow))
            wrapped_data = torch.cat(warped, dim=1)
            h2h = self.ret(wrapped_data)
            h2h_slices = torch.split(h2h, self.num_filter, dim=1)
            if i2h_slices is not None:
                reset_gate = torch.sigmoid(i2h_slices[0][t, ...] + h2h_slices[0])
                update_gate = torch.sigmoid(i2h_slices[1][t, ...] + h2h_slices[1])
                new_mem = self.act(i2h_slices[2][t, ...] + reset_gate * h2h_slices[2])
            else:
                reset_gate = torch.sigmoid(h2h_slices[0])
                update_gate = torch.sigmoid(h2h_slices[1])
                new_mem = self.act(reset_gate * h2h_slices[2])
            h_prev = update_gate * h_prev + (1 - update_gate) * new_mem
            outputs.append(h_prev)
        return torch.stack(outputs), h_prev


class HZEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        conv_kernels: List[int],
        conv_strides: List[int],
        L: List[int],
    ):
        super().__init__()
        assert (
            len(hidden_channels)
            == len(conv_kernels)
            == len(conv_strides)
            == len(L)
        )
        self.blocks = len(hidden_channels)
        self.stages = nn.ModuleList()
        self.rnns = nn.ModuleList()
        curr_in = in_channels
        for i in range(self.blocks):
            k, s, h = conv_kernels[i], conv_strides[i], hidden_channels[i]
            self.stages.append(
                nn.Conv2d(curr_in, h, kernel_size=k, stride=s, padding=k // 2)
            )
            self.rnns.append(HZTrajGRUCell(input_channel=h, num_filter=h, L=L[i]))
            curr_in = h

    def forward(self, x_sbhwc: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        # x: (S,B,C,H,W)
        S, B, C, H, W = x_sbhwc.size()
        hidden_states: List[torch.Tensor] = []
        feats = x_sbhwc
        for stage, rnn in zip(self.stages, self.rnns):
            # apply conv per frame
            feats_reshaped = feats.reshape(-1, feats.size(2), feats.size(3), feats.size(4))
            feats_reshaped = stage(feats_reshaped)
            feats = feats_reshaped.view(
                S, B, feats_reshaped.size(1), feats_reshaped.size(2), feats_reshaped.size(3)
            )
            outputs, state = rnn(feats, None, seq_len=S)
            feats = outputs
            hidden_states.append(state)
        return tuple(hidden_states), feats


class HZForecaster(nn.Module):
    def __init__(
        self,
        hidden_channels: List[int],
        conv_kernels: List[int],
        conv_strides: List[int],
        L: List[int],
        out_channels: int,
    ):
        super().__init__()
        assert (
            len(hidden_channels)
            == len(conv_kernels)
            == len(conv_strides)
            == len(L)
        )
        self.blocks = len(hidden_channels)
        self.rnns = nn.ModuleList()
        self.stages = nn.ModuleList()
        rev_h = list(reversed(hidden_channels))
        rev_k = list(reversed(conv_kernels))
        rev_s = list(reversed(conv_strides))
        rev_L = list(reversed(L))
        for i in range(self.blocks):
            h = rev_h[i]
            self.rnns.append(HZTrajGRUCell(input_channel=h, num_filter=h, L=rev_L[i]))
            stride = rev_s[i]
            k = rev_k[i]
            out_ch = rev_h[i + 1] if i + 1 < len(rev_h) else out_channels
            self.stages.append(
                nn.ConvTranspose2d(
                    h,
                    out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2,
                    output_padding=max(stride - 1, 0),
                )
            )

    def forward(
        self, hidden_states: Tuple[torch.Tensor, ...], out_len: int, start_feats: torch.Tensor
    ) -> torch.Tensor:
        # start_feats: (S,B,C,H,W) from last encoder stage outputs
        S, B = start_feats.size(0), start_feats.size(1)
        x: Optional[torch.Tensor] = None
        for i in range(self.blocks):
            rnn = self.rnns[i]
            state = hidden_states[-(i + 1)]
            outputs, _ = rnn(x, state, seq_len=out_len)
            o = outputs.reshape(-1, outputs.size(2), outputs.size(3), outputs.size(4))
            o = self.stages[i](o)
            x = o.view(out_len, B, o.size(1), o.size(2), o.size(3))
        return x


class HZTrajGRUEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        L,
        conv_kernels,
        conv_strides,
        seq_len_in=10,
        seq_len_out=1,
    ):
        super().__init__()
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.encoder = HZEncoder(
            input_channels, hidden_channels, conv_kernels, conv_strides, L
        )
        self.forecaster = HZForecaster(
            hidden_channels, conv_kernels, conv_strides, L, out_channels=input_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,S,H,W) or (B,S,C,H,W)
        if x.ndim == 5 and x.shape[1] == self.seq_len_in:
            x = x.permute(0, 2, 1, 3, 4)
        # to (S,B,C,H,W)
        x = x.permute(2, 0, 1, 3, 4)
        hidden_states, last_outputs = self.encoder(x)
        y = self.forecaster(hidden_states, self.seq_len_out, last_outputs)
        # back to (B, seq_len_out, C, H, W)
        return y.permute(1, 0, 2, 3, 4)


# Alias new implementation under the original exported name used by trainers
TrajGRUEncoderDecoder = HZTrajGRUEncoderDecoder

