import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class TrajGRUCell(nn.Module):
    """
    Trajectory GRU (TrajGRU) recurrent cell with flow-based warping.

    This implementation mirrors the design used in the Hzzone Precipitation-Nowcasting repository:
    - Generates per-step optical-flow-like displacements from inputs and the previous hidden state
    - Warps the previous hidden state using grid_sample
    - Computes GRU gates from warped features + input features

    Parameters
    ----------
    input_channel : int
        Number of channels of the cell input per time step.
    num_filter : int
        Number of hidden channels (feature maps) of the recurrent state.
    L : int, optional (default=5)
        Number of flow fields to generate and concatenate during warping.
    i2h_kernel : int, optional (default=3)
        Kernel size of the input-to-hidden convolution used for GRU gates.
    i2h_stride : int, optional (default=1)
        Stride of the input-to-hidden convolution.
    i2h_pad : int, optional (default=1)
        Padding of the input-to-hidden convolution.
    act_type : callable, optional (default=torch.tanh)
        Activation function used inside the flow generator and for new memory computation.
    """
    def __init__(
        self,
        *,
        input_channel: int,
        num_filter: int,
        L: int = 5,
        i2h_kernel: int = 3,
        i2h_stride: int = 1,
        i2h_pad: int = 1,
        act_type=torch.tanh,
    ):
        super().__init__()
        self.num_filter = num_filter
        self.L = L
        self.act = act_type
        # Gates: reset, update, new (stacked along channel dim)
        self.i2h = nn.Conv2d(
            input_channel, num_filter * 3, kernel_size=i2h_kernel, stride=i2h_stride, padding=i2h_pad
        )
        # Flow generator
        self.i2f_conv1 = nn.Conv2d(input_channel, 32, kernel_size=5, padding=2)
        self.h2f_conv1 = nn.Conv2d(num_filter, 32, kernel_size=5, padding=2)
        self.flows_conv = nn.Conv2d(32, L * 2, kernel_size=5, padding=2)
        # 1x1 conv on concatenated warped hidden maps
        self.ret = nn.Conv2d(num_filter * L, num_filter * 3, kernel_size=1)

    @staticmethod
    def _wrap(input: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp the input feature map with the given flow using grid_sample.
        input: (B, C, H, W)
        flow:  (B, 2, H, W) with (dx, dy) in pixels
        """
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
        return F.grid_sample(input, vgrid, align_corners=False)

    def _flow_generator(self, x_t: Optional[torch.Tensor], h_prev: torch.Tensor) -> List[torch.Tensor]:
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
        *,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the TrajGRU cell over a sequence or for a single step.

        inputs: (S, B, C, H, W) or None for single-step mode
        states: (B, F, H', W') previous hidden state (required when inputs is None)
        seq_len: number of steps to run (S)

        Returns
        -------
        outputs : torch.Tensor
            Stacked outputs with shape (S, B, F, H', W').
        last_state : torch.Tensor
            Final hidden state with shape (B, F, H', W').
        """
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h_full = self.i2h(inputs.reshape(-1, C, H, W))
            i2h_full = i2h_full.view(S, B, 3 * self.num_filter, i2h_full.size(2), i2h_full.size(3))
            i2h_slices = torch.split(i2h_full, self.num_filter, dim=2)
            Hs, Ws = i2h_full.size(3), i2h_full.size(4)
        else:
            i2h_slices = None
            assert states is not None, "states must be provided when inputs is None"
            B, _, Hs, Ws = states.size()

        if states is None:
            device = inputs.device if inputs is not None else "cpu"
            states = torch.zeros((B, self.num_filter, Hs, Ws), dtype=torch.float32, device=device)

        h_prev = states
        outputs = []
        for t in range(seq_len):
            x_t = inputs[t, ...] if inputs is not None else None
            flows = self._flow_generator(x_t, h_prev)
            warped = [self._wrap(h_prev, -flow) for flow in flows]
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


class TrajGRUEncoder(nn.Module):
    """
    TrajGRU Encoder: stage-wise Conv2d downsampling + TrajGRU stacks.

    Each stage applies a spatial convolution (kernel=stride parameters) followed by a
    TrajGRUCell run across the input sequence.

    Parameters
    ----------
    in_channels : int
        Number of input channels per time step.
    hidden_channels : list[int]
        Hidden channels for each encoder stage (also used as the TrajGRU hidden size per stage).
    conv_kernels : list[int]
        Kernel sizes for the stage convolutions.
    conv_strides : list[int]
        Strides for the stage convolutions (control downsampling).
    L : list[int]
        Number of flow fields per TrajGRU stage (same length as hidden_channels).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        conv_kernels: List[int],
        conv_strides: List[int],
        L: List[int],
    ):
        super().__init__()
        assert len(hidden_channels) == len(conv_kernels) == len(conv_strides) == len(L)
        self.blocks = len(hidden_channels)
        self.stages = nn.ModuleList()
        self.rnns = nn.ModuleList()
        curr_in = in_channels
        for i in range(self.blocks):
            k, s, h = conv_kernels[i], conv_strides[i], hidden_channels[i]
            self.stages.append(nn.Conv2d(curr_in, h, kernel_size=k, stride=s, padding=k // 2))
            self.rnns.append(
                TrajGRUCell(
                    input_channel=h,
                    num_filter=h,
                    L=L[i],
                    i2h_kernel=3,
                    i2h_stride=1,
                    i2h_pad=1,
                )
            )
            curr_in = h

    def forward(self, x_sbhwc: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Forward pass.
        x_sbhwc: (S, B, C, H, W)
        Returns: (hidden_states tuple, stage_outputs (S,B,C',H',W'))
        """
        S, B, C, H, W = x_sbhwc.size()
        hidden_states: List[torch.Tensor] = []
        feats = x_sbhwc
        for stage, rnn in zip(self.stages, self.rnns):
            # spatial conv per frame
            feats_reshaped = feats.reshape(-1, feats.size(2), feats.size(3), feats.size(4))
            feats_reshaped = stage(feats_reshaped)
            feats = feats_reshaped.view(S, B, feats_reshaped.size(1), feats_reshaped.size(2), feats_reshaped.size(3))
            outputs, state = rnn(feats, None, seq_len=S)
            feats = outputs
            hidden_states.append(state)
        return tuple(hidden_states), feats


class TrajGRUForecaster(nn.Module):
    """
    TrajGRU Forecaster: stage-wise TrajGRU generation + ConvTranspose2d upsampling.

    Parameters
    ----------
    hidden_channels : list[int]
        Hidden channels per stage (mirrors the encoder in reverse order).
    conv_kernels : list[int]
        Kernel sizes for upsampling stages (mirrors encoder in reverse order).
    conv_strides : list[int]
        Strides for upsampling stages (mirrors encoder in reverse order).
    L : list[int]
        Number of flow fields per TrajGRU stage.
    out_channels : int
        Number of output channels per time step (e.g., input radar channels).
    """
    def __init__(
        self,
        hidden_channels: List[int],
        conv_kernels: List[int],
        conv_strides: List[int],
        L: List[int],
        out_channels: int,
    ):
        super().__init__()
        assert len(hidden_channels) == len(conv_kernels) == len(conv_strides) == len(L)
        self.blocks = len(hidden_channels)
        self.rnns = nn.ModuleList()
        self.stages = nn.ModuleList()
        rev_h = list(reversed(hidden_channels))
        rev_k = list(reversed(conv_kernels))
        rev_s = list(reversed(conv_strides))
        rev_L = list(reversed(L))
        for i in range(self.blocks):
            h = rev_h[i]
            self.rnns.append(
                TrajGRUCell(
                    input_channel=h,
                    num_filter=h,
                    L=rev_L[i],
                    i2h_kernel=3,
                    i2h_stride=1,
                    i2h_pad=1,
                )
            )
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

    def forward(self, hidden_states: Tuple[torch.Tensor, ...], out_len: int, start_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        hidden_states: encoder states tuple, ordered by depth (len == #stages)
        out_len: number of frames to generate
        start_feats: (S,B,C,H,W) outputs from the last encoder stage outputs (used only for shape/carrier)
        Returns: (out_len, B, C, H, W)
        """
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


class TrajGRUEncoderDecoder(nn.Module):
    """
    TrajGRU Encoder–Decoder model (encoder–forecaster) for radar nowcasting.

    - Encoder: Conv2d downsampling per time step followed by TrajGRU stacks
    - Forecaster: TrajGRU generation followed by ConvTranspose2d upsampling per stage

    This mirrors the structure used in the HZ Precipitation-Nowcasting implementation, while
    keeping the CLI-configurable conv kernels/strides and flow-field count per stage.

    Parameters
    ----------
    input_channels : int
        Number of input channels per frame.
    hidden_channels : list[int]
        Hidden channels per stage.
    kernel_size : list[int]
        Unused in this implementation for gates (kept for CLI compatibility).
    L : list[int]
        Flow fields per stage (same length as hidden_channels).
    conv_kernels : list[int]
        Kernel sizes for encoder/forecaster stages.
    conv_strides : list[int]
        Strides for encoder/forecaster stages.
    seq_len_in : int, optional (default=10)
        Number of input frames.
    seq_len_out : int, optional (default=1)
        Number of output frames to generate.
    """
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
        self.encoder = TrajGRUEncoder(input_channels, hidden_channels, conv_kernels, conv_strides, L)
        self.forecaster = TrajGRUForecaster(hidden_channels, conv_kernels, conv_strides, L, out_channels=input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, C, H, W) or (B, C, S, H, W)
        returns: (B, seq_len_out, C, H, W)
        """
        if x.ndim == 5 and x.shape[1] == self.seq_len_in:
            x = x.permute(0, 2, 1, 3, 4)
        # (B,C,S,H,W) -> (S,B,C,H,W)
        x = x.permute(2, 0, 1, 3, 4)
        hidden_states, last_outputs = self.encoder(x)
        y = self.forecaster(hidden_states, self.seq_len_out, last_outputs)
        # back to (B, seq_len_out, C, H, W)
        return y.permute(1, 0, 2, 3, 4)

