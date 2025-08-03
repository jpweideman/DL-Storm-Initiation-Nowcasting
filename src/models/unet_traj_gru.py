import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double 2D Convolution block.
    
    Applies two consecutive 2D convolutions with ReLU activation.
    Used as a building block in the U-Net architecture.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel : int, optional
        Kernel size for convolutions (default: 3).
    """
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        p = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=p),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """
    2D Downscaling block.
    
    Applies max pooling followed by double convolution for downsampling.
    Used in the encoder path of the U-Net architecture.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel : int, optional
        Kernel size for convolutions (default: 3).
    """
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, kernel)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    """
    2D Upscaling block.
    
    Applies transposed convolution for upsampling followed by double convolution.
    Used in the decoder path of the U-Net architecture.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    skip_ch : int
        Number of channels from skip connection.
    out_ch : int
        Number of output channels.
    kernel : int, optional
        Kernel size for convolutions (default: 3).
    """
    def __init__(self, in_ch, skip_ch, out_ch, kernel=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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


class UNetTrajGRU(nn.Module):
    """
    U-Net + TrajGRU model for spatiotemporal prediction.
    
    Combines U-Net encoder-decoder architecture with TrajGRU bottleneck for spatiotemporal forecasting.
    The encoder processes spatial features, the TrajGRU bottleneck handles temporal dynamics,
    and the decoder reconstructs the spatial features.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    base_ch : int, optional
        Number of channels in the first encoder layer (default: 32).
    trajgru_hid : int or tuple/list of int, optional
        Number of hidden channels in the TrajGRU bottleneck (default: 64).
        If a tuple/list, multiple TrajGRU layers are stacked.
        **Each entry corresponds to a single TrajGRUCell (not a double block).**
        The number of entries determines the number of TrajGRU layers in the bottleneck.
    seq_len : int, optional
        Input sequence length (number of time steps) (default: 10).
    kernel : int, optional
        Convolution kernel size for all convolutions (must be odd) (default: 3).
    L : int or tuple/list of int, optional
        Number of flow fields for each TrajGRU layer (default: 5).
        If a tuple/list, must match length of trajgru_hid.
    """
    def __init__(self, in_ch, out_ch, base_ch=32, trajgru_hid=64, seq_len=10, kernel=3, L=5):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        # Encoder
        self.inc = DoubleConv(in_ch, base_ch, kernel)
        self.down1 = Down(base_ch, base_ch*2, kernel)
        self.down2 = Down(base_ch*2, base_ch*4, kernel)
        # Bottleneck TrajGRU (support multiple layers if trajgru_hid is tuple/list)
        if isinstance(trajgru_hid, (tuple, list)):
            self.trajgru_layers = nn.ModuleList()
            in_dim = base_ch*4
            # Support per-layer L values
            if isinstance(L, int):
                L = [L] * len(trajgru_hid)
            assert len(L) == len(trajgru_hid), f"L must have {len(trajgru_hid)} elements"
            for i, hid in enumerate(trajgru_hid):
                self.trajgru_layers.append(TrajGRUCell(in_dim, hid, kernel, L[i]))
                in_dim = hid
            self.trajgru_out_dim = trajgru_hid[-1]
        else:
            self.trajgru_layers = None
            self.trajgru_cell = TrajGRUCell(base_ch*4, trajgru_hid, kernel, L)
            self.trajgru_out_dim = trajgru_hid
        # Decoder
        self.up1 = Up(self.trajgru_out_dim, base_ch*4, base_ch*2, kernel)  
        self.up2 = Up(base_ch*2, base_ch, base_ch, kernel)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        device = x.device
        
        # Process each time step through the encoder
        encoded_features = []
        for t in range(S):
            xt = x[:, t]  # (B, C, H, W)
            x1 = self.inc(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            encoded_features.append((x1, x2, x3))  # Store all intermediate features
        
        # Stack encoded features along time dimension
        encoded_stack = torch.stack([feat[2] for feat in encoded_features], dim=1)  # (B, S, base_ch*4, H//4, W//4)
        
        # Process through TrajGRU bottleneck
        if self.trajgru_layers is not None:
            # Multiple TrajGRU layers
            h_list = []
            for cell in self.trajgru_layers:
                h = torch.zeros(B, cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
                h_list.append(h)
            
            for t in range(S):
                xt = encoded_stack[:, t]  # (B, base_ch*4, H//4, W//4)
                for i, cell in enumerate(self.trajgru_layers):
                    h_list[i] = cell(xt, h_list[i])
                    xt = h_list[i]  # Feed output of current layer to next layer
            bottleneck_out = h_list[-1]  # Use output from last layer
        else:
            # Single TrajGRU layer
            h = torch.zeros(B, self.trajgru_cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
            for t in range(S):
                xt = encoded_stack[:, t]  # (B, base_ch*4, H//4, W//4)
                h = self.trajgru_cell(xt, h)
            bottleneck_out = h
        
        # Decoder
        x = self.up1(bottleneck_out, encoded_features[-1][2])  # Use last encoded features (x3) for skip connection
        x = self.up2(x, encoded_features[0][0])  # Use first encoded features (x1) for skip connection
        x = self.outc(x)
        return x 