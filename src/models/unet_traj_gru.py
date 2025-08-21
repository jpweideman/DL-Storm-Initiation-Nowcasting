import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.traj_gru_enc_dec import TrajGRUCell as EncDecTrajGRUCell


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
        # input is C,H,W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TrajGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, L=5, zoneout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = EncDecTrajGRUCell(
            input_channel=input_channels,
            num_filter=hidden_channels,
            L=L,
            i2h_kernel=kernel_size,
            i2h_stride=1,
            i2h_pad=kernel_size // 2,
        )

    def forward(self, x, h_prev):
        # Use single-step interface via seq_len=1
        outputs, state = self.cell(x.unsqueeze(0), h_prev, seq_len=1)
        return state


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
        
        # Process each time step through the encoder and accumulate features
        # Store all encoder features for skip connections
        encoded_features = []
        for t in range(S):
            xt = x[:, t]  # (B, C, H, W)
            x1 = self.inc(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            encoded_features.append((x1, x2, x3))  # Store all features for skip connections
        
        # Stack encoded features along time dimension
        encoded_stack = torch.stack([feat[2] for feat in encoded_features], dim=1)  # (B, S, base_ch*4, H//4, W//4)
        
        # Process through TrajGRU bottleneck (HZ-style cell)
        if self.trajgru_layers is not None:
            # Multiple TrajGRU layers
            h_list = []
            for cell in self.trajgru_layers:
                h = torch.zeros(B, cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
                h_list.append(h)
            
            # Process temporal sequence through TrajGRU layers
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
        
        # Decoder with skip connections
        x = self.up1(bottleneck_out, encoded_features[-1][2])  # Use last time step's x3 (base_ch*4)
        x = self.up2(x, encoded_features[-1][0])  # Use last time step's x1 (base_ch)
        x = self.outc(x)
        return x 