import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.traj_gru_enc_dec import TrajGRUCell 


class DoubleTrajGRUBlock(nn.Module):
    """
    Double TrajGRU block.
    
    This block processes spatiotemporal dynamics through two consecutive TrajGRU cells.
    
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel : int, optional
        Kernel size for TrajGRU convolutions (default: 3).
    L : int, optional
        Number of flow fields for TrajGRU (default: 5).
    seq_len : int, optional
        Sequence length for temporal processing (default: 10).
    """
    def __init__(self, in_ch, out_ch, kernel=3, L=5, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        
        self.trajgru1 = TrajGRUCell(
            input_channel=in_ch,
            num_filter=out_ch,
            L=L,
            i2h_kernel=kernel,
            i2h_stride=1,
            i2h_pad=kernel // 2,
        )
        
        self.trajgru2 = TrajGRUCell(
            input_channel=out_ch,
            num_filter=out_ch,
            L=L,
            i2h_kernel=kernel,
            i2h_stride=1,
            i2h_pad=kernel // 2,
        )
        
    def forward(self, x, h_prev1=None, h_prev2=None):
        # x: (B, C, H, W) 
        B, C, H, W = x.shape
        device = x.device
        
        if h_prev1 is None:
            h_prev1 = torch.zeros(B, self.trajgru1.num_filter, H, W, device=device, dtype=x.dtype)
        if h_prev2 is None:
            h_prev2 = torch.zeros(B, self.trajgru2.num_filter, H, W, device=device, dtype=x.dtype)
        
        # Process through first TrajGRU
        outputs1, state1 = self.trajgru1(x.unsqueeze(0), h_prev1, seq_len=1)
        h1 = state1
        
        # Process through second TrajGRU 
        outputs2, state2 = self.trajgru2(h1.unsqueeze(0), h_prev2, seq_len=1)
        h2 = state2
        
        return h2


class Down(nn.Module):
    """
    2D Downscaling block with Double TrajGRU.
    
    Applies max pooling followed by Double TrajGRU block for downsampling.
    
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel : int, optional
        Kernel size for TrajGRU convolutions (default: 3).
    L : int, optional
        Number of flow fields for TrajGRU (default: 5).
    seq_len : int, optional
        Sequence length for temporal processing (default: 10).
    """
    def __init__(self, in_ch, out_ch, kernel=3, L=5, seq_len=10):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleTrajGRUBlock(in_ch, out_ch, kernel, L, seq_len)
        )

    def forward(self, x, h_prev1=None, h_prev2=None):
        # Max pooling first
        x_pooled = self.mpconv[0](x)
        # Double TrajGRU
        return self.mpconv[1](x_pooled, h_prev1, h_prev2)


class Up(nn.Module):
    """
    2D Upscaling block with Double TrajGRU.
    
    Applies transposed convolution for upsampling followed by Double TrajGRU block.
    
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    skip_ch : int
        Number of channels from skip connection.
    out_ch : int
        Number of output channels.
    kernel : int, optional
        Kernel size for TrajGRU convolutions (default: 3).
    L : int, optional
        Number of flow fields for TrajGRU (default: 5).
    seq_len : int, optional
        Sequence length for temporal processing (default: 10).
    """
    def __init__(self, in_ch, skip_ch, out_ch, kernel=3, L=5, seq_len=10):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.trajgru = DoubleTrajGRUBlock(in_ch // 2 + skip_ch, out_ch, kernel, L, seq_len)

    def forward(self, x1, x2, h_prev1=None, h_prev2=None):
        x1 = self.up(x1)
        # input is C,H,W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.trajgru(x, h_prev1, h_prev2)


class UNetTrajGRU(nn.Module):
    """
    U-Net + Double TrajGRU model for spatiotemporal forecasting.
    
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    base_ch : int, optional
        Base number of channels for U-Net encoder/decoder (default: 32).
    bottleneck_dims : tuple/list, optional
        Sequence of widths for the bottleneck TrajGRU layers (e.g., (64, 32)).
        Each entry corresponds to a DoubleTrajGRUBlock (i.e., two TrajGRU cells per entry).
        The number of entries determines the depth of the bottleneck.
        If None, defaults to (base_ch*4,) for a single bottleneck stage.
    seq_len : int, optional
        Input sequence length (number of time steps) (default: 10).
    kernel : int, optional
        Convolution kernel size for all TrajGRU convolutions (must be odd) (default: 3).
    L : int, optional
        Number of flow fields for all TrajGRU layers across encoder, bottleneck, and decoder (default: 5).
    """
    def __init__(self, in_ch, out_ch, base_ch=32, bottleneck_dims=None, seq_len=10, kernel=3, L=5):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        
        # Set default bottleneck if None
        if bottleneck_dims is None:
            bottleneck_dims = (base_ch*4,) 
        # Ensure bottleneck_dims is a tuple/list
        if not isinstance(bottleneck_dims, (tuple, list)):
            bottleneck_dims = (bottleneck_dims,)
        
        self.bottleneck_dims = bottleneck_dims
        self.L = L
        self.n_bottleneck_stages = len(bottleneck_dims)
        
        # encoder 
        self.inc = DoubleTrajGRUBlock(in_ch, base_ch, kernel, L, seq_len)
        self.down1 = Down(base_ch, base_ch*2, kernel, L, seq_len)
        self.down2 = Down(base_ch*2, base_ch*4, kernel, L, seq_len)
        
        #  bottleneck 
        bottleneck_layers = []
        in_channels = base_ch*4
        for width in bottleneck_dims:
            bottleneck_layers.append(DoubleTrajGRUBlock(in_channels, width, kernel, L, seq_len))
            in_channels = width
        self.bottleneck = nn.ModuleList(bottleneck_layers)
        
        # decoder with skip connections
        self.up1 = Up(in_channels, base_ch*2, base_ch*2, kernel, L, seq_len)
        self.up2 = Up(base_ch*2, base_ch, base_ch, kernel, L, seq_len)
        
        # Final output convolution
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        device = x.device
        
        hidden_states = {}
        
        # Encoder hidden states
        hidden_states['inc'] = [
            torch.zeros(B, self.inc.trajgru1.num_filter, H, W, device=device, dtype=x.dtype),
            torch.zeros(B, self.inc.trajgru2.num_filter, H, W, device=device, dtype=x.dtype)
        ]
        hidden_states['down1'] = [
            torch.zeros(B, self.down1.mpconv[1].trajgru1.num_filter, H//2, W//2, device=device, dtype=x.dtype),
            torch.zeros(B, self.down1.mpconv[1].trajgru2.num_filter, H//2, W//2, device=device, dtype=x.dtype)
        ]
        hidden_states['down2'] = [
            torch.zeros(B, self.down2.mpconv[1].trajgru1.num_filter, H//4, W//4, device=device, dtype=x.dtype),
            torch.zeros(B, self.down2.mpconv[1].trajgru2.num_filter, H//4, W//4, device=device, dtype=x.dtype)
        ]
        
        # Bottleneck hidden states
        bottleneck_hidden = []
        for i, bottleneck_layer in enumerate(self.bottleneck):
            h, w = H // 4, W // 4  
            bottleneck_hidden.append([
                torch.zeros(B, bottleneck_layer.trajgru1.num_filter, h, w, device=device, dtype=x.dtype),
                torch.zeros(B, bottleneck_layer.trajgru2.num_filter, h, w, device=device, dtype=x.dtype)
            ])
        
        # Process each time step through the encoder path
        encoded_features = []
        
        for t in range(S):
            xt = x[:, t]  # (B, C, H, W)
            
            # Input stage 
            h_inc = self.inc(xt, hidden_states['inc'][0], hidden_states['inc'][1])
            hidden_states['inc'][0] = h_inc  
            hidden_states['inc'][1] = h_inc
            encoded_features.append(h_inc)
            
            # Downsampling stages
            h_down1 = self.down1(h_inc, hidden_states['down1'][0], hidden_states['down1'][1])
            hidden_states['down1'][0] = h_down1
            hidden_states['down1'][1] = h_down1
            encoded_features.append(h_down1)
            
            h_down2 = self.down2(h_down1, hidden_states['down2'][0], hidden_states['down2'][1])
            hidden_states['down2'][0] = h_down2
            hidden_states['down2'][1] = h_down2
            encoded_features.append(h_down2)
        
        # Bottleneck processing
        x = encoded_features[-1]  # Deepest encoded feature 
        
        for i, bottleneck_layer in enumerate(self.bottleneck):
            x = bottleneck_layer(x, bottleneck_hidden[i][0], bottleneck_hidden[i][1])
            bottleneck_hidden[i][0] = x
            bottleneck_hidden[i][1] = x
        
        # Decoder path 
        x = self.up1(x, encoded_features[-2], hidden_states['down1'][0], hidden_states['down1'][1]) 
        x = self.up2(x, encoded_features[-3], hidden_states['inc'][0], hidden_states['inc'][1])
        x = self.outc(x)  # (B, out_ch, H, W)
        return x
