import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.traj_gru_enc_dec import TrajGRUCell 


class DoubleTrajGRUBlock(nn.Module):
    """
    Double TrajGRU block.
    
    This block processes temporal dynamics through two consecutive TrajGRU cells.
    
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
        
        # Initialize hidden states if not provided
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
        # Apply max pooling first
        x_pooled = self.mpconv[0](x)
        # Then apply Double TrajGRU
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
    U-Net + Double TrajGRU model for spatiotemporal prediction.
    
    
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    base_ch : int, optional
        Base number of channels for U-Net encoder/decoder (default: 32).
    trajgru_hid : int or tuple/list of int, optional
        Number of hidden channels for each TrajGRU layer. The length determines 
        the number of encoder/decoder stages.
        Example: [32, 64, 128] means 3 stages with 32, 64, and 128 channels respectively.
        If None, defaults to [32, 64, 128] (3 stages).
    seq_len : int, optional
        Input sequence length (number of time steps) (default: 10).
    kernel : int, optional
        Convolution kernel size for all TrajGRU convolutions (must be odd) (default: 3).
    L : int or tuple/list of int, optional
        Number of flow fields for each TrajGRU layer. Must have same length as trajgru_hid.
        Example: [5, 7, 13] for different L values per stage.
        If None, defaults to [5] * len(trajgru_hid).
    """
    def __init__(self, in_ch, out_ch, base_ch=32, trajgru_hid=None, seq_len=10, kernel=3, L=None):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        
        # Set default values if None
        if trajgru_hid is None:
            trajgru_hid = [base_ch, base_ch*2, base_ch*4]
        if L is None:
            L = [5] * len(trajgru_hid)
            
        # Ensure L_values matches trajgru_hid length
        if len(L) != len(trajgru_hid):
            if len(L) == 1:
                L = L * len(trajgru_hid)
            else:
                raise ValueError(f"L must have 1 or {len(trajgru_hid)} values, got {len(L)}")
        
        self.trajgru_hid = trajgru_hid
        self.L_values = L
        self.n_stages = len(trajgru_hid)
        
        # Input Double TrajGRU
        self.inc = DoubleTrajGRUBlock(in_ch, trajgru_hid[0], kernel, L[0], seq_len)
        
        # Encoder path: downsampling stages
        self.down_stages = nn.ModuleList()
        for i in range(self.n_stages - 1):
            self.down_stages.append(
                Down(trajgru_hid[i], trajgru_hid[i+1], kernel, L[i+1], seq_len)
            )
        
        # Decoder path:upsampling with skip connections
        self.up_stages = nn.ModuleList()
        for i in range(self.n_stages - 1):
            up_in_ch = trajgru_hid[-(i+1)]  # Current stage channels
            skip_ch = trajgru_hid[-(i+2)]    # Skip connection channels
            up_out_ch = trajgru_hid[-(i+2)]  # Output channels (same as skip)
            self.up_stages.append(
                Up(up_in_ch, skip_ch, up_out_ch, kernel, L[-(i+2)], seq_len)
            )
        
        # Final output convolution
        self.outc = nn.Conv2d(trajgru_hid[0], out_ch, 1)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        device = x.device
        
        # Initialize hidden states for each stage
        hidden_states = []
        for i, hid_dim in enumerate(self.trajgru_hid):
            # Calculate spatial dimensions for each stage
            if i == 0:
                # Input stage: full resolution
                h, w = H, W
            else:
                # Downsampled stages: halved each time
                h, w = H // (2**i), W // (2**i)
            
            # Each stage has 2 TrajGRU cells, so we need 2 hidden states per stage
            hidden_states.append([
                torch.zeros(B, hid_dim, h, w, device=device, dtype=x.dtype),  
                torch.zeros(B, hid_dim, h, w, device=device, dtype=x.dtype)   
            ])
        
        # Process each time step through the encoder path
        encoded_features = []
        
        for t in range(S):
            xt = x[:, t]  # (B, C, H, W)
            
            # Input stage 
            h_inc = self.inc(xt, hidden_states[0][0], hidden_states[0][1])
            hidden_states[0][0] = h_inc  
            hidden_states[0][1] = h_inc  # Final output from second TrajGRU
            encoded_features.append(h_inc)
            
            # Downsampling stages
            current_input = h_inc
            for i, down_stage in enumerate(self.down_stages):
                h_down = down_stage(current_input, hidden_states[i+1][0], hidden_states[i+1][1])
                hidden_states[i+1][0] = h_down
                hidden_states[i+1][1] = h_down
                current_input = h_down
                encoded_features.append(h_down)
        
        # decoder path 
        x = encoded_features[-1]  # Deepest encoded feature
        
        # Upsampling stages with skip connections
        for i, up_stage in enumerate(self.up_stages):
            skip_idx = -(i+2) 
            skip_feature = encoded_features[skip_idx]
            
            x = up_stage(x, skip_feature, hidden_states[skip_idx][0], hidden_states[skip_idx][1])
        
        x = self.outc(x)  # (B, out_ch, H, W)
        return x
