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
        # input is C,H,W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell for spatiotemporal processing.
    
    A single ConvLSTM cell that processes spatial features with temporal gating.
    Combines spatial convolutions with LSTM-style gating mechanisms.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    kernel_size : int, optional
        Kernel size for convolutions (default: 3).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Input to hidden
        self.i2h = nn.Conv2d(input_channels, hidden_channels * 4, kernel_size, padding=padding)
        # Hidden to hidden
        self.h2h = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        # x: (B, C, H, W), h_prev: (B, H, H, W), c_prev: (B, H, H, W)
        
        # Gates computation
        i2h = self.i2h(x)
        h2h = self.h2h(h_prev)
        
        # Split gates
        gates = i2h + h2h
        gates = torch.split(gates, self.hidden_channels, dim=1)
        
        input_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1])
        cell_gate = torch.tanh(gates[2])
        output_gate = torch.sigmoid(gates[3])
        
        # LSTM update
        c_new = forget_gate * c_prev + input_gate * cell_gate
        h_new = output_gate * torch.tanh(c_new)
        
        return h_new, c_new


class UNetConvLSTM(nn.Module):
    """
    U-Net + ConvLSTM model for spatiotemporal forecasting.
    
    Combines U-Net encoder-decoder architecture with ConvLSTM bottleneck for spatiotemporal forecasting.
    The encoder processes spatial features, the ConvLSTM bottleneck handles temporal dynamics,
    and the decoder reconstructs the spatial features.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    base_ch : int, optional
        Number of channels in the first encoder layer (default: 32).
    hidden_dims : int or tuple/list of int, optional
        Number of hidden channels in the ConvLSTM bottleneck (default: 64).
        If a tuple/list, multiple ConvLSTM layers are stacked.
        **Each entry corresponds to a single ConvLSTMCell (not a double block).**
        The number of entries determines the number of ConvLSTM layers in the bottleneck.
    seq_len : int, optional
        Input sequence length (number of time steps) (default: 10).
    kernel : int, optional
        Convolution kernel size for all convolutions (must be odd) (default: 3).
    """
    def __init__(self, in_ch, out_ch, base_ch=32, hidden_dims=64, seq_len=10, kernel=3):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        
        # Encoder 
        self.inc = DoubleConv(in_ch, base_ch, kernel)
        self.down1 = Down(base_ch, base_ch*2, kernel)
        self.down2 = Down(base_ch*2, base_ch*4, kernel)
        
        # Bottleneck ConvLSTM (support multiple layers if hidden_dims is tuple/list)
        if isinstance(hidden_dims, (tuple, list)):
            self.convlstm_layers = nn.ModuleList()
            in_dim = base_ch*4
            for hid in hidden_dims:
                self.convlstm_layers.append(ConvLSTMCell(in_dim, hid, kernel))
                in_dim = hid
            self.convlstm_out_dim = hidden_dims[-1]
        else:
            self.convlstm_layers = None
            self.convlstm_cell = ConvLSTMCell(base_ch*4, hidden_dims, kernel)
            self.convlstm_out_dim = hidden_dims
        
        # Decoder
        self.up1 = Up(self.convlstm_out_dim, base_ch*4, base_ch*2, kernel)  
        self.up2 = Up(base_ch*2, base_ch, base_ch, kernel)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        device = x.device
        
        # Process each time step through the encoder and accumulate features
        encoded_features = []
        for t in range(S):
            xt = x[:, t]  # (B, C, H, W)
            x1 = self.inc(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            encoded_features.append((x1, x2, x3))  # Store all features for skip connections
        
        encoded_stack = torch.stack([feat[2] for feat in encoded_features], dim=1)  # (B, S, base_ch*4, H//4, W//4)
        
        # Process through ConvLSTM bottleneck
        if self.convlstm_layers is not None:
            # Multiple ConvLSTM layers
            h_list = []
            c_list = []
            for cell in self.convlstm_layers:
                h = torch.zeros(B, cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
                c = torch.zeros(B, cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
                h_list.append(h)
                c_list.append(c)
            
            # Process temporal sequence through ConvLSTM layers
            for t in range(S):
                xt = encoded_stack[:, t]  # (B, base_ch*4, H//4, W//4)
                for i, cell in enumerate(self.convlstm_layers):
                    h_list[i], c_list[i] = cell(xt, h_list[i], c_list[i])
                    xt = h_list[i]  
            bottleneck_out = h_list[-1]  
        else:
            # Single ConvLSTM layer
            h = torch.zeros(B, self.convlstm_cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
            c = torch.zeros(B, self.convlstm_cell.hidden_channels, H//4, W//4, device=device, dtype=x.dtype)
            
            # Process temporal sequence through single ConvLSTM
            for t in range(S):
                xt = encoded_stack[:, t]  # (B, base_ch*4, H//4, W//4)
                h, c = self.convlstm_cell(xt, h, c)
            bottleneck_out = h
        
        # Decoder 
        x = self.up1(bottleneck_out, encoded_features[-1][2])  # Use last time step's x3 (base_ch*4)
        x = self.up2(x, encoded_features[-1][0])  # Use last time step's x1 (base_ch)
        x = self.outc(x)
        return x 