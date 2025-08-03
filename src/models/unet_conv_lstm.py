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


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell for spatiotemporal processing.
    
    A single ConvLSTM cell that applies LSTM operations with convolutional gates.
    Processes spatial data while maintaining temporal memory through LSTM mechanisms.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    hid_ch : int
        Number of hidden channels.
    kernel : int, optional
        Kernel size for convolutions (default: 3).
    """
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        p = kernel // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=p)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        h = torch.zeros(B, self.hid_ch, H, W, device=device)
        return h, h.clone()


class UNetConvLSTM(nn.Module):
    """
    U-Net + ConvLSTM model for spatiotemporal prediction.
    
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
    lstm_hid : int or tuple/list of int, optional
        Number of hidden channels in the ConvLSTM bottleneck (default: 64).
        If a tuple/list, multiple ConvLSTM layers are stacked.
        **Each entry corresponds to a single ConvLSTMCell (not a double block).**
        The number of entries determines the number of ConvLSTM layers in the bottleneck.
    seq_len : int, optional
        Input sequence length (number of time steps) (default: 10).
    kernel : int, optional
        Convolution kernel size for all convolutions (must be odd) (default: 3).
    """
    def __init__(self, in_ch, out_ch, base_ch=32, lstm_hid=64, seq_len=10, kernel=3):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        # Encoder
        self.inc = DoubleConv(in_ch, base_ch, kernel)
        self.down1 = Down(base_ch, base_ch*2, kernel)
        self.down2 = Down(base_ch*2, base_ch*4, kernel)
        # Bottleneck ConvLSTM (support multiple layers if lstm_hid is tuple/list)
        if isinstance(lstm_hid, (tuple, list)):
            self.lstm_layers = nn.ModuleList()
            in_dim = base_ch*4
            for hid in lstm_hid:
                self.lstm_layers.append(ConvLSTMCell(in_dim, hid, kernel))
                in_dim = hid
            self.lstm_out_dim = lstm_hid[-1]
        else:
            self.lstm_layers = None
            self.lstm_cell = ConvLSTMCell(base_ch*4, lstm_hid, kernel)
            self.lstm_out_dim = lstm_hid
        # Decoder
        self.up1 = Up(self.lstm_out_dim, base_ch*4, base_ch*2, kernel)  
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
        
        # Process through ConvLSTM bottleneck
        if self.lstm_layers is not None:
            # Multiple ConvLSTM layers
            h_list = []
            c_list = []
            for cell in self.lstm_layers:
                h, c = cell.init_hidden(B, H//4, W//4, device)
                h_list.append(h)
                c_list.append(c)
            
            for t in range(S):
                xt = encoded_stack[:, t]  # (B, base_ch*4, H//4, W//4)
                for i, cell in enumerate(self.lstm_layers):
                    h_list[i], c_list[i] = cell(xt, h_list[i], c_list[i])
                    xt = h_list[i]  # Feed output of current layer to next layer
            bottleneck_out = h_list[-1]  # Use output from last layer
        else:
            # Single ConvLSTM layer
            h, c = self.lstm_cell.init_hidden(B, H//4, W//4, device)
            for t in range(S):
                xt = encoded_stack[:, t]  # (B, base_ch*4, H//4, W//4)
                h, c = self.lstm_cell(xt, h, c)
            bottleneck_out = h
        
        # Decoder
        x = self.up1(bottleneck_out, encoded_features[-1][2])  # Use last encoded features (x3) for skip connection
        x = self.up2(x, encoded_features[0][0])  # Use first encoded features (x1) for skip connection
        x = self.outc(x)
        return x 