import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    Double 3D Convolution block.

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
            nn.Conv3d(in_ch, out_ch, kernel, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel, padding=p),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    """
    3D Downscaling block.

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
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch, kernel)
        )
    def forward(self, x):
        return self.mpconv(x)


class Up3D(nn.Module):
    """
    3D Upscaling block.

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
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv3D(in_ch // 2 + skip_ch, out_ch, kernel)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffD = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3DCNN(nn.Module):
    """
    U-Net 3D CNN model.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    base_ch : int, optional
        Base number of channels for U-Net encoder/decoder (default: 32).
    bottleneck_dims : tuple/list, optional
        Sequence of widths for the 3D CNN bottleneck layers (e.g., (32, 64, 32)).
        Each entry corresponds to a DoubleConv3D block (i.e., two Conv3D layers per entry).
        The number of entries determines the depth of the bottleneck in terms of blocks, but the total number of Conv3D layers is 2 × len(bottleneck_dims).
    kernel : int, optional
        Kernel size (default: 3).
    seq_len_out : int, optional
        Output sequence length (default: 1).
    """
    def __init__(self, in_ch, out_ch, base_ch=32, bottleneck_dims=(64,), kernel=3, seq_len_out=1):
        super().__init__()
        self.seq_len_out = seq_len_out
        self.inc = DoubleConv3D(in_ch, base_ch, kernel)
        self.down1 = Down3D(base_ch, base_ch*2, kernel)
        self.down2 = Down3D(base_ch*2, base_ch*4, kernel)
        bottleneck_layers = []
        in_channels = base_ch*4
        for width in bottleneck_dims:
            bottleneck_layers.append(DoubleConv3D(in_channels, width, kernel))
            in_channels = width
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        self.up1 = Up3D(in_channels, base_ch*2, base_ch*2, kernel)
        self.up2 = Up3D(base_ch*2, base_ch, base_ch, kernel)
        self.outc = nn.Conv3d(base_ch, out_ch, 1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x_bottleneck = self.bottleneck(x3)
        x = self.up1(x_bottleneck, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        x = x[:, :, -self.seq_len_out:, :, :]
        return x 