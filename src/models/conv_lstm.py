import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell.

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
        assert kernel % 2 == 1, "Kernel size must be odd for ConvLSTM to preserve spatial dimensions!"
        p = kernel // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=p)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i,f,o,g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g       = torch.tanh(g)
        c_next  = f * c + i * g
        h_next  = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        h = torch.zeros(B, self.hid_ch, H, W, device=device)
        return h, h.clone()
    
    
class ConvLSTM(nn.Module):
    """
    ConvLSTM model.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    hidden_dims : list of int, optional
        List of hidden dimensions for each ConvLSTM layer (default: (64, 64)).
    kernel : int, optional
        Kernel size for all ConvLSTM cells (default: 3).
    """
    def __init__(self, in_ch, hidden_dims=(64, 64), kernel=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx, h in enumerate(hidden_dims):
            i_ch = in_ch if idx == 0 else hidden_dims[idx-1]
            self.layers.append(ConvLSTMCell(i_ch, h, kernel))
        self.to_out = nn.Conv2d(hidden_dims[-1], in_ch, 1)

    def forward(self, x):
        B, S, _, H, W = x.shape
        device = x.device
        h_list, c_list = [], []
        for cell in self.layers:
            h, c = cell.init_hidden(B, H, W, device)
            h_list.append(h)
            c_list.append(c)
        for t in range(S):
            xt = x[:, t]
            for i, cell in enumerate(self.layers):
                h_list[i], c_list[i] = cell(xt, h_list[i], c_list[i])
                xt = h_list[i]
        return self.to_out(xt) 