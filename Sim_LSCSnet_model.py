import torch
import torch.nn as nn
from UNet1D import UNet1D


class SampMapGenerator(nn.Module):
    def __init__(self, input_dim=6950, output_dim=6950):
        super().__init__()
        self.L = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softplus()
        )

    def forward(self, sp_in):  # x (B,6950)
        # feed spectral magnitude so the generator knows where energy lives
        s = self.mlp(sp_in.view(sp_in.size(0), -1))
        samp = ((s - s.min(dim=-1, keepdim=True)[0]) /
                (s.max(dim=-1, keepdim=True)[0] - s.min(dim=-1, keepdim=True)[0] + 1e-8))
        return samp  # (B,6950) float mask


# ---------------------------------------------------------------
# 2.  One iteration block (works in IF domain)
# ---------------------------------------------------------------
class LSCSNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet1D(2, 2)  # conv on (B,2,N)

    def forward(self, if_in, samp_map, y0):
        spc = torch.fft.fft(if_in, dim=-1)  # complex (B,6950)
        sp2ch = torch.stack((spc.real, spc.imag), dim=1)  # (B,2,6950)
        sp2ch_h = self.unet(sp2ch)
        spc_h = torch.complex(sp2ch_h[:, 0, :], sp2ch_h[:, 1, :])
        if_hat = torch.fft.ifft(spc_h, dim=-1)  # back to IF
        y = if_hat * (1 - samp_map) + y0 * samp_map
        return y, if_hat


# ---------------------------------------------------------------
# 3.  Full LPTNet+ in IF domain
# ---------------------------------------------------------------
class LPTNetIF(nn.Module):
    def __init__(self, iters=1, M=600):
        super().__init__()
        self.iters = nn.ModuleList([LSCSNet1D() for _ in range(iters)])
        self.mapgen = SampMapGenerator()
        self.M = M
        self.uni_samp_map = None
        self.highest_certain_indices = None

    # -----------------------------------------------------------
    def forward(self, sp_in):
        """ sp_in : (B,6950) spectrogram """

        if self.uni_samp_map is not None:
            samp = self.mapgen(sp_in)
            with torch.no_grad():
                samp[:, self.highest_certain_indices] = self.uni_samp_map[:, self.highest_certain_indices]
        else:
            samp = self.mapgen(sp_in)

        if_in = torch.fft.ifft(sp_in, dim=-1)
        y0 = if_in * samp  # masked IF
        y = y0
        for blk in self.iters:
            y, _ = blk(y, samp, y0)

        sp_bin = torch.fft.fft(y, dim=-1).abs()
        return sp_bin, samp

    def get_binary_samp_map(self, samp_map):
        # Ensure samp_map has exactly M ones
        binary_samp_map = torch.zeros_like(samp_map)
        topk = torch.topk(samp_map, self.M, dim=-1)
        binary_samp_map.scatter_(-1, topk.indices, 1)
        return binary_samp_map
