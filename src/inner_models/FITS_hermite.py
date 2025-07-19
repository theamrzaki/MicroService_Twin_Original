import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.inner_models.TexFilter_Real import TexFilter  # Assuming TexFilter is in the same directory
from numpy.polynomial.hermite import Hermite as H

def hermite_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)  # [T, B * C]
    else:
        raise ValueError('Input must be 2D or 3D tensor.')

    data = data.to(device)
    tvals = np.linspace(-5, 5, T)
    hermite_polys = np.array([H.basis(i)(tvals) for i in range(degree)])  # [degree, T]
    hermite_polys = torch.from_numpy(hermite_polys).float().to(device)

    coeffs = torch.mm(hermite_polys, data) / T  # projection
    coeffs = coeffs.transpose(0, 1)  # [B*C, degree]

    if rtn_data:
        recon = torch.mm(coeffs, hermite_polys)  # [B*C, T]
        recon = recon.reshape(B, -1, T).permute(0, 2, 1)
        if ndim == 2:
            recon = recon.squeeze(0)
        return coeffs, recon
    else:
        return coeffs


def hermite_encode(input_seq, degree=64):
    B, T, C = input_seq.shape
    input_seq_flat = input_seq.reshape(B * C, T).T  # [T, B*C]
    device = input_seq.device
    coeffs = hermite_torch(input_seq_flat, degree - 1, rtn_data=False, device=device)
    coeffs = coeffs.reshape(B, C, degree).to(device)
    return coeffs


def hermite_decode(coeffs, seq_len):
    B, C, D = coeffs.shape
    coeffs_flat = coeffs.reshape(B * C, D)

    tvals = np.linspace(-5, 5, seq_len)
    hermite_polys = np.array([H.basis(i)(tvals) for i in range(D)])  # [D, T]
    hermite_polys = torch.from_numpy(hermite_polys).float().to(coeffs.device)

    recon = torch.mm(coeffs_flat, hermite_polys).reshape(B, C, seq_len).permute(0, 2, 1)
    return recon


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear((self.seq_len // 2 + 1), int((self.seq_len + self.pred_len) // 2 + 1)).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear(2, int((self.seq_len + self.pred_len) // 2 + 1))

        self.texfilter = TexFilter(
            embed_size=self.channels,
            use_gelu=True,
            use_skip=True,
            use_layernorm=True,
            hard_threshold=False,
            use_window=False,
            sparsity_threshold=0.0
        )

    def forward(self, x):
        # Step 1: RevIN normalization
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # Step 2: Encode with Hermite
        specx = hermite_encode(x, degree=2)  # [B, C, degree]

        # Step 3: Apply TexFilter
        specx = specx.permute(0, 2, 1)  # [B, degree, C]
        specx = specx * self.texfilter(specx)

        # Step 4: Frequency interpolation
        if self.individual:
            specxy_ = torch.zeros(
                x.size(0),
                int((self.seq_len + self.pred_len) // 2 + 1),
                x.size(2),
                dtype=specx.dtype,
                device=specx.device
            )
            for i in range(self.channels):
                specxy_[:, :, i] = self.freq_upsampler[i](specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            specxy_ = self.freq_upsampler(specx.permute(0, 2, 1)).permute(0, 2, 1)

        # Step 5: Pad if needed
        full_spec = torch.zeros(
            [specxy_.size(0), int((self.seq_len + self.pred_len) // 2 + 1), specxy_.size(2)],
            dtype=specxy_.dtype,
            device=specxy_.device
        )
        full_spec[:, :specxy_.size(1), :] = specxy_

        # Step 6: Decode from Hermite
        low_xy = hermite_decode(full_spec.permute(0, 2, 1), seq_len=self.seq_len)
        low_xy = low_xy * self.length_ratio

        # Step 7: De-normalize
        xy = (low_xy * torch.sqrt(x_var)) + x_mean

        return xy, low_xy * torch.sqrt(x_var)
