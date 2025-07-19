import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.inner_models.TexFilter_Real import TexFilter  # Assuming TexFilter is in the same directory

from numpy.polynomial.chebyshev import Chebyshev

def cheb_torch(data, degree, rtn_data=False, device='cpu'):
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
        raise ValueError('The input data should be 1D or 2D.')

    data = data.to(device)

    tvals = np.linspace(-1, 1, T)
    cheb_polys = np.array([Chebyshev.basis(i)(tvals) for i in range(degree)])  # [degree, T]
    cheb_polys = torch.from_numpy(cheb_polys).float().to(device)

    # Compute Chebyshev coefficients using projection
    coeffs = torch.mm(cheb_polys, data) / T * 2  # similar to orthogonal projection
    coeffs = coeffs.transpose(0, 1)  # [B*C, degree]

    if rtn_data:
        recon = torch.mm(coeffs, cheb_polys)
        recon = recon.reshape(B, -1, T).permute(0, 2, 1)  # [B, T, C]

        if ndim == 2:
            recon = recon.squeeze(0)
        return coeffs, recon
    else:
        return coeffs


def chebyshev_encode(input_seq, degree=64):
    B, T, C = input_seq.shape
    input_seq_flat = input_seq.reshape(B * C, T).T  # [T, B*C]
    device = input_seq.device

    coeffs = cheb_torch(input_seq_flat, degree - 1, rtn_data=False, device=device)
    coeffs = coeffs.reshape(B, C, degree).to(device)  # [B, C, D]

    return coeffs

def chebyshev_decode(coeffs, seq_len):
    B, C, D = coeffs.shape
    coeffs_flat = coeffs.reshape(B * C, D)

    tvals = np.linspace(-1, 1, seq_len)
    cheb_polys = np.array([Chebyshev.basis(i)(tvals) for i in range(D)])  # [D, T]
    cheb_polys = torch.from_numpy(cheb_polys).float().to(coeffs.device)

    recon = torch.mm(coeffs_flat, cheb_polys).reshape(B, C, seq_len).permute(0, 2, 1)  # [B, T, C]
    return recon




class Model(nn.Module):
    # Hybrid FITS: RIN + Learnable Frequency Filtering (TexFilter) + Interpolation
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
            self.freq_upsampler = nn.Linear((2), int((self.seq_len + self.pred_len) // 2 + 1))

        # NEW: Learnable frequency filter
        #self.texfilter = TexFilter(embed_size=self.channels,
        #                            use_gelu=False,
        #                            use_skip=False,
        #                            use_layernorm=False,
        #                            hard_threshold=False,
        #                            use_window=False)
        self.texfilter =    TexFilter(
                        embed_size=self.channels,
                        use_gelu=True,             # or use_swish=True for smoother nonlinearity
                        use_skip=True,             # ✅ Preserve original signal paths
                        use_layernorm=True,        # ✅ Stabilize across frequency bins
                        hard_threshold=False,      # ❌ Avoid hard cutting off weak signals
                        use_window=False,          # ❌ Avoid muting boundary info
                        sparsity_threshold=0.0     # ✅ Retain all weak signal components
                    )
        """
        🔝 Most Impactful Modifications (Ranked)
        Rank	Feature	                Expected Impact	    Notes
        1️⃣	use_skip (Skip connection)	⭐⭐⭐⭐	      Strongly stabilizes learning and gradient flow; allows feature reuse; widely helpful in deep and shallow nets alike.
        2️⃣	use_layernorm	            ⭐⭐⭐⭐	      Helps convergence and generalization, especially with long or dynamic sequences. Particularly useful for frequency-domain data which may have diverse scales.
        3️⃣	use_gelu / use_swish	    ⭐⭐⭐	           Both provide smoother nonlinearities than ReLU, improving expressiveness without hurting gradient flow. Swish is theoretically slightly better, but more expensive.
        4️⃣	hard_threshold	            ⭐⭐	             Forces exact sparsity. Can be helpful if you're modeling truly sparse frequency signals (e.g., anomalies), but risks killing useful low-energy signals.
        5️⃣	use_window	                ⭐⭐	             Applies a Hanning window in frequency. Helps with spectral leakage, but may discard useful boundary info in short sequences. Often useful in clean forecasting, but might be redundant in learned models.
        """
    def forward(self, x):
        # ---------------------
        # 1. RevIN (manual normalization)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # ---------------------
        # 2. Legendre encode (real-valued projection)
        # Input x shape: [B, seq_len, C]
        # Output specx shape: [B, C, degree]
        specx = chebyshev_encode(x, degree=2)  # tune degree if needed

        # ---------------------
        # 3. Apply TexFilter (learnable attention on Legendre coeffs)
        # specx shape: [B, C, degree]
        # Need to permute to [B, degree, C] to match TexFilter input shape [B, F, C]
        specx = specx.permute(0, 2, 1)  # now [B, degree, C]

        # Apply filter
        specx = specx * self.texfilter(specx)  # elementwise real multiply

        # ---------------------
        # 4. Frequency interpolation (upsample along degree dimension)
        if self.individual:
            specxy_ = torch.zeros(
                x.size(0),
                int((self.seq_len + self.pred_len) // 2 + 1),
                x.size(2),
                dtype=specx.dtype,
                device=specx.device
            )
            for i in range(self.channels):
                # Upsample each channel independently
                specxy_[:, :, i] = self.freq_upsampler[i](specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            specxy_ = self.freq_upsampler(specx.permute(0, 2, 1)).permute(0, 2, 1)

        # ---------------------
        # 5. Pad if needed (likely still needed)
        full_spec = torch.zeros(
            [specxy_.size(0), int((self.seq_len + self.pred_len) // 2 + 1), specxy_.size(2)],
            dtype=specxy_.dtype,
            device=specxy_.device
        )
        full_spec[:, :specxy_.size(1), :] = specxy_

        # ---------------------
        # 6. Legendre decode (reconstruct real-valued time domain signal)
        low_xy = chebyshev_decode(full_spec.permute(0, 2, 1), seq_len=self.seq_len )
        # legendre_decode expects [B, degree, C] input, permuted from [B, F, C]
        # Output shape: [B, out_len, C]

        low_xy = low_xy * self.length_ratio

        # ---------------------
        # 7. Reverse RevIN normalization
        xy = (low_xy * torch.sqrt(x_var)) + x_mean

        return xy, low_xy * torch.sqrt(x_var)
