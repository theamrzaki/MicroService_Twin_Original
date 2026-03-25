import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.inner_models.TexFilter_Real import TexFilter  # Assuming TexFilter is in the same directory

from numpy.polynomial import Legendre as L

def leg_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    # Make sure data is on the right device
    data = data.to(device)

    tvals = np.linspace(-1, 1, T)
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(degree)])
    legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]

    # Compute coefficients
    coeffs_candidate = torch.mm(legendre_polys, data) / T * 2
    coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(degree)]).to(device)
    coeffs = coeffs.transpose(0, 1)  # shape: [B * D, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def legendre_encode(input_seq, degree=64):
    B, T, C = input_seq.shape
    input_seq_flat = input_seq.reshape(B * C, T).T  # shape: [T, B*C]

    device = input_seq.device
    coeffs = leg_torch(input_seq_flat, degree - 1, rtn_data=False, device=device)
    coeffs = coeffs.reshape(B, C, degree).to(device)

    return coeffs

def legendre_decode(coeffs, seq_len):
    B, C, D = coeffs.shape
    coeffs_flat = coeffs.reshape(B * C, D)

    # Generate Legendre basis on correct device
    tvals = np.linspace(-1, 1, seq_len)
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(D)])  # [D, T]
    legendre_polys = torch.from_numpy(legendre_polys).float().to(coeffs.device)

    reconstructed = torch.mm(coeffs_flat, legendre_polys).reshape(B, C, seq_len).permute(0, 2, 1)
    return reconstructed




class Model_old(nn.Module):
    # Hybrid FITS: RIN + Learnable Frequency Filtering (TexFilter) + Interpolation
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.degree = 10
        self.channels = configs.enc_in
        self.optimize_precompute_legendre = getattr(configs, 'optimize_precompute_legendre', True)

        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear((self.seq_len // 2 + 1), int((self.seq_len + self.pred_len) // 2 + 1)).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            #self.freq_upsampler = nn.Linear((10), int((self.seq_len + self.pred_len) // 2 + 1))
            self.freq_upsampler = nn.Linear(self.degree, self.degree)

        if self.optimize_precompute_legendre:
            t = np.linspace(-1, 1, self.seq_len)
            basis = np.array([legendre(i)(t) for i in range(self.degree)])
            self.register_buffer('leg_basis', torch.tensor(basis, dtype=torch.float32))

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
        if self.optimize_precompute_legendre:
            # FAST: Batch Matrix Multiply [B, C, seq_len] @ [seq_len, degree]
            # This is significantly faster than standard 'legendre_encode' functions
            #specx = torch.matmul(x.transpose(1, 2), self.basis.t())
            specx = torch.einsum("bct,kt->bck", x.transpose(1, 2), self.basis)
        else:
            specx = legendre_encode(x, degree=self.degree)

        # ---------------------
        # 3. Apply TexFilter (learnable attention on Legendre coeffs)
        # specx shape: [B, C, degree]
        # Need to permute to [B, degree, C] to match TexFilter input shape [B, F, C]
        specx = specx.permute(0, 2, 1)  # now [B, degree, C]

        # Apply filter
        #specx = specx * self.texfilter(specx)  # elementwise real multiply
        specx.mul_(self.texfilter(specx))

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
            #specxy_ = self.freq_upsampler(specx.permute(0, 2, 1)).permute(0, 2, 1)
            specxy_ = self.freq_upsampler(specx.transpose(1, 2)).transpose(1, 2)
        # ---------------------
        # 5. Pad if needed (likely still needed)
        #full_spec = torch.zeros(
        #    [specxy_.size(0), int((self.seq_len + self.pred_len) // 2 + 1), specxy_.size(2)],
        #    dtype=specxy_.dtype,
        #    device=specxy_.device
        #)
        #full_spec[:, :specxy_.size(1), :] = specxy_

        # ---------------------
        # 6. Legendre decode (reconstruct real-valued time domain signal)
        if self.optimize_precompute_legendre:
            """
            In your forward method, you are using transpose and permute frequently. On a CPU, this often forces data movement.

T           he Change: Keep the Legendre basis precomputed, but consolidate the permute and matmul operations to ensure the CPU can use its cache effectively.
            """
            # Use .contiguous() before reshape/permute to help the CPU
            x_trans = x.transpose(1, 2).contiguous() 
            specx = torch.matmul(x_trans, self.basis.t())
            
            # TexFilter expects [B, degree, C]
            # Instead of permute(0, 2, 1), use transpose for better stride preservation
            specx = specx.transpose(1, 2) 
            specx = specx * self.texfilter(specx)

            # Upsampling & Decoding
            specxy_temp = self.freq_upsampler(specx.transpose(1, 2)) # [B, C, degree]
            #low_xy = torch.matmul(specxy_temp, self.basis).transpose(1, 2)
            low_xy = torch.einsum("bck,kt->bct", specxy_temp, self.basis)
        else:
            low_xy = legendre_decode(specxy_.transpose(1, 2), seq_len=self.seq_len)
        # legendre_decode expects [B, degree, C] input, permuted from [B, F, C]
        # Output shape: [B, out_len, C]

        low_xy = low_xy * self.length_ratio

        # ---------------------
        # 7. Reverse RevIN normalization
        xy_with_sqrt = low_xy * torch.sqrt(x_var)  # Scale back to original variance
        xy = xy_with_sqrt + x_mean

        return xy, xy_with_sqrt




class Model_old(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.degree = getattr(configs, "degree", 5)

        self.optimize_precompute_legendre = getattr(
            configs, "optimize_precompute_legendre", True
        )
        self.filter_used = getattr(configs, "filter_used", "TexFilter")
        self.basis_type = getattr(configs, "basis_type", "legendre")
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        
        # -------------------------------------------------
        # Frequency Upsampler
        # -------------------------------------------------

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.degree, self.degree)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear(self.degree, self.degree)
            #self.freq_upsampler = nn.Conv1d(
            #    in_channels=self.channels,
            #    out_channels=self.channels,
            #    kernel_size=1
            #)
        # -------------------------------------------------
        # Precompute Legendre Basis
        # -------------------------------------------------

        if self.optimize_precompute_legendre:
            t = np.linspace(-1, 1, self.seq_len)

            if self.basis_type == "legendre":
                from scipy.special import legendre 
                basis = np.array([legendre(i)(t) for i in range(self.degree)])

            elif self.basis_type == "chebyshev":
                from numpy.polynomial.chebyshev import chebvander
                basis = chebvander(t, self.degree - 1).T

            elif self.basis_type == "fourier":
                t = np.linspace(0, 1, self.seq_len)
                basis = self._build_fourier_basis(t)
                assert self.degree % 2 == 0, "Fourier basis requires even degree."

            elif self.basis_type == "hermite":
                from numpy.polynomial.hermite import hermvander
                basis = hermvander(t, self.degree - 1).T

            elif self.basis_type == "laguerre":
                from numpy.polynomial.laguerre import lagvander
                basis = lagvander(t, self.degree - 1).T

            basis = torch.tensor(basis, dtype=torch.float32)

            self.register_buffer("basis", basis)
            self.register_buffer("basis_T", basis.t().contiguous())

        # -------------------------------------------------
        # Learnable Frequency Filter
        # -------------------------------------------------
        if self.filter_used == "TexFilter":
            self.texfilter = TexFilter(
                embed_size=self.channels,
                use_gelu=True,
                use_skip=True,
                use_layernorm=True,
                hard_threshold=False,
                use_window=False,
                sparsity_threshold=0.0
            )
        elif self.filter_used == "LPF":
            self.texfilter = nn.Identity()
        elif self.filter_used == "nofilter":
            self.texfilter = nn.Identity()

    # =====================================================
    # Basis Construction (Fourier), using real sines and cosines
    # =====================================================
    def _build_fourier_basis(self, t):
        basis = []
        basis.append(np.ones_like(t))
        for k in range(1, self.degree // 2):

            basis.append(np.sin(2 * np.pi * k * t))
            basis.append(np.cos(2 * np.pi * k * t))

        return np.array(basis)

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        # -------------------------------------------------
        # 1. RevIN
        # -------------------------------------------------

        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5

        x = (x - x_mean) / x_std

        # -------------------------------------------------
        # 2. Legendre Encode
        # -------------------------------------------------

        if self.optimize_precompute_legendre:

            # [B,seq,C] -> [B,C,seq]
            x_t = x.transpose(1, 2).contiguous()

            # [B,C,seq] @ [seq,degree]
            #spec = torch.matmul(x_t, self.basis.t())
            spec = torch.matmul(x_t, self.basis_T)
        else:

            spec = legendre_encode(x, degree=self.degree)
            spec = spec.transpose(1, 2)

        # spec shape
        # [B,C,degree]

        # -------------------------------------------------
        # 3. TexFilter
        # -------------------------------------------------

        # TexFilter expects [B,F,C]
        #spec_f = spec.transpose(1, 2)
        spec_f = spec.permute(0,2,1).contiguous()
        if self.filter_used == "TexFilter":
            spec_f = spec_f * self.texfilter(spec_f)
            #spec_f.mul_(self.texfilter(spec_f))
        elif self.filter_used == "LPF":
            # Zero out high-frequency components (simple low-pass filter)
            cutoff = self.degree // 2  # Keep only the lower half of the frequencies
            spec_f[:, cutoff:, :] = 0
        elif self.filter_used == "nofilter":
            pass  # No filtering applied
        spec = spec_f.transpose(1, 2)

        # -------------------------------------------------
        # 4. Frequency Interpolation
        # -------------------------------------------------

        if self.individual:

            B = spec.size(0)

            spec_up = torch.empty_like(spec)

            for i in range(self.channels):

                spec_up[:, i, :] = self.freq_upsampler[i](spec[:, i, :])

        else:

            spec_up = self.freq_upsampler(spec)

        # -------------------------------------------------
        # 5. Legendre Decode
        # -------------------------------------------------

        if self.optimize_precompute_legendre:

            # [B,C,degree] @ [degree,seq]
            low_xy = torch.matmul(spec_up, self.basis)

        else:

            low_xy = legendre_decode(
                spec_up.transpose(1, 2),
                seq_len=self.seq_len
            ).transpose(1, 2)

        # [B,C,seq] -> [B,seq,C]
        low_xy = low_xy.transpose(1, 2)

        low_xy = low_xy * self.length_ratio

        # -------------------------------------------------
        # 6. Reverse RevIN
        # -------------------------------------------------

        xy_with_sqrt = low_xy * x_std

        xy = xy_with_sqrt + x_mean

        return xy, xy_with_sqrt
    

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.degree = getattr(configs, "degree", 5)

        self.optimize_precompute_legendre = getattr(
            configs, "optimize_precompute_legendre", True
        )
        self.filter_used = getattr(configs, "filter_used", "LPF")
        self.basis_type = getattr(configs, "basis_type", "legendre")
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # -----------------------------
        # NormLin (minimal params)
        # -----------------------------
        self.use_normlin = getattr(configs, "use_normlin", True)

        if self.use_normlin:
            self.normlin_W = nn.Parameter(torch.randn(self.degree, self.degree))
            nn.init.xavier_uniform_(self.normlin_W)

        # -----------------------------
        # Frequency Upsampler
        # -----------------------------
        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.degree, self.degree)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear(self.degree, self.degree)

        # -----------------------------
        # Precompute Basis
        # -----------------------------
        if self.optimize_precompute_legendre:

            t = np.linspace(-1, 1, self.seq_len)
            if self.basis_type != "fourier":
                if self.basis_type == "legendre":
                    from scipy.special import legendre
                    basis = np.array([legendre(i)(t) for i in range(self.degree)])

                elif self.basis_type == "chebyshev":
                    from numpy.polynomial.chebyshev import chebvander
                    basis = chebvander(t, self.degree - 1).T


                elif self.basis_type == "hermite":
                    from numpy.polynomial.hermite import hermvander
                    basis = hermvander(t, self.degree - 1).T

                elif self.basis_type == "laguerre":
                    from numpy.polynomial.laguerre import lagvander
                    basis = lagvander(t, self.degree - 1).T

                basis = torch.tensor(basis, dtype=torch.float32)

                self.register_buffer("basis", basis)
                self.register_buffer("basis_T", basis.t().contiguous())

            elif self.basis_type == "fourier":
                self.optimize_precompute_legendre = False
                self.freq_dim = 2 * self.degree # because of sin/cos pairs
                self.normlin_W = nn.Parameter(torch.randn(self.freq_dim, self.freq_dim))
                if self.individual:
                    self.freq_upsampler = nn.ModuleList([
                        nn.Linear(self.freq_dim, self.freq_dim)
                        for _ in range(self.channels)
                    ])
                else:
                    self.freq_upsampler = nn.Linear(self.freq_dim, self.freq_dim)
            
            #t = np.linspace(0, 1, self.seq_len)
            #device = self.normlin_W.device  # use same device as parameters
            #basis = self._build_fourier_basis(t,device)
            #assert self.degree % 2 == 0

    # -----------------------------
    # Fourier basis (if used)
    # -----------------------------
    def _build_fourier_basis_not_used(self, t,device):
        """
        Build a Fourier basis with exactly `degree` vectors, adding 1 if degree is odd.
        """
        # Ensure even degree for consistent sin/cos pairing
        degree = self.degree
        if degree % 2 != 0:
            degree += 1  # add 1 if odd

        # Convert t to tensor if it's not already
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)

        # Start with constant term
        basis = [torch.ones_like(t)]

        for k in range(1, degree // 2):
            basis.append(np.sin(2 * np.pi * k * t))
            basis.append(np.cos(2 * np.pi * k * t))

        return np.array(basis)
    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, x):

        # -------------------------------------------------
        # 1. RevIN
        # -------------------------------------------------
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - x_mean) / x_std

        # -------------------------------------------------
        # 2. Encode (Legendre / other basis)
        # -------------------------------------------------
        if self.optimize_precompute_legendre:
            x_t = x.transpose(1, 2).contiguous()
            spec = torch.matmul(x_t, self.basis_T)
        else:
            # -------------------------------------------------
            # 2. Encode (Fourier - correct)
            # -------------------------------------------------
            if self.basis_type == "fourier":
                # FFT: [B, L, C] → [B, L/2+1, C] (complex)
                spec = torch.fft.rfft(x, dim=1)

                # Truncate low frequencies
                spec = spec[:, :self.degree, :]                      # [B, degree, C]

                # Move to [B, C, degree]
                spec = spec.permute(0, 2, 1)

                # Convert complex → real representation
                spec = torch.view_as_real(spec)                      # [B, C, degree, 2]

                # Merge real/imag into feature dimension
                B, C, D, _ = spec.shape
                spec = spec.reshape(B, C, 2 * D)                     # [B, C, 2*degree]

            #spec = legendre_encode(x, degree=self.degree)
            #spec = spec.transpose(1, 2)

        # spec: [B, C, degree]

        # -------------------------------------------------
        # 2.5 NormLin (frequency mixing)
        # -------------------------------------------------
        if self.use_normlin:
            W_pos = F.softplus(self.normlin_W)
            W_norm = W_pos / (W_pos.sum(dim=1, keepdim=True) + 1e-8)
            spec = torch.matmul(spec, W_norm.T)

        # -------------------------------------------------
        # 3. LPF (optional hard constraint)
        # -------------------------------------------------
        if self.filter_used == "LPF" and self.basis_type == "fourier":
            cutoff = self.degree // 2

            real = spec[:, :, :self.degree]
            imag = spec[:, :, self.degree:]

            real[:, :, cutoff:] = 0
            imag[:, :, cutoff:] = 0

            spec = torch.cat([real, imag], dim=2)
        else:
            cutoff = self.degree // 2
            spec[:, :, cutoff:] = 0
        # -------------------------------------------------
        # 4. Frequency Interpolation
        # -------------------------------------------------
        if self.individual:
            spec_up = torch.empty_like(spec)
            for i in range(self.channels):
                spec_up[:, i, :] = self.freq_upsampler[i](spec[:, i, :])
        else:
            spec_up = self.freq_upsampler(spec)

        # -------------------------------------------------
        # 5. Decode
        # -------------------------------------------------
        if self.optimize_precompute_legendre:
            low_xy = torch.matmul(spec_up, self.basis)
        else:
            # -------------------------------------------------
            # 5. Decode (Fourier - correct)
            # -------------------------------------------------
            if self.basis_type == "fourier":
                B, C, D2 = spec_up.shape
                D = self.degree  # number of frequency bins

                # Restore real/imag pairs
                spec_up = spec_up.view(B, C, D, 2)                   # [B, C, degree, 2]

                # Convert back to complex
                spec_up_complex = torch.view_as_complex(spec_up)     # [B, C, degree]

                # Prepare full spectrum
                full_spec = torch.zeros(
                    B, self.seq_len // 2 + 1, C,
                    dtype=torch.cfloat,
                    device=spec_up.device
                )

                # Place learned low frequencies
                full_spec[:, :D, :] = spec_up_complex.permute(0, 2, 1)  # [B, degree, C]

                # Inverse FFT → time domain
                low_xy = torch.fft.irfft(full_spec, n=self.seq_len, dim=1)  # [B, L, C]
            #low_xy = legendre_decode(
            #    spec_up.transpose(1, 2),
            #    seq_len=self.seq_len
            #).transpose(1, 2)

        low_xy = low_xy.transpose(1, 2)
        low_xy = low_xy * self.length_ratio

        # -------------------------------------------------
        # 6. Reverse RevIN
        # -------------------------------------------------
        xy_with_sqrt = low_xy * x_std
        xy = xy_with_sqrt + x_mean

        return xy, xy_with_sqrt