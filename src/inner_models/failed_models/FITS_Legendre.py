

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.degree = getattr(configs, "degree", 10)
        self.individual = configs.individual

        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # ------------------------------------------------
        # Precompute Legendre Basis
        # ------------------------------------------------

        t = np.linspace(-1, 1, self.seq_len)
        basis = np.array([legendre(i)(t) for i in range(self.degree)])

        basis = torch.tensor(basis, dtype=torch.float32)

        # ------------------------------------------------
        # Legendre Projection (Conv)
        # ------------------------------------------------

        self.leg_proj = nn.Conv1d(
            in_channels=1,
            out_channels=self.degree,
            kernel_size=self.seq_len,
            bias=False
        )

        self.leg_proj.weight.data = basis.unsqueeze(1)
        self.leg_proj.weight.requires_grad = False

        # ------------------------------------------------
        # Legendre Reconstruction
        # ------------------------------------------------

        self.leg_recon = nn.ConvTranspose1d(
            in_channels=self.degree,
            out_channels=1,
            kernel_size=self.seq_len,
            bias=False
        )

        self.leg_recon.weight.data = basis.unsqueeze(1)
        self.leg_recon.weight.requires_grad = False

        # ------------------------------------------------
        # Frequency Interpolation
        # ------------------------------------------------

        if self.individual:

            self.freq_upsampler = nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=1,
                groups=self.channels
            )

        else:

            self.freq_upsampler = nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=1
            )

        # ------------------------------------------------
        # TexFilter
        # ------------------------------------------------

        self.texfilter = TexFilter(
            embed_size=self.channels,
            use_gelu=True,
            use_skip=True,
            use_layernorm=True,
            hard_threshold=False,
            use_window=False,
            sparsity_threshold=0.0
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        B, T, C = x.shape

        # ------------------------------------------------
        # RevIN
        # ------------------------------------------------

        x_mean = x.mean(1, keepdim=True)
        x_std = x.std(1, keepdim=True) + 1e-5

        x = (x - x_mean) / x_std

        # ------------------------------------------------
        # Prepare for Conv
        # ------------------------------------------------

        x = x.permute(0,2,1).contiguous()   # [B,C,T]

        x = x.view(B*C,1,T)

        # ------------------------------------------------
        # Legendre Projection
        # ------------------------------------------------

        spec = self.leg_proj(x)

        spec = spec.view(B,C,self.degree)

        # ------------------------------------------------
        # TexFilter
        # ------------------------------------------------

        spec = spec.transpose(1,2)   # [B,F,C]

        spec = spec * self.texfilter(spec)

        spec = spec.transpose(1,2)

        # ------------------------------------------------
        # Frequency Interpolation
        # ------------------------------------------------

        spec = self.freq_upsampler(spec)

        # ------------------------------------------------
        # Reconstruction
        # ------------------------------------------------

        spec = spec.view(B*C,self.degree,1)

        low_xy = self.leg_recon(spec)

        low_xy = low_xy.view(B,C,T)

        low_xy = low_xy.permute(0,2,1)

        low_xy = low_xy * self.length_ratio

        # ------------------------------------------------
        # Reverse RevIN
        # ------------------------------------------------

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
        self.degree = getattr(configs, "degree", 10)

        self.optimize_precompute_legendre = getattr(
            configs, "optimize_precompute_legendre", True
        )

        # new flag
        self.use_freq_conv = getattr(configs, "use_freq_conv", False)

        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # -------------------------------------------------
        # Frequency Mixer (Linear or Conv1D)
        # -------------------------------------------------

        if self.use_freq_conv:

            # depthwise spectral convolution
            self.freq_conv = nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels
            )

        else:

            if self.individual:
                self.freq_upsampler = nn.ModuleList([
                    nn.Linear(self.degree, self.degree)
                    for _ in range(self.channels)
                ])
            else:
                self.freq_upsampler = nn.Linear(self.degree, self.degree)

        # -------------------------------------------------
        # Precompute Legendre Basis
        # -------------------------------------------------

        if self.optimize_precompute_legendre:

            t = np.linspace(-1, 1, self.seq_len)

            basis = np.array([
                legendre(i)(t) for i in range(self.degree)
            ])

            self.register_buffer(
                "leg_basis",
                torch.tensor(basis, dtype=torch.float32)
            )

        # -------------------------------------------------
        # Learnable Frequency Filter
        # -------------------------------------------------

        self.texfilter = TexFilter(
            embed_size=self.channels,
            use_gelu=True,
            use_skip=True,
            use_layernorm=True,
            hard_threshold=False,
            use_window=False,
            sparsity_threshold=0.0
        )

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
            spec = torch.matmul(x_t, self.leg_basis.t())

        else:

            spec = legendre_encode(x, degree=self.degree)
            spec = spec.transpose(1, 2)

        # spec shape
        # [B,C,degree]

        # -------------------------------------------------
        # 3. TexFilter
        # -------------------------------------------------

        # TexFilter expects [B,F,C]
        spec_f = spec.transpose(1, 2)

        spec_f = spec_f * self.texfilter(spec_f)

        spec = spec_f.transpose(1, 2)

        # -------------------------------------------------
        # 4. Frequency Mixing
        # -------------------------------------------------

        if self.use_freq_conv:

            # spec already [B,C,degree]
            spec_up = self.freq_conv(spec)

        else:

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
            low_xy = torch.matmul(spec_up, self.leg_basis)

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