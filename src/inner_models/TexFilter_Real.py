import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class TexFilter_old(nn.Module):
    def __init__(self, embed_size, scale=0.02, sparsity_threshold=0.01,
                 use_gelu=False, use_swish=False, use_skip=False,
                 use_layernorm=False, hard_threshold=False,
                 use_window=False):
        super().__init__()
        self.embed_size = embed_size
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold

        # Ablation flags
        self.use_gelu = use_gelu
        self.use_swish = use_swish
        self.use_skip = use_skip
        self.use_layernorm = use_layernorm
        self.hard_threshold = hard_threshold
        self.use_window = use_window

        # Real-valued weights for linear layers
        self.w = nn.Parameter(self.scale * torch.randn(embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.bias1 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.bias2 = nn.Parameter(self.scale * torch.randn(embed_size))

        if self.use_layernorm:
            self.norm1 = nn.LayerNorm(embed_size, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(embed_size, elementwise_affine=False)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):  # x: [B, F, C], real-valued
        if self.use_window:
            window = torch.hann_window(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = x * window  # Apply Hanning window

        # Apply first linear transformation along channel dimension (C)
        # We do element-wise multiply by w and add bias (broadcasting along batch and freq)
        o1 = x * self.w + self.bias1

        # Activation
        if self.use_gelu:
            o1 = F.gelu(o1)
        elif self.use_swish:
            o1 = self.swish(o1)
        else:
            o1 = F.relu(o1)

        if self.use_skip:
            o1 = o1 + x  # skip connection

        # Second linear transformation
        o2 = o1 * self.w1 + self.bias2

        # Thresholding (hard or soft)
        if self.hard_threshold:
            o2 = torch.where(o2.abs() < self.sparsity_threshold, torch.zeros_like(o2), o2)
        else:
            o2 = F.softshrink(o2, lambd=self.sparsity_threshold)

        if self.use_layernorm:
            o2 = self.norm2(self.norm1(o2))

        return o2



class TexFilter(nn.Module):

    def __init__(self, embed_size, scale=0.02, sparsity_threshold=0.01,
                 use_gelu=False, use_swish=False, use_skip=False,
                 use_layernorm=False, hard_threshold=False,
                 use_window=False):

        super().__init__()

        self.embed_size = embed_size
        self.sparsity_threshold = sparsity_threshold

        self.use_gelu = use_gelu
        self.use_swish = use_swish
        self.use_skip = use_skip
        self.use_layernorm = use_layernorm
        self.hard_threshold = hard_threshold
        self.use_window = use_window

        std = 1 / math.sqrt(embed_size)

        self.w = nn.Parameter(torch.randn(embed_size) * std)
        self.w1 = nn.Parameter(torch.randn(embed_size) * std)

        self.bias1 = nn.Parameter(torch.zeros(embed_size))
        self.bias2 = nn.Parameter(torch.zeros(embed_size))

        if use_layernorm:
            self.norm = nn.LayerNorm(embed_size, elementwise_affine=False)

        if use_window:
            self.register_buffer("window", torch.hann_window(4096), persistent=False)

    def forward(self, x):  # [B,F,C]

        if self.use_window:
            w = self.window[:x.size(1)].view(1, -1, 1)
            x = x * w

        o1 = x * self.w + self.bias1

        if self.use_gelu:
            o1 = F.gelu(o1)
        elif self.use_swish:
            o1 = F.silu(o1)
        else:
            o1 = F.relu(o1)

        if self.use_skip:
            o1 = o1 + x

        o2 = o1 * self.w1 + self.bias2

        if self.use_layernorm:
            o2 = self.norm(o2)

        if self.hard_threshold:
            o2 = o2.masked_fill(o2.abs() < self.sparsity_threshold, 0.0)
        else:
            o2 = F.softshrink(o2, lambd=self.sparsity_threshold)

        return o2