import torch
import torch.nn as nn

class ResidualHyperBlock(nn.Module):
    """
    Supports:
    - Residual connection
    - Hyper-connection (cross-layer mixing)
    - Optional manifold hyper projection
    """

    def __init__(self, dim, use_residual=True, use_hyper=False, use_manifold=False):
        super().__init__()

        self.use_residual = use_residual
        self.use_hyper = use_hyper
        self.use_manifold = use_manifold

        if use_hyper:
            self.hyper_proj = nn.Linear(dim, dim)

        if use_manifold:
            self.manifold_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, prev=None):
        """
        x: current representation
        prev: previous layer representation (for hyper connections)
        """

        out = x

        # Residual connection
        if self.use_residual:
            out = out + x

        # Hyper connection
        if self.use_hyper and prev is not None:
            out = out + self.hyper_proj(prev)

        # Manifold hyper connection
        if self.use_manifold:
            out = out + self.manifold_proj(out)

        return self.norm(out)