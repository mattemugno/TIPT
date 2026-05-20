from __future__ import annotations

import torch
from torch import nn


class ShapeGuidedCrossAttention(nn.Module):
    """Cross-attention where shape tokens query pretrained texture tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, shape_tokens: torch.Tensor, texture_tokens: torch.Tensor) -> torch.Tensor:
        if shape_tokens.shape != texture_tokens.shape:
            raise ValueError(
                "shape_tokens and texture_tokens must have identical [B, N, D] shapes, "
                f"got {tuple(shape_tokens.shape)} and {tuple(texture_tokens.shape)}"
            )
        attn_out, _ = self.cross_attn(
            query=shape_tokens,
            key=texture_tokens,
            value=texture_tokens,
            need_weights=False,
        )
        return shape_tokens + self.alpha.to(dtype=attn_out.dtype) * attn_out
