from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got {value!r}")
    return (int(value[0]), int(value[1]))


class SobelEdgeLayer(nn.Module):
    """Fixed Sobel edge extractor, with a grayscale ablation mode."""

    def __init__(self, eps: float = 1e-6, normalize: bool = True, mode: str = "sobel") -> None:
        super().__init__()
        self.eps = eps
        self.normalize = normalize
        self.mode = mode

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _grayscale(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] == 1:
            return pixel_values
        return pixel_values.mean(dim=1, keepdim=True)

    def _normalize_unit(self, values: torch.Tensor) -> torch.Tensor:
        min_value = values.amin(dim=(-2, -1), keepdim=True)
        max_value = values.amax(dim=(-2, -1), keepdim=True)
        return (values - min_value) / (max_value - min_value).clamp_min(self.eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] pixel_values, got {tuple(pixel_values.shape)}")

        gray = self._grayscale(pixel_values)
        if self.mode == "grayscale":
            return self._normalize_unit(gray) if self.normalize else gray
        if self.mode != "sobel":
            raise ValueError(f"Unsupported structural view mode {self.mode!r}; use 'sobel' or 'grayscale'.")

        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

        if self.normalize:
            edge = self._normalize_unit(edge)
        return edge


class ShapePatchEmbedding(nn.Module):
    """Patchify a structural image view into ViT-compatible shape tokens."""

    def __init__(
        self,
        image_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        embed_dim: int,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos_embed_for_grid(self, height: int, width: int) -> torch.Tensor:
        if (height, width) == self.grid_size:
            return self.pos_embed

        pos = self.pos_embed.reshape(1, self.grid_size[0], self.grid_size[1], -1)
        pos = pos.permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, height * width, -1)

    def forward(self, edge_map: torch.Tensor) -> torch.Tensor:
        patches = self.proj(edge_map)
        batch_size, channels, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2).contiguous()
        pos_embed = self._pos_embed_for_grid(height, width)
        pos_embed = pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + pos_embed.expand(batch_size, -1, -1)


class ShapeEncoder(nn.Module):
    """Small Transformer encoder for shape tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        if depth <= 0:
            self.encoder = nn.Identity()
            return

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(embed_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
