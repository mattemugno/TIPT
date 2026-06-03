from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .shape_modules import ShapePatchEmbedding, to_2tuple
from .tipt_vitpose import (
    TiptVitPoseOutput,
    _choose_num_heads,
    _config_get,
    feature_map_to_tokens,
    tokens_to_feature_map,
)


def _normalize_unit(values: torch.Tensor, eps: float) -> torch.Tensor:
    min_value = values.amin(dim=(-2, -1), keepdim=True)
    max_value = values.amax(dim=(-2, -1), keepdim=True)
    return (values - min_value) / (max_value - min_value).clamp_min(eps)


def _normalize_signed(values: torch.Tensor, eps: float) -> torch.Tensor:
    scale = values.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return values / scale


class StructuralFeatureLayer(nn.Module):
    """Build multi-channel structural views from a normalized RGB crop."""

    def __init__(
        self,
        channels: Sequence[str] = ("sobel_x", "sobel_y", "magnitude"),
        eps: float = 1e-6,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.eps = eps
        self.normalize = normalize

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3), persistent=False)

    @property
    def out_channels(self) -> int:
        return len(self.channels)

    def _gray(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] == 1:
            return pixel_values
        return pixel_values.mean(dim=1, keepdim=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] pixel_values, got {tuple(pixel_values.shape)}")

        gray = self._gray(pixel_values)
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        laplacian_kernel = self.laplacian.to(device=gray.device, dtype=gray.dtype)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + self.eps)
        laplacian = F.conv2d(gray, laplacian_kernel, padding=1)

        available = {
            "gray": _normalize_unit(gray, self.eps) if self.normalize else gray,
            "sobel_x": _normalize_signed(grad_x, self.eps) if self.normalize else grad_x,
            "sobel_y": _normalize_signed(grad_y, self.eps) if self.normalize else grad_y,
            "magnitude": _normalize_unit(magnitude, self.eps) if self.normalize else magnitude,
            "laplacian": _normalize_signed(laplacian, self.eps) if self.normalize else laplacian,
        }

        unknown = [name for name in self.channels if name not in available]
        if unknown:
            raise ValueError(f"Unknown structural channel(s): {unknown}")
        return torch.cat([available[name] for name in self.channels], dim=1)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class EdgeCnnStem(nn.Module):
    """Small local feature extractor before patchifying structural maps."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth <= 0:
            self.net = nn.Identity()
            self.out_channels = in_channels
            return

        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(current_channels, stem_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(stem_channels), stem_channels),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            current_channels = stem_channels

        self.net = nn.Sequential(*layers)
        self.out_channels = stem_channels

    def forward(self, structural_maps: torch.Tensor) -> torch.Tensor:
        return self.net(structural_maps)


class DynamicShapeTextureGate(nn.Module):
    """Image-conditioned gate for shape-first texture residuals."""

    def __init__(self, embed_dim: int, hidden_ratio: float = 0.25, gate_init: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(16, int(embed_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.net[-1].bias, torch.logit(torch.tensor(gate_init)).item())

    def forward(self, shape_tokens: torch.Tensor) -> torch.Tensor:
        pooled = shape_tokens.mean(dim=1)
        return torch.sigmoid(self.net(pooled)).unsqueeze(1)


class ShapeTextureResidualBlock(nn.Module):
    """Shape encoder block followed by shape-guided cross-attention residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        gate_init: float = 0.1,
        gate_hidden_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.shape_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.texture_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = DynamicShapeTextureGate(
            embed_dim=embed_dim,
            hidden_ratio=gate_hidden_ratio,
            gate_init=gate_init,
        )

    def forward(self, shape_tokens: torch.Tensor, texture_tokens: torch.Tensor) -> torch.Tensor:
        shape_tokens = self.shape_block(shape_tokens)
        attn_out, _ = self.cross_attn(
            query=self.query_norm(shape_tokens),
            key=self.texture_norm(texture_tokens),
            value=texture_tokens,
            need_weights=False,
        )
        return shape_tokens + self.gate(shape_tokens).to(dtype=attn_out.dtype) * attn_out


class TiptVitPoseV3ForPoseEstimation(nn.Module):
    """Shape-first TIPT-v3 architecture with multi-level residual fusion."""

    def __init__(
        self,
        checkpoint: str = "usyd-community/vitpose-base-simple",
        structural_channels: Sequence[str] = ("sobel_x", "sobel_y", "magnitude"),
        stem_channels: int = 32,
        stem_depth: int = 3,
        shape_depth: int = 4,
        shape_dropout: float = 0.0,
        num_heads: int | None = None,
        gate_init: float = 0.1,
        gate_hidden_ratio: float = 0.25,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        try:
            from transformers import VitPoseForPoseEstimation
        except ImportError as exc:
            raise ImportError(
                "TiptVitPoseV3ForPoseEstimation requires transformers. "
                "Install the project requirements before constructing the model."
            ) from exc

        self.vitpose = VitPoseForPoseEstimation.from_pretrained(checkpoint)

        backbone_config = self.vitpose.config.backbone_config
        image_size = to_2tuple(_config_get(backbone_config, "image_size", (256, 192)))
        patch_size = to_2tuple(_config_get(backbone_config, "patch_size", (16, 16)))
        embed_dim = int(_config_get(backbone_config, "hidden_size"))
        heads = _choose_num_heads(embed_dim, num_heads or _config_get(backbone_config, "num_attention_heads"))

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.embed_dim = embed_dim
        self.num_heads = heads

        self.structural = StructuralFeatureLayer(channels=structural_channels)
        self.edge_stem = EdgeCnnStem(
            in_channels=self.structural.out_channels,
            stem_channels=stem_channels,
            depth=stem_depth,
            dropout=shape_dropout,
        )
        self.shape_embed = ShapePatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=self.edge_stem.out_channels,
        )
        self.fusion_blocks = nn.ModuleList(
            [
                ShapeTextureResidualBlock(
                    embed_dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=shape_dropout,
                    gate_init=gate_init,
                    gate_hidden_ratio=gate_hidden_ratio,
                )
                for _ in range(shape_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    @property
    def backbone(self) -> nn.Module:
        return self.vitpose.backbone

    @property
    def head(self) -> nn.Module:
        return self.vitpose.head

    def new_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.structural.parameters()
        yield from self.edge_stem.parameters()
        yield from self.shape_embed.parameters()
        yield from self.fusion_blocks.parameters()
        yield from self.final_norm.parameters()

    def pretrained_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.vitpose.parameters()

    def set_pretrained_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.vitpose.parameters():
            parameter.requires_grad = requires_grad

    def _run_backbone(
        self,
        pixel_values: torch.Tensor,
        dataset_index: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> Any:
        if hasattr(self.backbone, "forward_with_filtered_kwargs"):
            return self.backbone.forward_with_filtered_kwargs(
                pixel_values,
                dataset_index=dataset_index,
                **kwargs,
            )
        if dataset_index is None:
            return self.backbone(pixel_values=pixel_values, **kwargs)
        return self.backbone(pixel_values=pixel_values, dataset_index=dataset_index, **kwargs)

    def _extract_texture_tokens(self, backbone_outputs: Any) -> torch.Tensor:
        feature_maps = getattr(backbone_outputs, "feature_maps", None)
        if feature_maps is None:
            feature_maps = getattr(backbone_outputs, "hidden_states", None)
        if not feature_maps:
            raise RuntimeError("Backbone did not return feature_maps or hidden_states.")
        return feature_map_to_tokens(feature_maps[-1])

    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: torch.Tensor | None = None,
        flip_pairs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TiptVitPoseOutput:
        backbone_outputs = self._run_backbone(pixel_values, dataset_index, kwargs)
        texture_tokens = self._extract_texture_tokens(backbone_outputs)

        structural_maps = self.structural(pixel_values)
        shape_features = self.edge_stem(structural_maps)
        shape_tokens = self.shape_embed(shape_features)
        for block in self.fusion_blocks:
            shape_tokens = block(shape_tokens, texture_tokens)
        shape_tokens = self.final_norm(shape_tokens)

        feature_map = tokens_to_feature_map(shape_tokens, self.grid_size)
        heatmaps = self.head(feature_map, flip_pairs=flip_pairs)

        return TiptVitPoseOutput(
            heatmaps=heatmaps,
            F_shape=shape_tokens,
            F_tex=texture_tokens,
            fused_tokens=shape_tokens,
            hidden_states=getattr(backbone_outputs, "hidden_states", None),
            attentions=getattr(backbone_outputs, "attentions", None),
        )


TiptVitPoseV2ForPoseEstimation = TiptVitPoseV3ForPoseEstimation
