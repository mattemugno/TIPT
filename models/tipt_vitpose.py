from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .fusion import ShapeGuidedCrossAttention
from .shape_modules import ShapeEncoder, ShapePatchEmbedding, SobelEdgeLayer, to_2tuple


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _choose_num_heads(embed_dim: int, preferred: int | None) -> int:
    candidates = [preferred, 12, 8, 6, 4, 3, 2, 1]
    for candidate in candidates:
        if candidate and embed_dim % candidate == 0:
            return int(candidate)
    raise ValueError(f"Could not find a compatible attention head count for embed_dim={embed_dim}")


def tokens_to_feature_map(tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B, N, D] tokens, got {tuple(tokens.shape)}")
    batch_size, num_tokens, embed_dim = tokens.shape
    height, width = grid_size
    if num_tokens != height * width:
        raise ValueError(
            f"Cannot reshape {num_tokens} tokens to grid {grid_size}; "
            "check image_size and patch_size in the ViTPose config."
        )
    return tokens.transpose(1, 2).reshape(batch_size, embed_dim, height, width).contiguous()


def feature_map_to_tokens(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 3:
        return features
    if features.ndim == 4:
        return features.flatten(2).transpose(1, 2).contiguous()
    raise ValueError(f"Expected [B, N, D] or [B, D, H, W] features, got {tuple(features.shape)}")


@dataclass
class TiptVitPoseOutput:
    heatmaps: torch.Tensor
    F_shape: torch.Tensor
    F_tex: torch.Tensor
    fused_tokens: torch.Tensor
    hidden_states: Any | None = None
    attentions: Any | None = None
    loss: torch.Tensor | None = None


class TiptVitPoseForPoseEstimation(nn.Module):
    """Texture-Invariant Pose Transformer wrapper over Hugging Face ViTPose."""

    def __init__(
        self,
        checkpoint: str = "usyd-community/vitpose-base-simple",
        shape_depth: int = 4,
        shape_dropout: float = 0.0,
        num_heads: int | None = None,
        fusion: str = "cross_attention",
        structural_view: str = "sobel",
        alpha_init: float = 0.1,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        try:
            from transformers import VitPoseForPoseEstimation
        except ImportError as exc:
            raise ImportError(
                "TiptVitPoseForPoseEstimation requires transformers. "
                "Install the project requirements before constructing the model."
            ) from exc

        self.vitpose = VitPoseForPoseEstimation.from_pretrained(checkpoint)
        self.fusion = fusion

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

        self.edge = SobelEdgeLayer(mode=structural_view)
        self.shape_embed = ShapePatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=1,
        )
        self.shape_encoder = ShapeEncoder(
            embed_dim=embed_dim,
            num_heads=heads,
            depth=shape_depth,
            mlp_ratio=mlp_ratio,
            dropout=shape_dropout,
        )
        self.cross_fusion = ShapeGuidedCrossAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=shape_dropout,
            alpha_init=alpha_init,
        )
        self.simple_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    @property
    def backbone(self) -> nn.Module:
        return self.vitpose.backbone

    @property
    def head(self) -> nn.Module:
        return self.vitpose.head

    def new_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.edge.parameters()
        yield from self.shape_embed.parameters()
        yield from self.shape_encoder.parameters()
        yield from self.cross_fusion.parameters()
        yield self.simple_alpha

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

    def _fuse(self, shape_tokens: torch.Tensor, texture_tokens: torch.Tensor) -> torch.Tensor:
        if self.fusion == "cross_attention":
            return self.cross_fusion(shape_tokens, texture_tokens)
        if self.fusion in {"simple", "add", "simple_fusion"}:
            return shape_tokens + self.simple_alpha.to(dtype=texture_tokens.dtype) * texture_tokens
        if self.fusion in {"shape", "shape_only"}:
            return shape_tokens
        if self.fusion in {"texture", "texture_only", "baseline"}:
            return texture_tokens
        raise ValueError(
            "fusion must be one of cross_attention, simple/add, shape_only, or texture_only; "
            f"got {self.fusion!r}"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: torch.Tensor | None = None,
        flip_pairs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TiptVitPoseOutput:
        backbone_outputs = self._run_backbone(pixel_values, dataset_index, kwargs)
        texture_tokens = self._extract_texture_tokens(backbone_outputs)

        edge = self.edge(pixel_values)
        shape_tokens = self.shape_embed(edge)
        shape_tokens = self.shape_encoder(shape_tokens)

        fused_tokens = self._fuse(shape_tokens, texture_tokens)
        feature_map = tokens_to_feature_map(fused_tokens, self.grid_size)
        heatmaps = self.head(feature_map, flip_pairs=flip_pairs)

        return TiptVitPoseOutput(
            heatmaps=heatmaps,
            F_shape=shape_tokens,
            F_tex=texture_tokens,
            fused_tokens=fused_tokens,
            hidden_states=getattr(backbone_outputs, "hidden_states", None),
            attentions=getattr(backbone_outputs, "attentions", None),
        )
