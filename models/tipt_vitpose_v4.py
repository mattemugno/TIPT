from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn

from .shape_modules import ShapePatchEmbedding, to_2tuple
from .tipt_vitpose import (
    TiptVitPoseOutput,
    _choose_num_heads,
    _config_get,
    tokens_to_feature_map,
)
from .tipt_vitpose_v3 import DynamicShapeTextureGate, EdgeCnnStem, StructuralFeatureLayer


def _as_1indexed_layers(values: Sequence[int] | None, num_layers: int) -> tuple[int, ...]:
    if values is None:
        return tuple(layer for layer in (3, 6, 9, 12) if layer <= num_layers)

    layers = tuple(int(value) for value in values)
    if not layers:
        raise ValueError("fusion_layers cannot be empty")
    if min(layers) >= 1 and max(layers) <= num_layers:
        return layers
    if min(layers) >= 0 and max(layers) < num_layers:
        return tuple(layer + 1 for layer in layers)
    raise ValueError(f"fusion_layers must be 1-indexed in [1, {num_layers}] or 0-indexed in [0, {num_layers - 1}]")


class StructuralPatchEmbedding(nn.Module):
    """Patch embedding that injects structural channels before the ViT encoder."""

    def __init__(
        self,
        pretrained_projection: nn.Conv2d,
        structural_channels: int,
        edge_init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        in_channels = int(pretrained_projection.in_channels) + int(structural_channels)
        self.projection = nn.Conv2d(
            in_channels,
            pretrained_projection.out_channels,
            kernel_size=pretrained_projection.kernel_size,
            stride=pretrained_projection.stride,
            padding=pretrained_projection.padding,
            dilation=pretrained_projection.dilation,
            groups=pretrained_projection.groups,
            bias=pretrained_projection.bias is not None,
            padding_mode=pretrained_projection.padding_mode,
        )
        self.reset_from_pretrained(pretrained_projection, structural_channels, edge_init_scale)

    def reset_from_pretrained(
        self,
        pretrained_projection: nn.Conv2d,
        structural_channels: int,
        edge_init_scale: float,
    ) -> None:
        with torch.no_grad():
            self.projection.weight.zero_()
            rgb_channels = int(pretrained_projection.in_channels)
            self.projection.weight[:, :rgb_channels].copy_(pretrained_projection.weight)
            edge_seed = pretrained_projection.weight.mean(dim=1, keepdim=True) * float(edge_init_scale)
            self.projection.weight[:, rgb_channels:].copy_(edge_seed.expand(-1, structural_channels, -1, -1))
            if self.projection.bias is not None and pretrained_projection.bias is not None:
                self.projection.bias.copy_(pretrained_projection.bias)

    def forward(self, pixel_values: torch.Tensor, structural_maps: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        inputs = torch.cat([pixel_values, structural_maps.to(dtype=pixel_values.dtype)], dim=1)
        patches = self.projection(inputs)
        grid_size = (patches.shape[-2], patches.shape[-1])
        tokens = patches.flatten(2).transpose(1, 2).contiguous()
        return tokens, grid_size


class BidirectionalShapeTextureFusion(nn.Module):
    """Cross-attention that updates texture and shape streams inside the transformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        gate_init: float = 0.15,
        gate_hidden_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.texture_norm = nn.LayerNorm(embed_dim)
        self.shape_norm = nn.LayerNorm(embed_dim)
        self.texture_from_shape = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.shape_from_texture = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.texture_gate = DynamicShapeTextureGate(embed_dim, gate_hidden_ratio, gate_init)
        self.shape_gate = DynamicShapeTextureGate(embed_dim, gate_hidden_ratio, gate_init)

    def forward(self, texture_tokens: torch.Tensor, shape_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        texture_norm = self.texture_norm(texture_tokens)
        shape_norm = self.shape_norm(shape_tokens)
        texture_update, _ = self.texture_from_shape(
            query=texture_norm,
            key=shape_norm,
            value=shape_tokens,
            need_weights=False,
        )
        texture_tokens = texture_tokens + self.texture_gate(shape_tokens).to(dtype=texture_update.dtype) * texture_update

        texture_norm = self.texture_norm(texture_tokens)
        shape_update, _ = self.shape_from_texture(
            query=shape_norm,
            key=texture_norm,
            value=texture_tokens,
            need_weights=False,
        )
        shape_tokens = shape_tokens + self.shape_gate(shape_tokens).to(dtype=shape_update.dtype) * shape_update
        return texture_tokens, shape_tokens


class ShapeGuidedPoseDecoder(nn.Module):
    """Fuse final texture and shape tokens before the pretrained ViTPose heatmap head."""

    def __init__(
        self,
        embed_dim: int,
        hidden_ratio: float = 1.0,
        gate_init: float = 0.2,
        gate_hidden_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_dim = max(embed_dim, int(embed_dim * hidden_ratio))
        self.texture_norm = nn.LayerNorm(embed_dim)
        self.shape_norm = nn.LayerNorm(embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = DynamicShapeTextureGate(embed_dim, gate_hidden_ratio, gate_init)

    def forward(self, texture_tokens: torch.Tensor, shape_tokens: torch.Tensor) -> torch.Tensor:
        texture_norm = self.texture_norm(texture_tokens)
        shape_norm = self.shape_norm(shape_tokens)
        fused = torch.cat(
            [
                texture_norm,
                shape_norm,
                texture_norm - shape_norm,
                texture_norm * shape_norm,
            ],
            dim=-1,
        )
        update = self.fusion(fused)
        return texture_tokens + self.gate(shape_tokens).to(dtype=update.dtype) * update


class TiptVitPoseV4ForPoseEstimation(nn.Module):
    """TIPTv4: structural patch input, in-transformer fusion, and shape-guided decoding."""

    def __init__(
        self,
        checkpoint: str = "usyd-community/vitpose-base-simple",
        structural_channels: Sequence[str] = ("sobel_x", "sobel_y", "magnitude"),
        stem_channels: int = 48,
        stem_depth: int = 3,
        fusion_layers: Sequence[int] | None = None,
        shape_dropout: float = 0.0,
        num_heads: int | None = None,
        gate_init: float = 0.15,
        decoder_gate_init: float = 0.2,
        gate_hidden_ratio: float = 0.25,
        mlp_ratio: int = 4,
        structural_patch_scale: float = 0.05,
        decoder_hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        try:
            from transformers import VitPoseForPoseEstimation
        except ImportError as exc:
            raise ImportError(
                "TiptVitPoseV4ForPoseEstimation requires transformers. "
                "Install the project requirements before constructing the model."
            ) from exc

        self.vitpose = VitPoseForPoseEstimation.from_pretrained(checkpoint)

        backbone_config = self.vitpose.config.backbone_config
        image_size = to_2tuple(_config_get(backbone_config, "image_size", (256, 192)))
        patch_size = to_2tuple(_config_get(backbone_config, "patch_size", (16, 16)))
        embed_dim = int(_config_get(backbone_config, "hidden_size"))
        num_layers = int(_config_get(backbone_config, "num_hidden_layers", len(self.vitpose.backbone.encoder.layer)))
        heads = _choose_num_heads(embed_dim, num_heads or _config_get(backbone_config, "num_attention_heads"))
        fusion_layers_1indexed = _as_1indexed_layers(fusion_layers, num_layers)

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.fusion_layers = fusion_layers_1indexed
        self.fusion_layer_indices = {layer - 1: index for index, layer in enumerate(fusion_layers_1indexed)}

        self.structural = StructuralFeatureLayer(channels=structural_channels)
        pretrained_projection = self.vitpose.backbone.embeddings.patch_embeddings.projection
        self.structural_patch_embed = StructuralPatchEmbedding(
            pretrained_projection=pretrained_projection,
            structural_channels=self.structural.out_channels,
            edge_init_scale=structural_patch_scale,
        )
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
        self.shape_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=heads,
                    dim_feedforward=embed_dim * mlp_ratio,
                    dropout=shape_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in fusion_layers_1indexed
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                BidirectionalShapeTextureFusion(
                    embed_dim=embed_dim,
                    num_heads=heads,
                    dropout=shape_dropout,
                    gate_init=gate_init,
                    gate_hidden_ratio=gate_hidden_ratio,
                )
                for _ in fusion_layers_1indexed
            ]
        )
        self.shape_norm = nn.LayerNorm(embed_dim)
        self.decoder = ShapeGuidedPoseDecoder(
            embed_dim=embed_dim,
            hidden_ratio=decoder_hidden_ratio,
            gate_init=decoder_gate_init,
            gate_hidden_ratio=gate_hidden_ratio,
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

    @property
    def backbone(self) -> nn.Module:
        return self.vitpose.backbone

    @property
    def head(self) -> nn.Module:
        return self.vitpose.head

    def new_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.structural_patch_embed.parameters()
        yield from self.edge_stem.parameters()
        yield from self.shape_embed.parameters()
        yield from self.shape_blocks.parameters()
        yield from self.fusion_blocks.parameters()
        yield from self.shape_norm.parameters()
        yield from self.decoder.parameters()
        yield from self.decoder_norm.parameters()

    def pretrained_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.vitpose.parameters()

    def set_pretrained_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.vitpose.parameters():
            parameter.requires_grad = requires_grad

    def _add_position_embeddings(self, texture_tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        position_embeddings = self.backbone.embeddings.position_embeddings
        patch_pos = position_embeddings[:, 1:] + position_embeddings[:, :1]
        if patch_pos.shape[1] != texture_tokens.shape[1]:
            expected = self.grid_size
            patch_pos = patch_pos.reshape(1, expected[0], expected[1], -1).permute(0, 3, 1, 2)
            patch_pos = nn.functional.interpolate(patch_pos, size=grid_size, mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)
        patch_pos = patch_pos.to(device=texture_tokens.device, dtype=texture_tokens.dtype)
        return self.backbone.embeddings.dropout(texture_tokens + patch_pos)

    def _encode_texture_shape(
        self,
        texture_tokens: torch.Tensor,
        shape_tokens: torch.Tensor,
        dataset_index: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden_states = texture_tokens
        hidden_history: list[torch.Tensor] = []

        for layer_index, layer_module in enumerate(self.backbone.encoder.layer):
            hidden_states = layer_module(hidden_states, dataset_index=dataset_index, **kwargs)
            fusion_index = self.fusion_layer_indices.get(layer_index)
            if fusion_index is not None:
                shape_tokens = self.shape_blocks[fusion_index](shape_tokens)
                hidden_states, shape_tokens = self.fusion_blocks[fusion_index](hidden_states, shape_tokens)
            hidden_history.append(hidden_states)

        texture_tokens = self.backbone.layernorm(hidden_states)
        shape_tokens = self.shape_norm(shape_tokens)
        return texture_tokens, shape_tokens, tuple(hidden_history)

    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: torch.Tensor | None = None,
        flip_pairs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TiptVitPoseOutput:
        structural_maps = self.structural(pixel_values)
        texture_tokens, grid_size = self.structural_patch_embed(pixel_values, structural_maps)
        texture_tokens = self._add_position_embeddings(texture_tokens, grid_size)

        shape_features = self.edge_stem(structural_maps)
        shape_tokens = self.shape_embed(shape_features)
        texture_tokens, shape_tokens, hidden_states = self._encode_texture_shape(
            texture_tokens,
            shape_tokens,
            dataset_index,
            kwargs,
        )

        fused_tokens = self.decoder_norm(self.decoder(texture_tokens, shape_tokens))
        feature_map = tokens_to_feature_map(fused_tokens, grid_size)
        heatmaps = self.head(feature_map, flip_pairs=flip_pairs)

        return TiptVitPoseOutput(
            heatmaps=heatmaps,
            F_shape=shape_tokens,
            F_tex=texture_tokens,
            fused_tokens=fused_tokens,
            hidden_states=hidden_states,
            attentions=None,
        )


COCO_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


class PoseLimbTokenDecoder(nn.Module):
    """Decode COCO limb tokens from structural tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_limbs: int,
        depth: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_limbs, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(depth)]
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * mlp_ratio,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, structural_tokens: torch.Tensor) -> torch.Tensor:
        limb_tokens = self.queries.expand(structural_tokens.shape[0], -1, -1)
        for query_norm, memory_norm, cross_attn, block in zip(
            self.query_norms,
            self.memory_norms,
            self.cross_attn,
            self.blocks,
        ):
            update, _ = cross_attn(
                query=query_norm(limb_tokens),
                key=memory_norm(structural_tokens),
                value=structural_tokens,
                need_weights=False,
            )
            limb_tokens = block(limb_tokens + update)
        return self.norm(limb_tokens)


class KeypointQueryDecoder(nn.Module):
    """Decode 17 pose-aware keypoint tokens from texture, structure, and limb memory."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_keypoints: int = 17,
        depth: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_keypoints, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(depth)]
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * mlp_ratio,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        keypoint_tokens = self.queries.expand(memory_tokens.shape[0], -1, -1)
        for query_norm, memory_norm, cross_attn, block in zip(
            self.query_norms,
            self.memory_norms,
            self.cross_attn,
            self.blocks,
        ):
            update, _ = cross_attn(
                query=query_norm(keypoint_tokens),
                key=memory_norm(memory_tokens),
                value=memory_tokens,
                need_weights=False,
            )
            keypoint_tokens = block(keypoint_tokens + update)
        return self.norm(keypoint_tokens)


class SkeletonGraphRefinement(nn.Module):
    """Refine keypoint tokens with fixed COCO skeleton connectivity."""

    def __init__(
        self,
        embed_dim: int,
        num_keypoints: int = 17,
        edges: Sequence[tuple[int, int]] = COCO_SKELETON,
        depth: int = 2,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        adjacency = torch.eye(num_keypoints, dtype=torch.float32)
        for start, end in edges:
            adjacency[start, end] = 1.0
            adjacency[end, start] = 1.0
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        self.gates = nn.ParameterList(
            [
                nn.Parameter(torch.full((), torch.logit(torch.tensor(gate_init)).item(), dtype=torch.float32))
                for _ in range(depth)
            ]
        )

    def forward(self, keypoint_tokens: torch.Tensor) -> torch.Tensor:
        for norm, projection, gate in zip(self.norms, self.projections, self.gates):
            aggregated = torch.einsum("ij,bjd->bid", self.adjacency.to(dtype=keypoint_tokens.dtype), keypoint_tokens)
            update = projection(norm(aggregated))
            keypoint_tokens = keypoint_tokens + torch.sigmoid(gate).to(dtype=update.dtype) * update
        return keypoint_tokens


class PoseAwareHeatmapDecoder(nn.Module):
    """Condition the ViTPose feature map and heatmaps on graph-refined keypoint tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_keypoints: int = 17,
        pose_gate_init: float = 0.05,
        residual_heatmap_init: float = 0.05,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.texture_norm = nn.LayerNorm(embed_dim)
        self.keypoint_norm = nn.LayerNorm(embed_dim)
        self.spatial_from_keypoints = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        pose_gate_init = min(max(pose_gate_init, 1e-4), 1.0 - 1e-4)
        residual_heatmap_init = min(max(residual_heatmap_init, 1e-4), 1.0 - 1e-4)
        self.pose_gate = nn.Parameter(torch.full((), torch.logit(torch.tensor(pose_gate_init)).item()))
        self.residual_heatmap_gate = nn.Parameter(
            torch.logit(torch.full((num_keypoints,), residual_heatmap_init))
        )
        self.scale = embed_dim**-0.5
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        texture_tokens: torch.Tensor,
        keypoint_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        heatmap_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, _ = self.spatial_from_keypoints(
            query=self.texture_norm(texture_tokens),
            key=self.keypoint_norm(keypoint_tokens),
            value=keypoint_tokens,
            need_weights=False,
        )
        fused_tokens = texture_tokens + torch.sigmoid(self.pose_gate).to(dtype=update.dtype) * update
        fused_tokens = self.output_norm(fused_tokens)

        spatial = self.texture_norm(fused_tokens)
        keypoints = self.keypoint_norm(keypoint_tokens)
        residual = torch.einsum("bkd,bnd->bkn", keypoints, spatial) * self.scale
        residual = residual.reshape(residual.shape[0], residual.shape[1], grid_size[0], grid_size[1])
        residual = nn.functional.interpolate(residual, size=heatmap_size, mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.residual_heatmap_gate).view(1, -1, 1, 1).to(dtype=residual.dtype)
        return fused_tokens, gate * residual


class TiptVitPoseV4ForPoseEstimation(nn.Module):
    """Pose-aware TIPTv4 with intact ViTPose backbone, limb tokens, keypoint queries, and skeleton refinement."""

    def __init__(
        self,
        checkpoint: str = "usyd-community/vitpose-base-simple",
        structural_channels: Sequence[str] = ("sobel_x", "sobel_y", "magnitude"),
        stem_channels: int = 48,
        stem_depth: int = 3,
        shape_depth: int = 3,
        limb_depth: int = 2,
        keypoint_depth: int = 2,
        graph_depth: int = 2,
        shape_dropout: float = 0.0,
        num_heads: int | None = None,
        gate_init: float = 0.1,
        pose_gate_init: float = 0.05,
        residual_heatmap_init: float = 0.05,
        mlp_ratio: int = 4,
        num_keypoints: int = 17,
    ) -> None:
        super().__init__()
        try:
            from transformers import VitPoseForPoseEstimation
        except ImportError as exc:
            raise ImportError(
                "TiptVitPoseV4ForPoseEstimation requires transformers. "
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
        self.num_keypoints = num_keypoints

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
        self.shape_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=shape_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=shape_depth,
            norm=nn.LayerNorm(embed_dim),
        )
        self.limb_decoder = PoseLimbTokenDecoder(
            embed_dim=embed_dim,
            num_heads=heads,
            num_limbs=len(COCO_SKELETON),
            depth=limb_depth,
            mlp_ratio=mlp_ratio,
            dropout=shape_dropout,
        )
        self.keypoint_decoder = KeypointQueryDecoder(
            embed_dim=embed_dim,
            num_heads=heads,
            num_keypoints=num_keypoints,
            depth=keypoint_depth,
            mlp_ratio=mlp_ratio,
            dropout=shape_dropout,
        )
        self.graph_refiner = SkeletonGraphRefinement(
            embed_dim=embed_dim,
            num_keypoints=num_keypoints,
            depth=graph_depth,
            gate_init=gate_init,
        )
        self.pose_decoder = PoseAwareHeatmapDecoder(
            embed_dim=embed_dim,
            num_heads=heads,
            num_keypoints=num_keypoints,
            pose_gate_init=pose_gate_init,
            residual_heatmap_init=residual_heatmap_init,
            dropout=shape_dropout,
        )

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
        yield from self.shape_encoder.parameters()
        yield from self.limb_decoder.parameters()
        yield from self.keypoint_decoder.parameters()
        yield from self.graph_refiner.parameters()
        yield from self.pose_decoder.parameters()

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
        return feature_maps[-1]

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
        shape_tokens = self.shape_encoder(self.shape_embed(shape_features))
        limb_tokens = self.limb_decoder(shape_tokens)

        memory_tokens = torch.cat([texture_tokens, shape_tokens, limb_tokens], dim=1)
        keypoint_tokens = self.keypoint_decoder(memory_tokens)
        keypoint_tokens = self.graph_refiner(keypoint_tokens)

        feature_map = tokens_to_feature_map(texture_tokens, self.grid_size)
        base_heatmaps = self.head(feature_map, flip_pairs=flip_pairs)
        fused_tokens, residual_heatmaps = self.pose_decoder(
            texture_tokens,
            keypoint_tokens,
            self.grid_size,
            heatmap_size=tuple(base_heatmaps.shape[-2:]),
        )
        heatmaps = base_heatmaps + residual_heatmaps

        return TiptVitPoseOutput(
            heatmaps=heatmaps,
            F_shape=keypoint_tokens,
            F_tex=texture_tokens,
            fused_tokens=fused_tokens,
            hidden_states=getattr(backbone_outputs, "hidden_states", None),
            attentions=getattr(backbone_outputs, "attentions", None),
        )
