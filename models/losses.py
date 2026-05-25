from __future__ import annotations

import torch
import torch.nn.functional as F


def _broadcast_weights(target_weights: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
    if target_weights.ndim == 2:
        target_weights = target_weights[:, :, None, None]
    elif target_weights.ndim == 3:
        target_weights = target_weights[:, :, :, None]
    if target_weights.ndim != 4:
        raise ValueError(f"Expected target_weights with 2-4 dims, got {tuple(target_weights.shape)}")
    return target_weights.to(device=target_heatmaps.device, dtype=target_heatmaps.dtype)


def weighted_heatmap_mse_loss(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_weights: torch.Tensor,
    resize_target: bool = False,
) -> torch.Tensor:
    """Visibility-weighted MSE for COCO keypoint heatmaps."""

    if pred_heatmaps.ndim != 4 or target_heatmaps.ndim != 4:
        raise ValueError("pred_heatmaps and target_heatmaps must be [B, K, H, W]")

    target_heatmaps = target_heatmaps.to(device=pred_heatmaps.device, dtype=pred_heatmaps.dtype)
    if pred_heatmaps.shape[-2:] != target_heatmaps.shape[-2:]:
        if not resize_target:
            raise ValueError(
                "Predicted and target heatmaps have different spatial shapes: "
                f"{tuple(pred_heatmaps.shape[-2:])} vs {tuple(target_heatmaps.shape[-2:])}"
            )
        target_heatmaps = F.interpolate(
            target_heatmaps,
            size=pred_heatmaps.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    weights = _broadcast_weights(target_weights, target_heatmaps)
    return ((pred_heatmaps - target_heatmaps).square() * weights).mean()


def shape_invariance_loss(
    shape_a: torch.Tensor,
    shape_b: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """Encourage two obfuscated views of the same crop to share shape tokens."""

    if shape_a.shape != shape_b.shape:
        raise ValueError(f"Shape token tensors must match, got {tuple(shape_a.shape)} and {tuple(shape_b.shape)}")

    if mode == "cosine":
        norm_a = F.normalize(F.layer_norm(shape_a, shape_a.shape[-1:]), dim=-1)
        norm_b = F.normalize(F.layer_norm(shape_b, shape_b.shape[-1:]), dim=-1)
        return 1.0 - (norm_a * norm_b).sum(dim=-1).mean()
    if mode == "mse":
        norm_a = F.layer_norm(shape_a, shape_a.shape[-1:])
        norm_b = F.layer_norm(shape_b, shape_b.shape[-1:])
        return F.mse_loss(norm_a, norm_b)

    raise ValueError(f"Unsupported shape invariance loss mode {mode!r}; use cosine or mse.")
