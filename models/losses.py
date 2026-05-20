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
