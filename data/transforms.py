from __future__ import annotations

from collections.abc import Sequence
import random

import numpy as np
from PIL import Image, ImageFilter
import torch

try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def to_hw(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"Expected [height, width], got {size!r}")
    return (int(size[0]), int(size[1]))


def aspect_ratio_box(
    bbox_xywh: Sequence[float],
    output_size: tuple[int, int],
    padding: float = 1.25,
) -> np.ndarray:
    x, y, width, height = [float(v) for v in bbox_xywh]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox with non-positive size: {bbox_xywh!r}")

    out_h, out_w = output_size
    target_ratio = out_w / out_h
    center_x = x + width * 0.5
    center_y = y + height * 0.5

    if width / height > target_ratio:
        height = width / target_ratio
    else:
        width = height * target_ratio

    width *= padding
    height *= padding
    return np.array([center_x - width * 0.5, center_y - height * 0.5, width, height], dtype=np.float32)


def crop_and_resize(image: Image.Image, crop_xywh: Sequence[float], output_size: tuple[int, int]) -> Image.Image:
    x, y, width, height = [float(v) for v in crop_xywh]
    crop = image.crop((x, y, x + width, y + height))
    out_h, out_w = output_size
    return crop.resize((out_w, out_h), BILINEAR)


def _sample_range(value: float | int | Sequence[float | int], rng: random.Random) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if len(value) != 2:
        raise ValueError(f"Expected scalar or [min, max], got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    return rng.uniform(lo, hi)


def _sample_int_range(value: int | Sequence[int], rng: random.Random) -> int:
    if isinstance(value, int):
        return value
    if len(value) != 2:
        raise ValueError(f"Expected scalar or [min, max], got {value!r}")
    lo, hi = int(value[0]), int(value[1])
    return rng.randint(lo, hi)


def _sample_odd_int_range(value: int | Sequence[int], rng: random.Random) -> int:
    if isinstance(value, int):
        return value
    if len(value) != 2:
        raise ValueError(f"Expected scalar or [min, max], got {value!r}")
    lo, hi = int(value[0]), int(value[1])
    if lo > hi:
        raise ValueError(f"Invalid range [{lo}, {hi}]")
    if lo % 2 == 0:
        lo += 1
    if hi % 2 == 0:
        hi -= 1
    if lo > hi:
        raise ValueError(f"Range {value!r} contains no odd kernel size")
    return rng.randrange(lo, hi + 1, 2)


def blur_kernel_to_radius(kernel_size: int) -> float:
    if kernel_size < 1:
        raise ValueError(f"blur_kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"blur_kernel_size must be odd, got {kernel_size}")
    return (kernel_size - 1) / 6.0


def apply_obfuscation(
    image: Image.Image,
    config: dict | None,
    rng: random.Random | None = None,
) -> Image.Image:
    """Apply blur/pixelation online to a PIL crop without saving a second dataset."""

    if not config:
        return image

    rng = rng or random
    mode = config.get("mode", "none")
    probability = float(config.get("probability", 1.0))
    if mode in {None, "none"} or probability <= 0 or rng.random() > probability:
        return image

    if mode == "random":
        modes = config.get("modes", ["none", "blur", "pixelate"])
        if not modes:
            return image
        mode = rng.choice(list(modes))
        if mode == "none":
            return image

    if mode == "blur":
        if "blur_kernel_size" in config:
            kernel_size = _sample_odd_int_range(config["blur_kernel_size"], rng)
            radius = blur_kernel_to_radius(kernel_size)
        else:
            radius = _sample_range(config.get("blur_radius", 3.0), rng)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    if mode in {"pixelate", "pixelation"}:
        pixel_size = _sample_int_range(config.get("pixel_size", config.get("pixelation_factor", 8)), rng)
        pixel_size = max(1, pixel_size)
        width, height = image.size
        small_size = (max(1, width // pixel_size), max(1, height // pixel_size))
        return image.resize(small_size, NEAREST).resize((width, height), NEAREST)

    raise ValueError(f"Unsupported obfuscation mode {mode!r}; use none, blur, pixelate, or random.")


def keypoints_to_crop(
    keypoints_xy: np.ndarray,
    crop_xywh: Sequence[float],
    output_size: tuple[int, int],
) -> np.ndarray:
    x, y, width, height = [float(v) for v in crop_xywh]
    out_h, out_w = output_size
    transformed = keypoints_xy.astype(np.float32).copy()
    transformed[:, 0] = (transformed[:, 0] - x) * (out_w / width)
    transformed[:, 1] = (transformed[:, 1] - y) * (out_h / height)
    return transformed


def normalize_image(
    image: Image.Image,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(array.transpose(2, 0, 1))
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def generate_keypoint_heatmaps(
    keypoints_xy: np.ndarray,
    visibility: np.ndarray,
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
    sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_h, input_w = input_size
    heatmap_h, heatmap_w = heatmap_size
    num_keypoints = keypoints_xy.shape[0]
    heatmaps = np.zeros((num_keypoints, heatmap_h, heatmap_w), dtype=np.float32)
    weights = np.zeros((num_keypoints, 1, 1), dtype=np.float32)

    radius = int(3 * sigma)
    gaussian_size = 2 * radius + 1
    xs = np.arange(gaussian_size, dtype=np.float32)
    ys = xs[:, None]
    center = radius
    gaussian = np.exp(-((xs - center) ** 2 + (ys - center) ** 2) / (2 * sigma**2))

    for keypoint_idx in range(num_keypoints):
        if visibility[keypoint_idx] <= 0:
            continue

        x, y = keypoints_xy[keypoint_idx]
        if x < 0 or y < 0 or x >= input_w or y >= input_h:
            continue

        mu_x = x * heatmap_w / input_w
        mu_y = y * heatmap_h / input_h
        ul_x = int(mu_x + 0.5) - radius
        ul_y = int(mu_y + 0.5) - radius
        br_x = ul_x + gaussian_size
        br_y = ul_y + gaussian_size

        if br_x <= 0 or br_y <= 0 or ul_x >= heatmap_w or ul_y >= heatmap_h:
            continue

        g_x0 = max(0, -ul_x)
        g_y0 = max(0, -ul_y)
        g_x1 = min(br_x, heatmap_w) - ul_x
        g_y1 = min(br_y, heatmap_h) - ul_y

        h_x0 = max(0, ul_x)
        h_y0 = max(0, ul_y)
        h_x1 = min(br_x, heatmap_w)
        h_y1 = min(br_y, heatmap_h)

        heatmaps[keypoint_idx, h_y0:h_y1, h_x0:h_x1] = gaussian[g_y0:g_y1, g_x0:g_x1]
        weights[keypoint_idx, 0, 0] = 1.0

    return torch.from_numpy(heatmaps), torch.from_numpy(weights)


def batch_heatmap_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    batch_size, num_keypoints, height, width = heatmaps.shape
    flat_indices = heatmaps.reshape(batch_size, num_keypoints, -1).argmax(dim=-1)
    y = torch.div(flat_indices, width, rounding_mode="floor")
    x = flat_indices % width
    return torch.stack((x, y), dim=-1).to(dtype=heatmaps.dtype)


def heatmap_pck(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_weights: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    pred_xy = batch_heatmap_argmax(pred_heatmaps.detach())
    target_xy = batch_heatmap_argmax(target_heatmaps.detach().to(pred_heatmaps.device))
    if target_weights.ndim == 4:
        visible = target_weights[:, :, 0, 0] > 0
    else:
        visible = target_weights.squeeze(-1) > 0
    visible = visible.to(pred_heatmaps.device)

    distances = torch.linalg.vector_norm(pred_xy - target_xy, dim=-1)
    limit = threshold * max(pred_heatmaps.shape[-2:])
    correct = (distances <= limit) & visible
    denom = visible.sum().clamp_min(1)
    return correct.sum().to(dtype=pred_heatmaps.dtype) / denom
