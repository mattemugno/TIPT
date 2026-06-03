from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


COCO_KEYPOINT_METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APM",
    "APL",
    "AR",
    "AR50",
    "AR75",
    "ARM",
    "ARL",
)


@lru_cache(maxsize=4)
def _vitpose_image_processor(input_size: tuple[int, int]):
    try:
        from transformers import VitPoseImageProcessor
    except ImportError as exc:
        raise ImportError("HF ViTPose decoding requires transformers. Install requirements.txt first.") from exc

    input_h, input_w = input_size
    return VitPoseImageProcessor(size={"height": input_h, "width": input_w})


def heatmaps_to_image_keypoints(
    heatmaps: torch.Tensor,
    crop_boxes: torch.Tensor,
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode heatmap argmax locations back to original image coordinates."""

    if heatmaps.ndim != 4:
        raise ValueError(f"Expected [B, K, H, W] heatmaps, got {tuple(heatmaps.shape)}")

    batch_size, num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
    flat_heatmaps = heatmaps.reshape(batch_size, num_keypoints, -1)
    flat_indices = flat_heatmaps.argmax(dim=-1)
    scores = torch.sigmoid(flat_heatmaps.amax(dim=-1))

    heatmap_y = torch.div(flat_indices, heatmap_w, rounding_mode="floor").to(dtype=heatmaps.dtype)
    heatmap_x = (flat_indices % heatmap_w).to(dtype=heatmaps.dtype)

    input_h, input_w = input_size
    crop_boxes = crop_boxes.to(device=heatmaps.device, dtype=heatmaps.dtype)
    crop_x = crop_boxes[:, 0, None]
    crop_y = crop_boxes[:, 1, None]
    crop_w = crop_boxes[:, 2, None].clamp_min(1e-6)
    crop_h = crop_boxes[:, 3, None].clamp_min(1e-6)

    crop_point_x = heatmap_x * (input_w / heatmap_w)
    crop_point_y = heatmap_y * (input_h / heatmap_h)
    image_x = crop_x + crop_point_x * (crop_w / input_w)
    image_y = crop_y + crop_point_y * (crop_h / input_h)
    keypoints = torch.stack((image_x, image_y), dim=-1)
    return keypoints, scores


def batch_to_coco_keypoint_results(
    heatmaps: torch.Tensor,
    batch: dict[str, Any],
    input_size: tuple[int, int],
    decode_method: str = "hf",
    dark_kernel_size: int = 11,
) -> list[dict[str, Any]]:
    if decode_method == "hf":
        return batch_to_coco_keypoint_results_hf(
            heatmaps,
            batch,
            input_size=input_size,
            dark_kernel_size=dark_kernel_size,
        )
    if decode_method != "simple":
        raise ValueError(f"Unsupported decode_method {decode_method!r}; use 'hf' or 'simple'.")

    keypoints_xy, keypoint_scores = heatmaps_to_image_keypoints(
        heatmaps.detach().float().cpu(),
        batch["crop_box"].detach().float().cpu(),
        input_size,
    )
    bboxes = batch["bbox"].detach().float().cpu()
    image_ids = batch["image_id"].detach().cpu().tolist()

    results = []
    for idx, image_id in enumerate(image_ids):
        coco_keypoints: list[float] = []
        for point, score in zip(keypoints_xy[idx], keypoint_scores[idx], strict=True):
            coco_keypoints.extend([float(point[0]), float(point[1]), float(score)])

        mean_score = float(keypoint_scores[idx].mean())
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "keypoints": coco_keypoints,
                "score": mean_score,
                "bbox": [float(value) for value in bboxes[idx].tolist()],
            }
        )
    return results


def batch_to_coco_keypoint_results_hf(
    heatmaps: torch.Tensor,
    batch: dict[str, Any],
    input_size: tuple[int, int],
    dark_kernel_size: int = 11,
) -> list[dict[str, Any]]:
    """Decode heatmaps with the Hugging Face ViTPose DARK/unbiased post-processing."""

    processor = _vitpose_image_processor(input_size)
    bboxes = batch["bbox"].detach().float().cpu()
    boxes = [[bbox.tolist()] for bbox in bboxes]
    outputs = SimpleNamespace(heatmaps=heatmaps.detach().float().cpu())
    try:
        pose_batches = processor.post_process_pose_estimation(
            outputs,
            boxes=boxes,
            kernel_size=dark_kernel_size,
        )
    except NameError as exc:
        raise ImportError("HF ViTPose DARK decoding requires scipy. Install requirements.txt first.") from exc
    image_ids = batch["image_id"].detach().cpu().tolist()

    results = []
    for idx, image_id in enumerate(image_ids):
        pose = pose_batches[idx][0]
        keypoints_xy = pose["keypoints"].detach().float().cpu()
        keypoint_scores = pose["scores"].detach().float().cpu()

        coco_keypoints: list[float] = []
        for point, score in zip(keypoints_xy, keypoint_scores, strict=True):
            coco_keypoints.extend([float(point[0]), float(point[1]), float(score)])

        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "keypoints": coco_keypoints,
                "score": float(keypoint_scores.mean()),
                "bbox": [float(value) for value in bboxes[idx].tolist()],
            }
        )
    return results


def save_coco_results(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle)
    return path


def evaluate_coco_keypoints(
    annotation_file: str | Path,
    results: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Run official pycocotools OKS/AP metrics for COCO keypoints."""

    if not results:
        return {name: 0.0 for name in COCO_KEYPOINT_METRIC_NAMES}

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("COCO keypoint AP requires pycocotools. Install requirements.txt first.") from exc

    if output_path is None:
        output_path = Path("runs") / "tmp_coco_keypoint_results.json"
    results_path = save_coco_results(results, output_path)

    coco_gt = COCO(str(annotation_file))
    coco_dt = coco_gt.loadRes(str(results_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.params.imgIds = sorted({int(item["image_id"]) for item in results})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        name: float(value)
        for name, value in zip(COCO_KEYPOINT_METRIC_NAMES, coco_eval.stats.tolist(), strict=True)
    }
