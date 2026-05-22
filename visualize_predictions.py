from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from data import CocoKeypointsTopDownDataset
from data.transforms import aspect_ratio_box, batch_heatmap_argmax, crop_and_resize
from eval_coco import get_heatmaps, load_config, make_model, maybe_dataset_index, resolve_device


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
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def make_dataset(cfg: dict) -> CocoKeypointsTopDownDataset:
    data_cfg = cfg["data"]
    return CocoKeypointsTopDownDataset(
        image_root=data_cfg["val_image_dir"],
        annotation_file=data_cfg["val_annotations"],
        input_size=tuple(data_cfg.get("input_size", [256, 192])),
        heatmap_size=tuple(data_cfg.get("heatmap_size", [64, 48])),
        sigma=float(data_cfg.get("sigma", 2.0)),
        bbox_padding=float(data_cfg.get("bbox_padding", 1.25)),
        skip_empty=bool(data_cfg.get("skip_empty", True)),
        max_samples=data_cfg.get("val_max_samples"),
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)


def crop_for_index(dataset: CocoKeypointsTopDownDataset, index: int) -> Image.Image:
    ann = dataset.annotations[index]
    image_info = dataset.images[int(ann["image_id"])]
    image_path = dataset.image_root / image_info["file_name"]
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop_box = aspect_ratio_box(ann["bbox"], dataset.input_size, padding=dataset.bbox_padding)
        return crop_and_resize(image, crop_box, dataset.input_size)


def heatmaps_to_crop_points(heatmaps: torch.Tensor, crop_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected [K, H, W] heatmaps, got {tuple(heatmaps.shape)}")
    num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
    points = batch_heatmap_argmax(heatmaps.unsqueeze(0))[0]
    scores = heatmaps.reshape(num_keypoints, -1).amax(dim=-1)

    crop_h, crop_w = crop_size
    points = points.clone()
    points[:, 0] = points[:, 0] * (crop_w / heatmap_w)
    points[:, 1] = points[:, 1] * (crop_h / heatmap_h)
    return points.cpu(), scores.cpu()


def draw_pose(
    image: Image.Image,
    pred_points: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_points: torch.Tensor | None = None,
    gt_visibility: torch.Tensor | None = None,
    score_threshold: float = 0.0,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    pred_visible = pred_scores >= score_threshold
    for start, end in COCO_SKELETON:
        if pred_visible[start] and pred_visible[end]:
            draw.line(
                [
                    tuple(pred_points[start].tolist()),
                    tuple(pred_points[end].tolist()),
                ],
                fill=(255, 64, 64),
                width=3,
            )

    for idx, point in enumerate(pred_points):
        if not pred_visible[idx]:
            continue
        x, y = point.tolist()
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 40, 40), outline=(255, 255, 255))

    if gt_points is not None and gt_visibility is not None:
        gt_visible = gt_visibility > 0
        for start, end in COCO_SKELETON:
            if gt_visible[start] and gt_visible[end]:
                draw.line(
                    [
                        tuple(gt_points[start].tolist()),
                        tuple(gt_points[end].tolist()),
                    ],
                    fill=(40, 220, 120),
                    width=2,
                )
        for idx, point in enumerate(gt_points):
            if not gt_visible[idx]:
                continue
            x, y = point.tolist()
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(40, 220, 120))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Save visual TIPT/ViTPose predictions on COCO validation crops.")
    parser.add_argument("--config", default="configs/tipt_vitpose_hf_coco.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/tipt_vitpose/visuals")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--hide-targets", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("eval", {}).get("device", cfg.get("training", {}).get("device", "auto")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(cfg)
    model = make_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    end_index = min(args.start_index + args.num_samples, len(dataset))
    with torch.no_grad():
        for index in range(args.start_index, end_index):
            sample = dataset[index]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            dataset_index = maybe_dataset_index(1, cfg, device)
            if dataset_index is None:
                outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)

            heatmaps = get_heatmaps(outputs)[0].detach().float().cpu()
            pred_points, pred_scores = heatmaps_to_crop_points(heatmaps, dataset.input_size)
            crop = crop_for_index(dataset, index)
            gt_points = None if args.hide_targets else sample["keypoints"].float()
            gt_visibility = None if args.hide_targets else sample["visibility"].float()
            visual = draw_pose(
                crop,
                pred_points,
                pred_scores,
                gt_points=gt_points,
                gt_visibility=gt_visibility,
                score_threshold=args.score_threshold,
            )

            image_id = int(sample["image_id"])
            ann_id = int(sample["annotation_id"])
            visual.save(output_dir / f"sample_{index:06d}_image_{image_id}_ann_{ann_id}.png")

    print(f"Saved {end_index - args.start_index} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
