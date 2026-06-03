from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import CocoKeypointsTopDownDataset
from eval_coco import get_heatmaps, load_config, make_model, maybe_dataset_index, resolve_device
from visualize_predictions import draw_pose, heatmaps_to_crop_points


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)


def obfuscation_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.obfuscation == "blur":
        return {"mode": "blur", "blur_kernel_size": args.blur_kernel_size}
    if args.obfuscation == "pixelate":
        return {"mode": "pixelate", "pixel_size": args.pixel_size}
    return {"mode": "none"}


def make_dataset(cfg: dict[str, Any], obfuscation: dict[str, Any]) -> CocoKeypointsTopDownDataset:
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
        obfuscation=obfuscation,
        deterministic_obfuscation=True,
        obfuscation_seed=int(data_cfg.get("obfuscation_seed", 0)),
        crop_method=str(data_cfg.get("crop_method", "vitpose")),
    )


def denormalize_crop(pixel_values: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=pixel_values.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=pixel_values.dtype).view(3, 1, 1)
    array = (pixel_values.cpu() * std + mean).clamp(0, 1).numpy()
    array = np.asarray(array.transpose(1, 2, 0) * 255.0, dtype=np.uint8)
    return Image.fromarray(array)


def predict_points(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
    crop_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    dataset_index = maybe_dataset_index(1, cfg, device)
    with torch.no_grad():
        if dataset_index is None:
            outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)
    heatmaps = get_heatmaps(outputs)[0].detach().float().cpu()
    return heatmaps_to_crop_points(heatmaps, crop_size)


def add_label(image: Image.Image, label: str) -> Image.Image:
    label_h = 28
    canvas = Image.new("RGB", (image.width, image.height + label_h), (248, 248, 248))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), label, fill=(20, 20, 20), font=font)
    return canvas


def make_panel(columns: list[tuple[str, Image.Image]], gap: int = 10) -> Image.Image:
    labeled = [add_label(image, label) for label, image in columns]
    width = sum(image.width for image in labeled) + gap * (len(labeled) - 1)
    height = max(image.height for image in labeled)
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for image in labeled:
        panel.paste(image, (x, 0))
        x += image.width + gap
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ViTPose-B and TIPT-v3 predictions on obfuscated COCO crops.")
    parser.add_argument("--baseline-config", default="configs/vitpose_b_baseline_hf_coco.yaml")
    parser.add_argument("--tipt-config", default="configs/tipt_vitpose_v3_hf_coco.yaml")
    parser.add_argument("--tipt-checkpoint", required=True)
    parser.add_argument("--output-dir", default="reports/figures/inference_examples")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--obfuscation", choices=["blur", "pixelate", "none"], default="blur")
    parser.add_argument("--blur-kernel-size", type=int, default=15)
    parser.add_argument("--pixel-size", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()

    baseline_cfg = load_config(args.baseline_config)
    tipt_cfg = load_config(args.tipt_config)
    device = resolve_device(tipt_cfg.get("eval", {}).get("device", "auto"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(tipt_cfg, obfuscation_config(args))
    baseline_model = make_model(baseline_cfg).to(device)
    tipt_model = make_model(tipt_cfg).to(device)
    load_checkpoint(tipt_model, args.tipt_checkpoint, device)
    baseline_model.eval()
    tipt_model.eval()

    obf_label = args.obfuscation
    if args.obfuscation == "blur":
        obf_label = f"blur k={args.blur_kernel_size}"
    elif args.obfuscation == "pixelate":
        obf_label = f"pixel size={args.pixel_size}"

    for index in args.indices:
        sample = dataset[index]
        crop = denormalize_crop(sample["pixel_values"])
        gt_points = sample["keypoints"].float()
        gt_visibility = sample["visibility"].float()

        baseline_points, baseline_scores = predict_points(
            baseline_model,
            baseline_cfg,
            sample,
            device,
            dataset.input_size,
        )
        tipt_points, tipt_scores = predict_points(
            tipt_model,
            tipt_cfg,
            sample,
            device,
            dataset.input_size,
        )

        baseline_visual = draw_pose(
            crop,
            baseline_points,
            baseline_scores,
            gt_points=gt_points,
            gt_visibility=gt_visibility,
            score_threshold=args.score_threshold,
        )
        tipt_visual = draw_pose(
            crop,
            tipt_points,
            tipt_scores,
            gt_points=gt_points,
            gt_visibility=gt_visibility,
            score_threshold=args.score_threshold,
        )
        gt_visual = draw_pose(
            crop,
            gt_points,
            torch.ones(gt_points.shape[0]),
            gt_points=gt_points,
            gt_visibility=gt_visibility,
            score_threshold=args.score_threshold,
        )
        panel = make_panel(
            [
                (f"Input ({obf_label})", crop),
                ("ViTPose-B", baseline_visual),
                ("TIPT-v3", tipt_visual),
                ("Ground truth", gt_visual),
            ]
        )
        image_id = int(sample["image_id"])
        ann_id = int(sample["annotation_id"])
        output_path = output_dir / f"{args.obfuscation}_sample_{index:06d}_image_{image_id}_ann_{ann_id}.png"
        panel.save(output_path)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
