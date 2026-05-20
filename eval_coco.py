from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from data import CocoKeypointsTopDownDataset
from data.transforms import heatmap_pck
from models import TiptVitPoseForPoseEstimation, weighted_heatmap_mse_loss


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_heatmaps(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["heatmaps"]
    return outputs.heatmaps


def make_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    checkpoint = model_cfg.get("checkpoint", "usyd-community/vitpose-base-simple")
    if model_cfg.get("variant") == "baseline":
        from transformers import VitPoseForPoseEstimation

        return VitPoseForPoseEstimation.from_pretrained(checkpoint)
    return TiptVitPoseForPoseEstimation(
        checkpoint=checkpoint,
        shape_depth=int(model_cfg.get("shape_depth", 4)),
        shape_dropout=float(model_cfg.get("shape_dropout", 0.0)),
        num_heads=model_cfg.get("num_heads"),
        fusion=model_cfg.get("fusion", "cross_attention"),
        structural_view=model_cfg.get("structural_view", "sobel"),
        alpha_init=float(model_cfg.get("alpha_init", 0.1)),
        mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
    )


def maybe_dataset_index(batch_size: int, cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    value = cfg.get("model", {}).get("dataset_index")
    if value is None:
        return None
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)


def resolve_device(value: str | None) -> torch.device:
    if value in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TIPT-ViTPose with heatmap proxy metrics.")
    parser.add_argument("--config", default="configs/tipt_vitpose_hf_coco.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("eval", {}).get("device", cfg.get("training", {}).get("device", "auto")))
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    dataset = CocoKeypointsTopDownDataset(
        image_root=data_cfg["val_image_dir"],
        annotation_file=data_cfg["val_annotations"],
        input_size=tuple(data_cfg.get("input_size", [256, 192])),
        heatmap_size=tuple(data_cfg.get("heatmap_size", [64, 48])),
        sigma=float(data_cfg.get("sigma", 2.0)),
        bbox_padding=float(data_cfg.get("bbox_padding", 1.25)),
        skip_empty=bool(data_cfg.get("skip_empty", True)),
        max_samples=data_cfg.get("val_max_samples"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", cfg.get("training", {}).get("eval_batch_size", 16))),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", cfg.get("training", {}).get("num_workers", 4))),
        pin_memory=device.type == "cuda",
    )

    model = make_model(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    total_loss = 0.0
    total_pck = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            target_heatmaps = batch["target_heatmaps"].to(device, non_blocking=True)
            target_weights = batch["target_weights"].to(device, non_blocking=True)
            dataset_index = maybe_dataset_index(pixel_values.shape[0], cfg, device)
            if dataset_index is None:
                outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)
            pred_heatmaps = get_heatmaps(outputs)
            loss = weighted_heatmap_mse_loss(pred_heatmaps, target_heatmaps, target_weights)
            pck = heatmap_pck(pred_heatmaps, target_heatmaps, target_weights, eval_cfg.get("pck_threshold", 0.05))
            total_loss += float(loss.cpu())
            total_pck += float(pck.cpu())
            batches += 1

    print(f"val_loss={total_loss / max(batches, 1):.6f} val_pck={total_pck / max(batches, 1):.4f}")


if __name__ == "__main__":
    main()
