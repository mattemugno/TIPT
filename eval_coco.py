from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from coco_metrics import batch_to_coco_keypoint_results, evaluate_coco_keypoints
from data import CocoKeypointsTopDownDataset
from data.transforms import heatmap_pck
from models import TiptVitPoseForPoseEstimation, TiptVitPoseV2ForPoseEstimation, weighted_heatmap_mse_loss


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_heatmaps(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["heatmaps"]
    return outputs.heatmaps


def get_obfuscation_config(
    cfg: dict[str, Any],
    mode_override: str | None = None,
    blur_kernel_size: int | None = None,
    pixel_size: int | None = None,
) -> dict[str, Any]:
    data_cfg = cfg["data"]
    value = data_cfg.get("val_obfuscation", data_cfg.get("obfuscation", {"mode": "none"}))
    if value is None:
        obfuscation = {"mode": "none"}
    elif isinstance(value, str):
        obfuscation = {"mode": value}
    else:
        obfuscation = deepcopy(value)

    if mode_override is not None:
        obfuscation["mode"] = mode_override
        obfuscation["probability"] = 1.0
    if blur_kernel_size is not None:
        obfuscation["blur_kernel_size"] = blur_kernel_size
    if pixel_size is not None:
        obfuscation["pixel_size"] = pixel_size
    return obfuscation


def make_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    checkpoint = model_cfg.get("checkpoint", "usyd-community/vitpose-base-simple")
    if model_cfg.get("variant") == "baseline":
        from transformers import VitPoseForPoseEstimation

        return VitPoseForPoseEstimation.from_pretrained(checkpoint)
    if model_cfg.get("variant") in {"tipt_v2", "tipt_v3", "tipt_shape_residual", "tipt_multilevel"}:
        return TiptVitPoseV2ForPoseEstimation(
            checkpoint=checkpoint,
            structural_channels=tuple(model_cfg.get("structural_channels", ["sobel_x", "sobel_y", "magnitude"])),
            stem_channels=int(model_cfg.get("stem_channels", 32)),
            stem_depth=int(model_cfg.get("stem_depth", 3)),
            shape_depth=int(model_cfg.get("shape_depth", 4)),
            shape_dropout=float(model_cfg.get("shape_dropout", 0.0)),
            num_heads=model_cfg.get("num_heads"),
            gate_init=float(model_cfg.get("gate_init", 0.1)),
            gate_hidden_ratio=float(model_cfg.get("gate_hidden_ratio", 0.25)),
            mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
        )
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


def save_metrics(path: str | Path, metrics: dict[str, float], cfg: dict[str, Any], obfuscation: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics": metrics,
                "obfuscation": obfuscation,
                "config": cfg,
            },
            handle,
            indent=2,
        )


def model_metadata(model: torch.nn.Module, checkpoint: str | None) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "vitpose"):
        config = getattr(model.vitpose, "config", None)

    backbone_config = getattr(config, "backbone_config", None)
    metadata = {
        "model_class": model.__class__.__name__,
        "checkpoint_arg": checkpoint,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    if config is not None:
        metadata["hf_model_type"] = getattr(config, "model_type", None)
        metadata["use_simple_decoder"] = getattr(config, "use_simple_decoder", None)
        metadata["scale_factor"] = getattr(config, "scale_factor", None)
    if backbone_config is not None:
        metadata["backbone_model_type"] = getattr(backbone_config, "model_type", None)
        metadata["backbone_hidden_size"] = getattr(backbone_config, "hidden_size", None)
        metadata["backbone_num_hidden_layers"] = getattr(backbone_config, "num_hidden_layers", None)
        metadata["backbone_num_attention_heads"] = getattr(backbone_config, "num_attention_heads", None)
        metadata["backbone_image_size"] = getattr(backbone_config, "image_size", None)
        metadata["backbone_patch_size"] = getattr(backbone_config, "patch_size", None)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TIPT-ViTPose with heatmap proxy metrics.")
    parser.add_argument("--config", default="configs/tipt_vitpose_hf_coco.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--obfuscation", choices=["none", "blur", "pixelate", "random"])
    parser.add_argument("--blur-kernel-size", type=int)
    parser.add_argument("--pixel-size", type=int)
    parser.add_argument("--no-coco", action="store_true")
    parser.add_argument("--results-json", default="runs/eval_coco_keypoints.json")
    parser.add_argument("--metrics-json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("eval", {}).get("device", cfg.get("training", {}).get("device", "auto")))
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    obfuscation = get_obfuscation_config(
        cfg,
        mode_override=args.obfuscation,
        blur_kernel_size=args.blur_kernel_size,
        pixel_size=args.pixel_size,
    )
    dataset = CocoKeypointsTopDownDataset(
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
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    metadata = model_metadata(model, args.checkpoint)
    print(f"model_metadata={json.dumps(metadata, sort_keys=True)}")
    model.eval()

    total_loss = 0.0
    total_pck = 0.0
    batches = 0
    coco_results: list[dict[str, Any]] = []
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
            if not args.no_coco:
                coco_results.extend(
                    batch_to_coco_keypoint_results(
                        pred_heatmaps,
                        batch,
                        input_size=dataset.input_size,
                    )
                )

    metrics = {"loss": total_loss / max(batches, 1), "pck": total_pck / max(batches, 1)}
    if not args.no_coco:
        metrics.update(evaluate_coco_keypoints(dataset.annotation_file, coco_results, output_path=args.results_json))

    print(f"val_loss={metrics['loss']:.6f} val_pck={metrics['pck']:.4f}")
    if not args.no_coco:
        print(f"coco_AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} AP75={metrics['AP75']:.4f}")
    if args.metrics_json:
        save_metrics(args.metrics_json, metrics, cfg, obfuscation)
        with Path(args.metrics_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["model_metadata"] = metadata
        with Path(args.metrics_json).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
