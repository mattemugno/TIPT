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


def make_dataset(cfg: dict[str, Any], split: str) -> CocoKeypointsTopDownDataset:
    data_cfg = cfg["data"]
    prefix = "train" if split == "train" else "val"
    return CocoKeypointsTopDownDataset(
        image_root=data_cfg[f"{prefix}_image_dir"],
        annotation_file=data_cfg[f"{prefix}_annotations"],
        input_size=tuple(data_cfg.get("input_size", [256, 192])),
        heatmap_size=tuple(data_cfg.get("heatmap_size", [64, 48])),
        sigma=float(data_cfg.get("sigma", 2.0)),
        bbox_padding=float(data_cfg.get("bbox_padding", 1.25)),
        skip_empty=bool(data_cfg.get("skip_empty", True)),
        max_samples=data_cfg.get(f"{prefix}_max_samples"),
    )


def make_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    checkpoint = model_cfg.get("checkpoint", "usyd-community/vitpose-base-simple")
    variant = model_cfg.get("variant", "tipt_cross_attention")
    if variant == "baseline":
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


def make_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    if hasattr(model, "new_parameters") and hasattr(model, "pretrained_parameters"):
        param_groups = [
            {"params": list(model.new_parameters()), "lr": float(train_cfg.get("new_lr", 1e-4))},
            {"params": list(model.pretrained_parameters()), "lr": float(train_cfg.get("pretrained_lr", 1e-5))},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": float(train_cfg.get("pretrained_lr", 1e-5))}]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def resolve_device(value: str | None) -> torch.device:
    if value in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def maybe_dataset_index(batch_size: int, cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    value = cfg.get("model", {}).get("dataset_index")
    if value is None:
        return None
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_pck = 0.0
    total_batches = 0
    log_every = int(cfg.get("training", {}).get("log_every", 25))

    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        target_heatmaps = batch["target_heatmaps"].to(device, non_blocking=True)
        target_weights = batch["target_weights"].to(device, non_blocking=True)
        dataset_index = maybe_dataset_index(pixel_values.shape[0], cfg, device)

        with torch.set_grad_enabled(is_train):
            if dataset_index is None:
                outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)
            pred_heatmaps = get_heatmaps(outputs)
            loss = weighted_heatmap_mse_loss(pred_heatmaps, target_heatmaps, target_weights)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                max_norm = cfg.get("training", {}).get("grad_clip_norm")
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
                optimizer.step()

        pck = heatmap_pck(pred_heatmaps, target_heatmaps, target_weights, cfg.get("eval", {}).get("pck_threshold", 0.05))
        total_loss += float(loss.detach().cpu())
        total_pck += float(pck.detach().cpu())
        total_batches += 1

        if is_train and step % log_every == 0:
            print(f"epoch={epoch} step={step} loss={total_loss / total_batches:.6f} pck={total_pck / total_batches:.4f}")

    return {"loss": total_loss / max(total_batches, 1), "pck": total_pck / max(total_batches, 1)}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TIPT-ViTPose on COCO keypoint crops.")
    parser.add_argument("--config", default="configs/tipt_vitpose_hf_coco.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))
    train_cfg = cfg.get("training", {})

    train_dataset = make_dataset(cfg, "train")
    val_dataset = make_dataset(cfg, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    model = make_model(cfg).to(device)
    freeze_epochs = int(train_cfg.get("freeze_pretrained_epochs", 0))
    if hasattr(model, "set_pretrained_requires_grad") and freeze_epochs > 0:
        model.set_pretrained_requires_grad(False)

    optimizer = make_optimizer(model, cfg)
    output_dir = Path(train_cfg.get("output_dir", "runs/tipt_vitpose"))
    best_pck = -1.0

    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        if hasattr(model, "set_pretrained_requires_grad") and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.set_pretrained_requires_grad(True)

        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer, epoch)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, cfg, device, optimizer=None, epoch=epoch)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_pck={val_metrics['pck']:.4f}"
        )

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, cfg, val_metrics)
        if val_metrics["pck"] > best_pck:
            best_pck = val_metrics["pck"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, val_metrics)


if __name__ == "__main__":
    main()
