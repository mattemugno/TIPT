from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from coco_metrics import batch_to_coco_keypoint_results, evaluate_coco_keypoints
from data import CocoKeypointsTopDownDataset
from data.transforms import heatmap_pck
from models import (
    TiptVitPoseForPoseEstimation,
    TiptVitPoseV3ForPoseEstimation,
    TiptVitPoseV4ForPoseEstimation,
    shape_invariance_loss,
    weighted_heatmap_mse_loss,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_heatmaps(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["heatmaps"]
    return outputs.heatmaps


def get_obfuscation_config(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    data_cfg = cfg["data"]
    value = data_cfg.get(f"{split}_obfuscation", data_cfg.get("obfuscation", {"mode": "none"}))
    if value is None:
        return {"mode": "none"}
    if isinstance(value, str):
        return {"mode": value}
    return deepcopy(value)


def get_invariance_views_config(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    data_cfg = cfg["data"]
    if split != "train":
        return {"enabled": False}
    value = data_cfg.get("train_invariance_views", {"enabled": False})
    return deepcopy(value) if value is not None else {"enabled": False}


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
        obfuscation=get_obfuscation_config(cfg, prefix),
        deterministic_obfuscation=prefix != "train",
        obfuscation_seed=int(data_cfg.get("obfuscation_seed", 0)),
        invariance_views=get_invariance_views_config(cfg, prefix),
        crop_method=str(data_cfg.get("crop_method", "pil")),
    )


def make_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    checkpoint = model_cfg.get("checkpoint", "usyd-community/vitpose-base-simple")
    variant = model_cfg.get("variant", "tipt_cross_attention")
    if variant == "baseline":
        from transformers import VitPoseForPoseEstimation

        return VitPoseForPoseEstimation.from_pretrained(checkpoint)

    if variant in {"tipt_v2", "tipt_v3"}:
        return TiptVitPoseV3ForPoseEstimation(
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

    if variant in {"tiptv4", "tipt_v4"}:
        return TiptVitPoseV4ForPoseEstimation(
            checkpoint=checkpoint,
            structural_channels=tuple(model_cfg.get("structural_channels", ["sobel_x", "sobel_y", "magnitude"])),
            stem_channels=int(model_cfg.get("stem_channels", 48)),
            stem_depth=int(model_cfg.get("stem_depth", 3)),
            shape_depth=int(model_cfg.get("shape_depth", 3)),
            limb_depth=int(model_cfg.get("limb_depth", 2)),
            keypoint_depth=int(model_cfg.get("keypoint_depth", 2)),
            graph_depth=int(model_cfg.get("graph_depth", 2)),
            shape_dropout=float(model_cfg.get("shape_dropout", 0.0)),
            num_heads=model_cfg.get("num_heads"),
            gate_init=float(model_cfg.get("gate_init", 0.1)),
            pose_gate_init=float(model_cfg.get("pose_gate_init", 0.05)),
            residual_heatmap_init=float(model_cfg.get("residual_heatmap_init", 0.05)),
            mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
            num_keypoints=int(model_cfg.get("num_keypoints", 17)),
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


def make_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    scope = str(train_cfg.get("trainable", "all")).lower()
    if hasattr(model, "new_parameters") and hasattr(model, "pretrained_parameters"):
        new_params = list(model.new_parameters())
        pretrained_params = list(model.pretrained_parameters())
        if scope not in {"all", "full"}:
            new_params = [parameter for parameter in new_params if parameter.requires_grad]
            pretrained_params = [parameter for parameter in pretrained_params if parameter.requires_grad]
        param_groups = [
            {"params": new_params, "lr": float(train_cfg.get("new_lr", 1e-4))},
            {"params": pretrained_params, "lr": float(train_cfg.get("pretrained_lr", 1e-5))},
        ]
        param_groups = [group for group in param_groups if group["params"]]
        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer.")
    else:
        params = list(model.parameters()) if scope in {"all", "full"} else [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found for optimizer.")
        param_groups = [{"params": params, "lr": float(train_cfg.get("pretrained_lr", 1e-5))}]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _resolve_module_attr(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = getattr(model, name, None)
    if module is None:
        raise ValueError(f"Model {model.__class__.__name__} has no module named {name!r}.")
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"Attribute {name!r} on {model.__class__.__name__} is not a torch module.")
    return module


def apply_trainable_scope(model: torch.nn.Module, cfg: dict[str, Any]) -> str:
    train_cfg = cfg.get("training", {})
    scope = str(train_cfg.get("trainable", "all")).lower()
    if scope in {"all", "full"}:
        return "all"

    if scope in {"head", "head_only", "decoder"}:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in _resolve_module_attr(model, "head").parameters():
            parameter.requires_grad = True
        return "head"

    if scope in {"new_plus_head", "shape_fusion_head", "tipt_fine_tuning"}:
        if not hasattr(model, "new_parameters"):
            raise ValueError(f"training.trainable={scope!r} requires a TIPT-style model with new_parameters().")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.new_parameters():
            parameter.requires_grad = True
        for parameter in _resolve_module_attr(model, "head").parameters():
            parameter.requires_grad = True
        return "new_plus_head"

    raise ValueError(f"Unsupported training.trainable={scope!r}. Use 'all', 'head', or 'new_plus_head'.")


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def resolve_device(value: str | None) -> torch.device:
    if value in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def maybe_dataset_index(batch_size: int, cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    value = cfg.get("model", {}).get("dataset_index")
    if value is None:
        return None
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)


def get_shape_tokens(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        shape_tokens = outputs.get("F_shape")
    else:
        shape_tokens = getattr(outputs, "F_shape", None)
    if shape_tokens is None:
        raise RuntimeError("Invariance training requires model outputs with F_shape.")
    return shape_tokens


def sanitize_run_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe.strip("_") or "run"


def create_run_dir(root: Path, cfg: dict[str, Any]) -> Path:
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = train_cfg.get("run_name") or model_cfg.get("variant", "run")
    base_name = f"{timestamp}_{sanitize_run_name(str(run_name))}"
    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (root / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    return run_dir


def save_run_summary(
    path: Path,
    cfg: dict[str, Any],
    run_dir: Path,
    history: list[dict[str, Any]],
    best: dict[str, Any],
    final_coco: dict[str, float] | None,
    timing: dict[str, Any],
    completed: bool,
    started_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed": completed,
                "config": cfg,
                "run_dir": str(run_dir),
                "best": best,
                "final_coco": final_coco,
                "timing": timing,
                "history": history,
            },
            handle,
            indent=2,
        )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    coco_results_path: Path | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_invariance_loss = 0.0
    total_pck = 0.0
    total_batches = 0
    coco_results: list[dict[str, Any]] = []
    log_every = int(cfg.get("training", {}).get("log_every", 25))
    invariance_cfg = cfg.get("training", {}).get("invariance", {})
    invariance_weight = float(invariance_cfg.get("weight", 0.0)) if is_train and invariance_cfg.get("enabled", False) else 0.0
    invariance_mode = str(invariance_cfg.get("mode", "cosine"))

    for step, batch in enumerate(loader, start=1):
        target_heatmaps = batch["target_heatmaps"].to(device, non_blocking=True)
        target_weights = batch["target_weights"].to(device, non_blocking=True)
        batch_size = target_heatmaps.shape[0]
        dataset_index = maybe_dataset_index(batch_size, cfg, device)
        use_invariance = (
            invariance_weight > 0
            and "pixel_values_view_a" in batch
            and "pixel_values_view_b" in batch
        )

        with torch.set_grad_enabled(is_train):
            if use_invariance:
                pixel_values_a = batch["pixel_values_view_a"].to(device, non_blocking=True)
                pixel_values_b = batch["pixel_values_view_b"].to(device, non_blocking=True)
                if dataset_index is None:
                    outputs_a = model(pixel_values=pixel_values_a)
                    outputs_b = model(pixel_values=pixel_values_b)
                else:
                    outputs_a = model(pixel_values=pixel_values_a, dataset_index=dataset_index)
                    outputs_b = model(pixel_values=pixel_values_b, dataset_index=dataset_index)

                pred_heatmaps = get_heatmaps(outputs_a)
                heatmap_loss_a = weighted_heatmap_mse_loss(pred_heatmaps, target_heatmaps, target_weights)
                heatmap_loss_b = weighted_heatmap_mse_loss(get_heatmaps(outputs_b), target_heatmaps, target_weights)
                heatmap_loss = 0.5 * (heatmap_loss_a + heatmap_loss_b)
                inv_loss = shape_invariance_loss(
                    get_shape_tokens(outputs_a),
                    get_shape_tokens(outputs_b),
                    mode=invariance_mode,
                )
                loss = heatmap_loss + invariance_weight * inv_loss
            else:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                if dataset_index is None:
                    outputs = model(pixel_values=pixel_values)
                else:
                    outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)
                pred_heatmaps = get_heatmaps(outputs)
                heatmap_loss = weighted_heatmap_mse_loss(pred_heatmaps, target_heatmaps, target_weights)
                inv_loss = pred_heatmaps.new_tensor(0.0)
                loss = heatmap_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                max_norm = cfg.get("training", {}).get("grad_clip_norm")
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
                optimizer.step()

        pck = heatmap_pck(pred_heatmaps, target_heatmaps, target_weights, cfg.get("eval", {}).get("pck_threshold", 0.05))
        total_loss += float(loss.detach().cpu())
        total_heatmap_loss += float(heatmap_loss.detach().cpu())
        total_invariance_loss += float(inv_loss.detach().cpu())
        total_pck += float(pck.detach().cpu())
        total_batches += 1
        if coco_results_path is not None:
            coco_results.extend(
                batch_to_coco_keypoint_results(
                    pred_heatmaps,
                    batch,
                    input_size=loader.dataset.input_size,
                    decode_method=cfg.get("eval", {}).get("decode", "hf"),
                    dark_kernel_size=int(cfg.get("eval", {}).get("dark_kernel_size", 11)),
                )
            )

        if is_train and step % log_every == 0:
            print(
                f"epoch={epoch} step={step} loss={total_loss / total_batches:.6f} "
                f"heatmap_loss={total_heatmap_loss / total_batches:.6f} "
                f"inv_loss={total_invariance_loss / total_batches:.6f} "
                f"pck={total_pck / total_batches:.4f}"
            )

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "heatmap_loss": total_heatmap_loss / max(total_batches, 1),
        "invariance_loss": total_invariance_loss / max(total_batches, 1),
        "pck": total_pck / max(total_batches, 1),
    }
    if coco_results_path is not None:
        metrics.update(
            evaluate_coco_keypoints(
                loader.dataset.annotation_file,
                coco_results,
                output_path=coco_results_path,
            )
        )
    return metrics


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
    trainable_scope = apply_trainable_scope(model, cfg)
    freeze_epochs = int(train_cfg.get("freeze_pretrained_epochs", 0))
    if trainable_scope != "all":
        freeze_epochs = 0
    if hasattr(model, "set_pretrained_requires_grad") and freeze_epochs > 0:
        model.set_pretrained_requires_grad(False)
    counts = parameter_counts(model)
    print(
        f"model_parameters={counts['total']} trainable_parameters={counts['trainable']} "
        f"trainable_scope={trainable_scope}"
    )

    optimizer = make_optimizer(model, cfg)
    output_root = Path(train_cfg.get("output_dir", "runs/tipt_vitpose"))
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = create_run_dir(output_root, cfg)
    print(f"run_dir={output_dir}")
    best_pck = -1.0
    history: list[dict[str, Any]] = []
    final_coco: dict[str, float] | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    train_start_time = time.perf_counter()
    timing: dict[str, Any] = {
        "total_seconds": None,
        "total_minutes": None,
        "epochs": [],
        "final_coco_seconds": None,
    }
    summary_path = output_dir / train_cfg.get("summary_file", "summary.json")
    best: dict[str, Any] = {
        "epoch": None,
        "metric": "pck",
        "value": best_pck,
        "checkpoint": str(output_dir / "best.pt"),
    }

    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        epoch_start_time = time.perf_counter()
        if hasattr(model, "set_pretrained_requires_grad") and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.set_pretrained_requires_grad(True)

        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer, epoch)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, cfg, device, optimizer=None, epoch=epoch)
        epoch_seconds = time.perf_counter() - epoch_start_time
        timing["epochs"].append({"epoch": epoch, "seconds": epoch_seconds, "minutes": epoch_seconds / 60.0})
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_pck={val_metrics['pck']:.4f} "
            f"time_min={epoch_seconds / 60.0:.2f}"
        )

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, cfg, val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["pck"] > best_pck:
            best_pck = val_metrics["pck"]
            best = {
                "epoch": epoch,
                "metric": "pck",
                "value": best_pck,
                "checkpoint": str(output_dir / "best.pt"),
                "metrics": val_metrics,
            }
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, val_metrics)
        elapsed = time.perf_counter() - train_start_time
        timing["total_seconds"] = elapsed
        timing["total_minutes"] = elapsed / 60.0
        save_run_summary(summary_path, cfg, output_dir, history, best, final_coco, timing, completed=False, started_at=started_at)

    if bool(train_cfg.get("final_coco_eval", True)):
        final_coco_start_time = time.perf_counter()
        best_checkpoint = output_dir / "best.pt"
        if best_checkpoint.exists():
            state = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(state["model"])
        with torch.no_grad():
            final_coco = run_epoch(
                model,
                val_loader,
                cfg,
                device,
                optimizer=None,
                epoch=int(best["epoch"] or 0),
                coco_results_path=output_dir / "coco_keypoints_val.json",
            )
        timing["final_coco_seconds"] = time.perf_counter() - final_coco_start_time
        print(
            f"final_coco AP={final_coco['AP']:.4f} AP50={final_coco['AP50']:.4f} "
            f"AP75={final_coco['AP75']:.4f}"
        )

    elapsed = time.perf_counter() - train_start_time
    timing["total_seconds"] = elapsed
    timing["total_minutes"] = elapsed / 60.0
    save_run_summary(summary_path, cfg, output_dir, history, best, final_coco, timing, completed=True, started_at=started_at)


if __name__ == "__main__":
    main()
