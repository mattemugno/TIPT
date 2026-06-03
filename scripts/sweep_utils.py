from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import yaml


METRIC_KEYS = ("AP", "AP50", "AP75", "APM", "APL", "AR", "AR50", "AR75", "ARM", "ARL", "loss", "pck")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def slug_text(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe.strip("_") or "value"


def slug_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return slug_text(text)


def odd_kernel_range(min_kernel: int, max_kernel: int) -> list[int]:
    if min_kernel > max_kernel:
        raise ValueError("minimum blur kernel cannot be greater than maximum blur kernel")
    if min_kernel % 2 == 0:
        min_kernel += 1
    if max_kernel % 2 == 0:
        max_kernel -= 1
    if min_kernel > max_kernel:
        raise ValueError("blur kernel range contains no odd values")
    return list(range(min_kernel, max_kernel + 1, 2))


def build_obfuscation_settings(
    sweeps: Iterable[str],
    blur_kernels: Iterable[int],
    pixel_sizes: Iterable[int],
) -> list[tuple[str, str | None, int | None]]:
    settings: list[tuple[str, str | None, int | None]] = []
    requested = list(dict.fromkeys(sweeps))
    for sweep in requested:
        if sweep == "blur":
            for kernel in blur_kernels:
                if kernel % 2 == 0:
                    raise ValueError(f"blur kernel sizes must be odd, got {kernel}")
                settings.append(("blur", "blur_kernel_size", int(kernel)))
        elif sweep in {"pixelate", "pixelation"}:
            for pixel_size in pixel_sizes:
                if pixel_size < 1:
                    raise ValueError(f"pixel size must be positive, got {pixel_size}")
                settings.append(("pixelate", "pixel_size", int(pixel_size)))
        elif sweep in {"clean", "none"}:
            settings.append(("none", None, None))
        else:
            raise ValueError(f"unsupported sweep {sweep!r}; use blur, pixelate, or clean")
    return settings


def default_tipt_checkpoint(root: Path, run_root: str | Path = "runs/tipt_vitpose_v3") -> Path:
    latest_run_path = resolve_path(root, run_root) / "latest_run.txt"
    if not latest_run_path.exists():
        raise FileNotFoundError(
            f"Could not find {latest_run_path}. Pass --tipt-checkpoint explicitly."
        )
    run_dir = resolve_path(root, latest_run_path.read_text(encoding="utf-8").strip())
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Could not find TIPT checkpoint {checkpoint}")
    return checkpoint


def latest_checkpoint_from_output_root(root: Path, output_root: str | Path) -> Path:
    latest_run_path = resolve_path(root, output_root) / "latest_run.txt"
    if not latest_run_path.exists():
        raise FileNotFoundError(f"Could not find {latest_run_path}")
    run_dir = resolve_path(root, latest_run_path.read_text(encoding="utf-8").strip())
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Could not find checkpoint {checkpoint}")
    return checkpoint


def make_baseline_config(source_config: Path, output_config: Path) -> Path:
    cfg = load_yaml(source_config)
    cfg.setdefault("model", {})
    cfg["model"]["variant"] = "baseline"
    cfg["model"].setdefault("checkpoint", "usyd-community/vitpose-base-simple")
    save_yaml(output_config, cfg)
    return output_config


def run_subprocess(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def eval_filename(label: str, obfuscation: str, value_name: str | None, value: int | None, suffix: str) -> str:
    safe_label = slug_text(label)
    if obfuscation == "blur":
        stem = f"{safe_label}_blur_k{value:02d}"
    elif obfuscation == "pixelate":
        stem = f"{safe_label}_pixelate_px{value:02d}"
    else:
        stem = f"{safe_label}_clean"
    return f"{stem}_{suffix}.json"


def run_eval(
    root: Path,
    config: Path,
    output_dir: Path,
    label: str,
    obfuscation: str,
    value_name: str | None = None,
    value: int | None = None,
    checkpoint: Path | None = None,
    extra_row: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / eval_filename(label, obfuscation, value_name, value, "metrics")
    results_path = output_dir / eval_filename(label, obfuscation, value_name, value, "preds")
    cli_obfuscation = "none" if obfuscation in {"clean", "none"} else obfuscation
    cmd = [
        sys.executable,
        "eval_coco.py",
        "--config",
        str(config),
        "--obfuscation",
        cli_obfuscation,
        "--metrics-json",
        str(metrics_path),
        "--results-json",
        str(results_path),
    ]
    if cli_obfuscation == "blur":
        if value is None:
            raise ValueError("blur evaluation requires a kernel size")
        cmd.extend(["--blur-kernel-size", str(value)])
    elif cli_obfuscation == "pixelate":
        if value is None:
            raise ValueError("pixelation evaluation requires a pixel size")
        cmd.extend(["--pixel-size", str(value)])
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])

    run_subprocess(cmd, cwd=root, dry_run=dry_run)
    row: dict[str, Any] = {
        "model": label,
        "obfuscation": "clean" if cli_obfuscation == "none" else cli_obfuscation,
        "intensity": value,
    }
    if value_name is not None:
        row[value_name] = value
    if checkpoint is not None:
        row["checkpoint"] = str(checkpoint)
    if extra_row:
        row.update(extra_row)
    if dry_run:
        return row

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload["metrics"]
    row.update({key: metrics.get(key) for key in METRIC_KEYS})
    return row


def _summary_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "model",
        "invariance_weight",
        "obfuscation",
        "intensity",
        "blur_kernel_size",
        "pixel_size",
        "checkpoint",
        *METRIC_KEYS,
    ]
    fieldnames = [key for key in preferred if any(key in row for row in rows)]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    return [*fieldnames, *extras]


def write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"rows": rows}
    if metadata is not None:
        payload["metadata"] = metadata

    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary_csv = output_dir / "summary.csv"
    fieldnames = _summary_fieldnames(rows) if rows else ["model", "obfuscation", "intensity", *METRIC_KEYS]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
