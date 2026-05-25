from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


METRIC_KEYS = ("AP", "AP50", "AP75", "APM", "APL", "AR", "AR50", "AR75", "ARM", "ARL", "loss", "pck")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def odd_kernel_range(min_kernel: int, max_kernel: int) -> list[int]:
    if min_kernel > max_kernel:
        raise ValueError("--min-kernel cannot be greater than --max-kernel")
    if min_kernel % 2 == 0:
        min_kernel += 1
    if max_kernel % 2 == 0:
        max_kernel -= 1
    if min_kernel > max_kernel:
        raise ValueError("Kernel range contains no odd values")
    return list(range(min_kernel, max_kernel + 1, 2))


def default_tipt_checkpoint(project_root: Path) -> Path:
    latest_run_path = project_root / "runs" / "tipt_vitpose_v3" / "latest_run.txt"
    if not latest_run_path.exists():
        raise FileNotFoundError(
            f"Could not find {latest_run_path}. Pass --tipt-checkpoint explicitly."
        )
    run_dir = Path(latest_run_path.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Could not find TIPT checkpoint {checkpoint}")
    return checkpoint


def make_baseline_config(source_config: Path, output_config: Path) -> Path:
    cfg = load_yaml(source_config)
    cfg.setdefault("model", {})
    cfg["model"]["variant"] = "baseline"
    cfg["model"].setdefault("checkpoint", "usyd-community/vitpose-base-simple")
    save_yaml(output_config, cfg)
    return output_config


def run_eval(
    project_root: Path,
    config: Path,
    output_dir: Path,
    label: str,
    kernel_size: int,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    metrics_path = output_dir / f"{label}_blur_k{kernel_size:02d}_metrics.json"
    results_path = output_dir / f"{label}_blur_k{kernel_size:02d}_preds.json"
    cmd = [
        sys.executable,
        "eval_coco.py",
        "--config",
        str(config),
        "--obfuscation",
        "blur",
        "--blur-kernel-size",
        str(kernel_size),
        "--metrics-json",
        str(metrics_path),
        "--results-json",
        str(results_path),
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=project_root, check=True)

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload["metrics"]
    row = {"model": label, "blur_kernel_size": kernel_size}
    row.update({key: metrics.get(key) for key in METRIC_KEYS})
    return row


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)

    summary_csv = output_dir / "summary.csv"
    fieldnames = ["model", "blur_kernel_size", *METRIC_KEYS]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ViTPose-B baseline and TIPT-v3 across blur intensities.")
    parser.add_argument("--baseline-config", default="configs/tipt_vitpose_hf_coco.yaml")
    parser.add_argument("--tipt-config", default="configs/tipt_vitpose_v3_hf_coco.yaml")
    parser.add_argument("--tipt-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-kernel", type=int, default=3)
    parser.add_argument("--max-kernel", type=int, default=17)
    parser.add_argument("--kernels", type=int, nargs="+")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-tipt", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "runs" / "blur_sweeps" / timestamp
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels = args.kernels if args.kernels else odd_kernel_range(args.min_kernel, args.max_kernel)
    for kernel in kernels:
        if kernel % 2 == 0:
            raise ValueError(f"Blur kernel sizes must be odd, got {kernel}")

    baseline_config = make_baseline_config(
        project_root / args.baseline_config,
        output_dir / "configs" / "baseline_vitpose_b.yaml",
    )
    tipt_config = project_root / args.tipt_config
    tipt_checkpoint = Path(args.tipt_checkpoint) if args.tipt_checkpoint else default_tipt_checkpoint(project_root)
    if not tipt_checkpoint.is_absolute():
        tipt_checkpoint = project_root / tipt_checkpoint

    rows: list[dict[str, Any]] = []
    for kernel in kernels:
        if not args.skip_baseline:
            rows.append(run_eval(project_root, baseline_config, output_dir, "baseline_vitpose_b", kernel))
            write_summary(output_dir, rows)
        if not args.skip_tipt:
            rows.append(run_eval(project_root, tipt_config, output_dir, "tipt_v3", kernel, checkpoint=tipt_checkpoint))
            write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"Saved blur sweep summary to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
