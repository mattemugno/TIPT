from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

from sweep_utils import (
    build_obfuscation_settings,
    latest_checkpoint_from_output_root,
    load_yaml,
    make_baseline_config,
    odd_kernel_range,
    project_root,
    resolve_path,
    run_eval,
    run_subprocess,
    save_yaml,
    slug_float,
    utc_timestamp,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TIPT-v3 with multiple shape-invariance weights and evaluate each checkpoint."
    )
    parser.add_argument("--base-config", default="configs/tipt_vitpose_v3_hf_coco.yaml")
    parser.add_argument("--baseline-config", default="configs/vitpose_b_baseline_hf_coco.yaml")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--output-dir")
    parser.add_argument("--sweeps", nargs="+", choices=["blur", "pixelate", "clean"], default=["blur", "pixelate", "clean"])
    parser.add_argument("--blur-min-kernel", "--min-kernel", dest="blur_min_kernel", type=int, default=3)
    parser.add_argument("--blur-max-kernel", "--max-kernel", dest="blur_max_kernel", type=int, default=17)
    parser.add_argument("--blur-kernels", "--kernels", dest="blur_kernels", type=int, nargs="+")
    parser.add_argument("--pixel-sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Generate plots from the combined summary after evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print commands without training/evaluating.")
    return parser.parse_args()


def weight_config(
    base_cfg: dict[str, Any],
    suite_dir: Path,
    weight: float,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    train_cfg = cfg.setdefault("training", {})
    invariance_cfg = train_cfg.setdefault("invariance", {})
    invariance_cfg["enabled"] = True
    invariance_cfg["weight"] = float(weight)

    suffix = f"w{slug_float(weight)}"
    base_run_name = str(train_cfg.get("run_name", "tipt_v3_shape_invariance"))
    train_cfg["run_name"] = f"{base_run_name}_{suffix}"
    train_cfg["output_dir"] = str(suite_dir / "training" / suffix)
    return cfg


def run_training(root: Path, config_path: Path, dry_run: bool) -> None:
    run_subprocess([sys.executable, "train.py", "--config", str(config_path)], cwd=root, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    root = project_root()
    suite_dir = Path(args.output_dir) if args.output_dir else root / "runs" / "invariance_weight_sweeps" / utc_timestamp()
    suite_dir = resolve_path(root, suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = resolve_path(root, args.base_config)
    base_cfg = load_yaml(base_config_path)
    blur_kernels = args.blur_kernels or odd_kernel_range(args.blur_min_kernel, args.blur_max_kernel)
    settings = build_obfuscation_settings(args.sweeps, blur_kernels, args.pixel_sizes)
    metadata = {
        "base_config": str(base_config_path),
        "baseline_config": str(resolve_path(root, args.baseline_config)),
        "weights": list(args.weights),
        "sweeps": list(args.sweeps),
        "blur_kernels": list(blur_kernels),
        "pixel_sizes": list(args.pixel_sizes),
    }
    rows: list[dict[str, Any]] = []
    write_summary(suite_dir, rows, metadata)

    if not args.no_eval and not args.skip_baseline_eval:
        baseline_config = make_baseline_config(
            resolve_path(root, args.baseline_config),
            suite_dir / "configs" / "baseline_vitpose_b.yaml",
        )
        for obfuscation, value_name, value in settings:
            rows.append(
                run_eval(
                    root,
                    baseline_config,
                    suite_dir / "eval",
                    "baseline_vitpose_b",
                    obfuscation,
                    value_name,
                    value,
                    extra_row={"invariance_weight": None},
                    dry_run=args.dry_run,
                )
            )
            write_summary(suite_dir, rows, metadata)

    for weight in args.weights:
        suffix = f"w{slug_float(weight)}"
        cfg = weight_config(base_cfg, suite_dir, weight)
        config_path = suite_dir / "configs" / f"tipt_v3_{suffix}.yaml"
        save_yaml(config_path, cfg)

        run_training(root, config_path, args.dry_run)
        if args.no_eval or args.dry_run:
            write_summary(suite_dir, rows, metadata)
            continue

        checkpoint = latest_checkpoint_from_output_root(root, cfg["training"]["output_dir"])
        label = f"tipt_v3_{suffix}"
        for obfuscation, value_name, value in settings:
            rows.append(
                run_eval(
                    root,
                    config_path,
                    suite_dir / "eval",
                    label,
                    obfuscation,
                    value_name,
                    value,
                    checkpoint=checkpoint,
                    extra_row={"invariance_weight": float(weight)},
                )
            )
            write_summary(suite_dir, rows, metadata)

    write_summary(suite_dir, rows, metadata)
    print(f"Saved weight sweep summary to {suite_dir / 'summary.csv'}")

    if args.plot and not args.no_eval and not args.dry_run:
        try:
            subprocess.run(
                [sys.executable, "scripts/plot_sweep.py", str(suite_dir / "summary.json")],
                cwd=root,
                check=True,
            )
        except subprocess.CalledProcessError:
            print("Plot generation failed; the weight sweep summary was still saved.", file=sys.stderr)


if __name__ == "__main__":
    main()
