from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from sweep_utils import (
    build_obfuscation_settings,
    default_tipt_checkpoint,
    latest_checkpoint_from_output_root,
    make_baseline_config,
    odd_kernel_range,
    project_root,
    resolve_path,
    run_eval,
    utc_timestamp,
    write_summary,
)


def parse_args(
    default_sweeps: Sequence[str],
    default_output_subdir: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ViTPose-B baseline, fine-tuned ViTPose-B, and TIPT across clean/blur/pixelation settings."
    )
    parser.add_argument("--baseline-config", default="configs/vitpose_b_baseline_hf_coco.yaml")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--finetuned-config", default="configs/vitpose_b_finetune_hf_coco.yaml")
    parser.add_argument("--finetuned-checkpoint")
    parser.add_argument("--finetuned-run-root", default="runs/vitpose_b_finetune")
    parser.add_argument("--tipt-config", default="configs/tipt_vitpose_v3_hf_coco.yaml")
    parser.add_argument("--tipt-checkpoint")
    parser.add_argument("--tipt-run-root", default="runs/tipt_vitpose_v3")
    parser.add_argument("--baseline-label", default="baseline_vitpose_b")
    parser.add_argument("--finetuned-label", default="vitpose_b_finetuned")
    parser.add_argument("--tipt-label", default="tipt_v3")
    parser.add_argument("--output-dir")
    parser.add_argument("--sweeps", nargs="+", choices=["blur", "pixelate", "clean"], default=list(default_sweeps))
    parser.add_argument("--blur-min-kernel", "--min-kernel", dest="blur_min_kernel", type=int, default=3)
    parser.add_argument("--blur-max-kernel", "--max-kernel", dest="blur_max_kernel", type=int, default=17)
    parser.add_argument("--blur-kernels", "--kernels", dest="blur_kernels", type=int, nargs="+")
    parser.add_argument("--pixel-sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--skip-tipt", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots after the sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print eval commands without executing them.")
    parser.set_defaults(default_output_subdir=default_output_subdir)
    return parser.parse_args()


def main(
    default_sweeps: Sequence[str] = ("blur", "pixelate", "clean"),
    default_output_subdir: str = "obfuscation_sweeps",
) -> None:
    args = parse_args(default_sweeps, default_output_subdir)
    root = project_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "runs" / args.default_output_subdir / utc_timestamp()
    output_dir = resolve_path(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blur_kernels = args.blur_kernels or odd_kernel_range(args.blur_min_kernel, args.blur_max_kernel)
    settings = build_obfuscation_settings(args.sweeps, blur_kernels, args.pixel_sizes)

    baseline_config = None
    if not args.skip_baseline:
        baseline_config = make_baseline_config(
            resolve_path(root, args.baseline_config),
            output_dir / "configs" / "baseline_vitpose_b.yaml",
        )
    baseline_checkpoint = resolve_path(root, args.baseline_checkpoint) if args.baseline_checkpoint else None

    finetuned_config = None
    if not args.skip_finetuned:
        finetuned_config = make_baseline_config(
            resolve_path(root, args.finetuned_config),
            output_dir / "configs" / "vitpose_b_finetuned.yaml",
        )
    finetuned_checkpoint = None
    if finetuned_config is not None:
        finetuned_checkpoint = (
            resolve_path(root, args.finetuned_checkpoint)
            if args.finetuned_checkpoint
            else latest_checkpoint_from_output_root(root, args.finetuned_run_root)
        )

    tipt_config = resolve_path(root, args.tipt_config)
    tipt_checkpoint = None
    if not args.skip_tipt:
        tipt_checkpoint = (
            resolve_path(root, args.tipt_checkpoint)
            if args.tipt_checkpoint
            else default_tipt_checkpoint(root, args.tipt_run_root)
        )

    metadata = {
        "sweeps": list(args.sweeps),
        "blur_kernels": list(blur_kernels),
        "pixel_sizes": list(args.pixel_sizes),
        "baseline_config": str(resolve_path(root, args.baseline_config)),
        "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
        "finetuned_config": str(resolve_path(root, args.finetuned_config)),
        "finetuned_checkpoint": str(finetuned_checkpoint) if finetuned_checkpoint else None,
        "tipt_config": str(tipt_config),
        "tipt_checkpoint": str(tipt_checkpoint) if tipt_checkpoint else None,
    }
    rows: list[dict[str, object]] = []
    write_summary(output_dir, rows, metadata)

    for obfuscation, value_name, value in settings:
        if baseline_config is not None:
            rows.append(
                run_eval(
                    root,
                    baseline_config,
                    output_dir,
                    args.baseline_label,
                    obfuscation,
                    value_name,
                    value,
                    checkpoint=baseline_checkpoint,
                    dry_run=args.dry_run,
                )
            )
            write_summary(output_dir, rows, metadata)
        if finetuned_config is not None:
            rows.append(
                run_eval(
                    root,
                    finetuned_config,
                    output_dir,
                    args.finetuned_label,
                    obfuscation,
                    value_name,
                    value,
                    checkpoint=finetuned_checkpoint,
                    dry_run=args.dry_run,
                )
            )
            write_summary(output_dir, rows, metadata)
        if tipt_checkpoint is not None:
            rows.append(
                run_eval(
                    root,
                    tipt_config,
                    output_dir,
                    args.tipt_label,
                    obfuscation,
                    value_name,
                    value,
                    checkpoint=tipt_checkpoint,
                    dry_run=args.dry_run,
                )
            )
            write_summary(output_dir, rows, metadata)

    write_summary(output_dir, rows, metadata)
    print(f"Saved sweep summary to {output_dir / 'summary.csv'}")

    if args.plot and not args.dry_run:
        try:
            subprocess.run(
                [sys.executable, "scripts/plot_sweep.py", str(output_dir / "summary.json")],
                cwd=root,
                check=True,
            )
        except subprocess.CalledProcessError:
            print("Plot generation failed; the sweep summary was still saved.", file=sys.stderr)


if __name__ == "__main__":
    main()
