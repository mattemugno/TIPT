from __future__ import annotations

from eval_obfuscation_sweep import main as run_obfuscation_sweep


def main() -> None:
    run_obfuscation_sweep(default_sweeps=("blur",), default_output_subdir="blur_sweeps")


if __name__ == "__main__":
    main()
