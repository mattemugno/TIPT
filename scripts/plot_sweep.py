from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    summary_path = path / "summary.json" if path.is_dir() else path
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["rows"]


def output_dir_for(summary_path: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    elif summary_path.is_dir():
        path = summary_path / "plots"
    else:
        path = summary_path.parent / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_names(rows: list[dict[str, Any]]) -> list[str]:
    names = sorted({str(row["model"]) for row in rows})
    return sorted(names, key=lambda name: (0 if "baseline" in name else 1, name))


def x_label_for(mode: str) -> str:
    if mode == "blur":
        return "blur kernel size"
    if mode == "pixelate":
        return "pixel size"
    return "condition"


def plot_metric(rows: list[dict[str, Any]], mode: str, metric: str, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    mode_rows = [row for row in rows if row.get("obfuscation") == mode and row.get(metric) is not None]
    if not mode_rows:
        return None

    plt.figure(figsize=(8, 5))
    if mode == "clean":
        names = model_names(mode_rows)
        values = []
        for name in names:
            model_values = [float(row[metric]) for row in mode_rows if row["model"] == name]
            values.append(model_values[-1] if model_values else float("nan"))
        plt.bar(names, values)
        plt.xticks(rotation=25, ha="right")
    else:
        for name in model_names(mode_rows):
            model_rows = [row for row in mode_rows if row["model"] == name and row.get("intensity") is not None]
            points = sorted((int(row["intensity"]), float(row[metric])) for row in model_rows)
            if points:
                xs, ys = zip(*points)
                plt.plot(xs, ys, marker="o", linewidth=2, label=name)
        plt.xlabel(x_label_for(mode))
        plt.legend()

    plt.ylabel(metric)
    plt.title(f"{metric} vs {x_label_for(mode)}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    output_path = output_dir / f"{mode}_{metric}.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from an obfuscation sweep summary.")
    parser.add_argument("summary", help="Path to summary.json or to a sweep directory containing summary.json.")
    parser.add_argument("--metrics", nargs="+", default=["AP"], help="Metrics to plot, e.g. AP AP50 AP75 AR.")
    parser.add_argument("--modes", nargs="+", default=["blur", "pixelate", "clean"])
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    rows = load_rows(summary_path)
    output_dir = output_dir_for(summary_path, args.output_dir)

    saved: list[Path] = []
    try:
        for mode in args.modes:
            for metric in args.metrics:
                output_path = plot_metric(rows, mode, metric, output_dir)
                if output_path is not None:
                    saved.append(output_path)
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it or run the sweep without --plot.") from exc

    if saved:
        for path in saved:
            print(f"saved {path}")
    else:
        print("No plots generated; check requested modes/metrics and summary rows.")


if __name__ == "__main__":
    main()
