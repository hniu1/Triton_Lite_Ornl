#!/usr/bin/env python3
"""Plot Stage-1 progress directly from a Slurm log, even after a timeout."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage1-current")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(
    r"epoch=(?P<epoch>\d+) train_loss=(?P<train_loss>[\d.eE+-]+) "
    r"val_loss=(?P<val_loss>[\d.eE+-]+) val_depth_wet_rmse=(?P<depth>[\d.eE+-]+) "
    r"val_component_rmse=(?P<component>[\d.eE+-]+) val_f1=(?P<f1>[\d.eE+-]+) "
    r"physical_score=(?P<physical>[\d.eE+-]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-diagnostic", type=Path, default=None)
    parser.add_argument("--current-diagnostic", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--previous-metrics", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [{k: (int(v) if k == "epoch" else float(v)) for k, v in match.groupdict().items()}
            for match in EPOCH_RE.finditer(args.log.read_text())]
    if not rows:
        raise ValueError(f"No completed epochs found in {args.log}")
    best = min(rows, key=lambda row: row["physical"])
    epochs = [row["epoch"] for row in rows]
    panels = [
        ("train_loss", "Training loss", "Loss"),
        ("val_loss", "Validation loss", "Loss"),
        ("depth", "Wet-depth RMSE", "m"),
        ("component", "Velocity-component RMSE", "m/s"),
        ("f1", "Wet/dry F1", "Score"),
        ("physical", "Physical selection score", "Score"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(epochs, [row[key] for row in rows], marker="o", color="#2878B5")
        ax.axvline(best["epoch"], color="#D55E00", linestyle="--", label=f"Selected epoch {best['epoch']}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.set_xticks(epochs[::max(1, len(epochs)//8)])
        style(ax)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Stage-1 maximum-performance training progress", fontsize=15, fontweight="bold")
    fig.savefig(output / "01_training_progress.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    with (output / "epoch_progress.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"completed_epochs": len(rows), "selected_epoch": best["epoch"], "selected_validation": best}

    if args.baseline_diagnostic and args.current_diagnostic:
        old = json.loads(args.baseline_diagnostic.read_text())
        new = json.loads(args.current_diagnostic.read_text())
        def categories(payload):
            f = payload["patch_fractions"]
            return [f.get("dry", 0), f.get("boundary", 0), f.get("partial_wet", 0)+f.get("mostly_wet", 0), f.get("deep", 0)]
        labels = ["Dry", "Boundary", "Wet", "Deep"]
        x = np.arange(4); width = .36
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        bars1=ax.bar(x-width/2, np.array(categories(old))*100, width, label="Original", color="#777777")
        bars2=ax.bar(x+width/2, np.array(categories(new))*100, width, label="Current", color="#2878B5")
        ax.bar_label(bars1, fmt="%.1f%%", padding=3); ax.bar_label(bars2, fmt="%.1f%%", padding=3)
        ax.set(xticks=x, xticklabels=labels, ylabel="Sampled patches (%)", title="Training sampler distribution")
        ax.legend(frameon=False); style(ax)
        fig.savefig(output / "02_sampler_comparison.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        summary["wet_cell_fraction_original"] = old["wet_cell_fraction"]
        summary["wet_cell_fraction_current"] = new["wet_cell_fraction"]

    if args.evaluation and args.evaluation.exists():
        summary["held_out_evaluation"] = json.loads(args.evaluation.read_text())

    if args.previous_metrics:
        previous = json.loads(args.previous_metrics.read_text())
        old_history = previous["history"]
        old_best = min(old_history, key=lambda row: row["val"]["physical_score"])
        old_epochs = [row["epoch"] for row in old_history]
        comparisons = [
            ("depth", "depth_wet_rmse", "Wet-depth RMSE", "m"),
            ("component", "component_rmse", "Velocity-component RMSE", "m/s"),
            ("f1", "wet_f1", "Wet/dry F1", "Score"),
            ("physical", "physical_score", "Physical selection score", "Score"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
        for ax, (new_key, old_key, title, ylabel) in zip(axes.flat, comparisons):
            ax.plot(old_epochs, [row["val"][old_key] for row in old_history], marker="o", markersize=3, color="#777777", label="Previous stratified")
            ax.plot(epochs, [row[new_key] for row in rows], marker="o", markersize=3, color="#2878B5", label="Current maximum-performance")
            ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
            style(ax)
        axes.flat[0].legend(frameon=False)
        fig.suptitle("Validation metrics: previous versus current", fontsize=15, fontweight="bold")
        fig.savefig(output / "03_previous_vs_current_validation.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        comparison_rows = []
        for new_key, old_key, _, _ in comparisons:
            old_value = float(old_best["val"][old_key])
            new_value = float(best[new_key])
            higher_better = new_key == "f1"
            improvement = 100 * ((new_value - old_value) if higher_better else (old_value - new_value)) / old_value
            comparison_rows.append({"metric": old_key, "previous": old_value, "current": new_value, "relative_improvement_percent": improvement})
        with (output / "previous_vs_current_validation.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader(); writer.writerows(comparison_rows)
        summary["previous_selected_epoch"] = old_best["epoch"]
        summary["previous_vs_current"] = comparison_rows
    (output / "progress_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote current-run plots to {output}")


if __name__ == "__main__":
    main()
