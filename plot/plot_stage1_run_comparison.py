#!/usr/bin/env python3
"""Compare baseline and improved Stage-1 runs with meeting-ready figures."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_COLOR = "#6B7280"
NEW_COLOR = "#2878B5"
GOOD_COLOR = "#2E8B57"
BAD_COLOR = "#C44E52"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--improved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def annotate(ax, bars, digits=3):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def grouped_bars(ax, baseline, improved, metrics, labels, title, ylabel, ylim=None):
    x = np.arange(len(metrics))
    width = 0.36
    old = ax.bar(
        x - width / 2,
        [baseline[key] for key in metrics],
        width,
        color=BASE_COLOR,
        label="Original",
    )
    new = ax.bar(
        x + width / 2,
        [improved[key] for key in metrics],
        width,
        color=NEW_COLOR,
        label="Stratified",
    )
    annotate(ax, old)
    annotate(ax, new)
    ax.set_xticks(x, labels)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    style_axis(ax)


def plot_test_comparison(baseline, improved, output_path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
    grouped_bars(
        axes[0],
        baseline,
        improved,
        ["depth_all_mae", "depth_all_rmse", "depth_wet_mae", "depth_wet_rmse"],
        ["All\nMAE", "All\nRMSE", "Wet\nMAE", "Wet\nRMSE"],
        "Depth error",
        "Error (m)",
    )
    grouped_bars(
        axes[1],
        baseline,
        improved,
        ["component_mae", "component_rmse"],
        ["MAE", "RMSE"],
        "Velocity-component error",
        "Error (m/s)",
    )
    grouped_bars(
        axes[2],
        baseline,
        improved,
        ["wet_precision", "wet_recall", "wet_f1", "wet_csi"],
        ["Precision", "Recall", "F1", "CSI"],
        "Wet/dry inundation skill",
        "Score",
        (0, 1.05),
    )
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle("Held-out D030: original vs stratified sampling", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def improvement_percent(key, baseline, improved):
    lower_is_better = key not in {"wet_precision", "wet_recall", "wet_f1", "wet_csi"}
    if lower_is_better:
        return 100.0 * (baseline[key] - improved[key]) / baseline[key]
    return 100.0 * (improved[key] - baseline[key]) / baseline[key]


def plot_relative_improvement(baseline, improved, output_path, dpi):
    metrics = [
        "loss",
        "depth_all_mae",
        "depth_all_rmse",
        "depth_wet_mae",
        "depth_wet_rmse",
        "component_mae",
        "component_rmse",
        "wet_precision",
        "wet_recall",
        "wet_f1",
        "wet_csi",
    ]
    labels = [
        "Loss",
        "All-depth MAE",
        "All-depth RMSE",
        "Wet-depth MAE",
        "Wet-depth RMSE",
        "Velocity MAE",
        "Velocity RMSE",
        "Wet precision",
        "Wet recall",
        "Wet F1",
        "Wet CSI",
    ]
    values = [improvement_percent(key, baseline, improved) for key in metrics]
    colors = [GOOD_COLOR if value >= 0 else BAD_COLOR for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    bars = ax.barh(np.arange(len(values)), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Improvement relative to original run (%)")
    ax.set_title("Held-out D030 relative improvement", fontsize=15, fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        offset = 4 if value >= 0 else -4
        ax.annotate(
            f"{value:+.1f}%",
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(base_history, new_history, output_path, dpi):
    panels = [
        ("depth_wet_rmse", "Wet-depth RMSE", "m", False),
        ("component_rmse", "Velocity-component RMSE", "m/s", False),
        ("wet_f1", "Wet/dry F1", "Score", True),
        ("wet_csi", "Wet/dry CSI", "Score", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    base_epochs = [row["epoch"] for row in base_history]
    new_epochs = [row["epoch"] for row in new_history]
    for ax, (metric, title, ylabel, _) in zip(axes.flat, panels):
        ax.plot(
            base_epochs,
            [row["val"][metric] for row in base_history],
            color=BASE_COLOR,
            marker="o",
            markersize=3.2,
            linewidth=1.7,
            label="Original",
        )
        ax.plot(
            new_epochs,
            [row["val"][metric] for row in new_history],
            color=NEW_COLOR,
            marker="o",
            markersize=3.2,
            linewidth=1.7,
            label="Stratified",
        )
        ax.axvline(20, color=NEW_COLOR, linestyle="--", alpha=0.65, linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Validation performance by epoch (dashed line: selected stratified checkpoint)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(baseline, improved, output_path):
    metrics = [
        "loss",
        "depth_all_mae",
        "depth_all_rmse",
        "depth_wet_mae",
        "depth_wet_rmse",
        "component_mae",
        "component_rmse",
        "wet_precision",
        "wet_recall",
        "wet_f1",
        "wet_csi",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric", "original", "stratified", "absolute_change", "relative_improvement_percent"]
        )
        for key in metrics:
            writer.writerow(
                [
                    key,
                    baseline[key],
                    improved[key],
                    improved[key] - baseline[key],
                    improvement_percent(key, baseline, improved),
                ]
            )


def main():
    args = parse_args()
    baseline_payload = json.loads((args.baseline_dir.resolve() / "metrics.json").read_text())
    improved_payload = json.loads((args.improved_dir.resolve() / "metrics.json").read_text())
    output_dir = (args.output_dir or args.improved_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = baseline_payload["test"]
    improved = improved_payload["test"]
    plot_test_comparison(
        baseline, improved, output_dir / "01_baseline_vs_stratified_test.png", args.dpi
    )
    plot_relative_improvement(
        baseline, improved, output_dir / "02_relative_improvement.png", args.dpi
    )
    plot_validation_curves(
        baseline_payload["history"],
        improved_payload["history"],
        output_dir / "03_validation_curves_comparison.png",
        args.dpi,
    )
    write_csv(baseline, improved, output_dir / "comparison_metrics.csv")
    print(f"Wrote Stage-1 run comparison to {output_dir}")


if __name__ == "__main__":
    main()
