#!/usr/bin/env python3
"""Create meeting-ready diagnostics for a completed Stage-1 training run."""

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


COLORS = {
    "train": "#3B6FB6",
    "validation": "#E07A3F",
    "test": "#4A9D70",
    "marker": "#555555",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot learning history and held-out Stage-1 metrics"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("results/stage1_timestamp")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, decimals=3):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_learning_curves(history, best_epoch, output_path, dpi):
    epochs = np.asarray([row["epoch"] for row in history])
    panels = [
        ("loss", "Composite loss", None),
        ("depth_wet_rmse", "Wet-cell depth RMSE", "m"),
        ("component_rmse", "Velocity-component RMSE", "m/s"),
        ("wet_f1", "Wet/dry F1", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for ax, (metric, title, unit) in zip(axes.flat, panels):
        train = [row["train"][metric] for row in history]
        val = [row["val"][metric] for row in history]
        ax.plot(
            epochs,
            train,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=COLORS["train"],
            label="Training",
        )
        ax.plot(
            epochs,
            val,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=COLORS["validation"],
            label="Validation",
        )
        ax.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.2,
            color=COLORS["marker"],
            label=f"Selected epoch {best_epoch}",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(unit or metric.replace("_", " ").title())
        ax.set_xticks(epochs[:: max(1, len(epochs) // 9)])
        style_axis(ax)
    axes.flat[0].legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.suptitle("Stage-1 training history", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_depth(best_val, test, output_path, dpi):
    metrics = ["depth_all_mae", "depth_all_rmse", "depth_wet_mae", "depth_wet_rmse"]
    labels = ["All-cell\nMAE", "All-cell\nRMSE", "Wet-cell\nMAE", "Wet-cell\nRMSE"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    val_bars = ax.bar(
        x - width / 2,
        [best_val[key] for key in metrics],
        width,
        label="Validation (selected epoch)",
        color=COLORS["validation"],
    )
    test_bars = ax.bar(
        x + width / 2,
        [test[key] for key in metrics],
        width,
        label="Held-out D030",
        color=COLORS["test"],
    )
    annotate_bars(ax, val_bars)
    annotate_bars(ax, test_bars)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error (m)")
    ax.set_title("Depth prediction error", fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.98,
        "Wet cells: true depth ≥ 0.05 m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#555555",
        fontsize=9,
    )
    style_axis(ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_velocity(best_val, test, output_path, dpi):
    metrics = ["component_mae", "component_rmse"]
    labels = ["MAE", "RMSE"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    val_bars = ax.bar(
        x - width / 2,
        [best_val[key] for key in metrics],
        width,
        label="Validation (selected epoch)",
        color=COLORS["validation"],
    )
    test_bars = ax.bar(
        x + width / 2,
        [test[key] for key in metrics],
        width,
        label="Held-out D030",
        color=COLORS["test"],
    )
    annotate_bars(ax, val_bars)
    annotate_bars(ax, test_bars)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Velocity-component error (m/s)")
    ax.set_title("Velocity prediction on true wet cells", fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_inundation(best_val, test, output_path, dpi):
    metrics = ["wet_precision", "wet_recall", "wet_f1", "wet_csi"]
    labels = ["Precision", "Recall", "F1", "CSI"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    val_bars = ax.bar(
        x - width / 2,
        [best_val[key] for key in metrics],
        width,
        label="Validation (selected epoch)",
        color=COLORS["validation"],
    )
    test_bars = ax.bar(
        x + width / 2,
        [test[key] for key in metrics],
        width,
        label="Held-out D030",
        color=COLORS["test"],
    )
    annotate_bars(ax, val_bars)
    annotate_bars(ax, test_bars)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Wet/dry inundation skill", fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_tables(history, best_epoch, best_val, test, output_dir):
    metric_names = sorted(best_val)
    with (output_dir / "summary_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "validation_selected_epoch", "test_D030"])
        for metric in metric_names:
            writer.writerow([metric, best_val[metric], test.get(metric, "")])

    history_metrics = sorted(history[0]["train"])
    with (output_dir / "epoch_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "selected_checkpoint"]
            + [f"train_{key}" for key in history_metrics]
            + [f"val_{key}" for key in history_metrics]
        )
        for row in history:
            writer.writerow(
                [row["epoch"], row["epoch"] == best_epoch]
                + [row["train"][key] for key in history_metrics]
                + [row["val"][key] for key in history_metrics]
            )


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((run_dir / "metrics.json").read_text())
    history = payload["history"]
    if not history:
        raise ValueError("metrics.json has no epoch history")
    selected = min(history, key=lambda row: row["val"]["loss"])
    best_epoch = int(selected["epoch"])
    best_val = selected["val"]
    test = payload["test"]

    plot_learning_curves(
        history, best_epoch, output_dir / "01_learning_curves.png", args.dpi
    )
    plot_depth(best_val, test, output_dir / "02_depth_metrics.png", args.dpi)
    plot_velocity(best_val, test, output_dir / "03_velocity_metrics.png", args.dpi)
    plot_inundation(best_val, test, output_dir / "04_inundation_metrics.png", args.dpi)
    write_tables(history, best_epoch, best_val, test, output_dir)
    print(f"Wrote Stage-1 figures and tables to {output_dir}")


if __name__ == "__main__":
    main()
