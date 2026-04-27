import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEPTH_BINS = [-0.001, 0.0, 0.1, 0.5, 1.0, 2.0, np.inf]
DEPTH_LABELS = ["0", "0-0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction diagnostics for blockwise flood model")
    parser.add_argument("--predictions-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def add_depth_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["depth_bin"] = pd.cut(result["y"], bins=DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)
    return result


def plot_scatter(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = frame if len(frame) <= 10000 else frame.sample(10000, random_state=42)
    ax.scatter(sample["y"], sample["y_pred"], s=8, alpha=0.25)

    max_value = float(max(frame["y"].max(), frame["y_pred"].max()))
    ax.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1.5, color="black")
    ax.set_title("Prediction vs Truth")
    ax.set_xlabel("True Depth y")
    ax.set_ylabel("Predicted Depth y_pred")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_depth_bin_metrics(frame: pd.DataFrame, output_dir: Path) -> None:
    records = []
    for label, group in frame.groupby("depth_bin", observed=False):
        if group.empty:
            continue
        error = group["y_pred"] - group["y"]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        mae = float(np.mean(np.abs(error)))
        records.append({"depth_bin": str(label), "count": len(group), "rmse": rmse, "mae": mae})

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(output_dir / "depth_bin_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(metrics_df["depth_bin"], metrics_df["count"], color="#355070")
    axes[0].set_title("Samples by True-Depth Bin")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(metrics_df["depth_bin"], metrics_df["rmse"], color="#6d597a")
    axes[1].set_title("RMSE by True-Depth Bin")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(metrics_df["depth_bin"], metrics_df["mae"], color="#b56576")
    axes[2].set_title("MAE by True-Depth Bin")
    axes[2].set_ylabel("MAE")
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_dir / "depth_bin_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.predictions_parquet.resolve())
    required = {"y", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions parquet missing columns: {sorted(missing)}")

    frame = add_depth_bins(frame)
    plot_scatter(frame, output_dir)
    plot_depth_bin_metrics(frame, output_dir)
    print(f"Wrote prediction plots to {output_dir}")


if __name__ == "__main__":
    main()