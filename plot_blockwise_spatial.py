import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spatial prediction diagnostics for blockwise flood model")
    parser.add_argument("--predictions-parquet", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def load_frame(predictions_parquet: Path, blocks_parquet: Path) -> pd.DataFrame:
    pred = pd.read_parquet(predictions_parquet.resolve())
    blocks = pd.read_parquet(blocks_parquet.resolve())
    merged = pred.merge(
        blocks[["watershed_id", "block_id", "centroid_x", "centroid_y"]],
        on=["watershed_id", "block_id"],
        how="inner",
        validate="one_to_one",
    )
    merged["error"] = merged["y_pred"] - merged["y"]
    return merged


def draw_panel(ax, frame: pd.DataFrame, value_column: str, title: str, cmap: str, norm) -> None:
    scatter = ax.scatter(
        frame["centroid_x"],
        frame["centroid_y"],
        c=frame[value_column],
        cmap=cmap,
        norm=norm,
        s=18,
        marker="s",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("Centroid X")
    ax.set_ylabel("Centroid Y")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    return scatter


def main() -> None:
    args = parse_args()
    frame = load_frame(args.predictions_parquet, args.blocks_parquet)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    value_max = float(max(frame["y"].max(), frame["y_pred"].max()))
    value_norm = mcolors.Normalize(vmin=0.0, vmax=value_max)

    error_abs = float(np.max(np.abs(frame["error"])))
    error_norm = mcolors.TwoSlopeNorm(vmin=-error_abs, vcenter=0.0, vmax=error_abs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    draw_panel(axes[0], frame, "y", "True Flood Depth", "viridis", value_norm)
    draw_panel(axes[1], frame, "y_pred", "Predicted Flood Depth", "viridis", value_norm)
    draw_panel(axes[2], frame, "error", "Prediction Error (Pred - True)", "coolwarm", error_norm)

    value_sm = plt.cm.ScalarMappable(norm=value_norm, cmap="viridis")
    value_sm.set_array([])
    error_sm = plt.cm.ScalarMappable(norm=error_norm, cmap="coolwarm")
    error_sm.set_array([])

    fig.colorbar(value_sm, ax=axes[:2], shrink=0.85, label="Flood depth")
    fig.colorbar(error_sm, ax=axes[2], shrink=0.85, label="Prediction error")

    event_ids = sorted(frame["event_id"].unique().tolist())
    if len(event_ids) == 1:
        fig.suptitle(f"Spatial Prediction Map for {event_ids[0]}", fontsize=14)

    fig.savefig(args.output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote spatial plot to {args.output_path}")


if __name__ == "__main__":
    main()