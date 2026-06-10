import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blockwise_data import event_key
from blockwise_matrix_data import _compute_block_windows


DEPTH_BINS = [-0.001, 0.0, 0.1, 0.5, 1.0, 2.0, np.inf]
DEPTH_LABELS = ["0", "0-0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot matrix prediction diagnostics for one event")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scatter-points", type=int, default=200000)
    return parser.parse_args()


def resolve_peak_path(raw_path: str, labels_10m_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()
    resolved = (labels_10m_dir / candidate).resolve()
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Could not resolve path_to_peak_10m: {raw_path}")


def choose_event(manifest: pd.DataFrame, event_id_arg: str | None) -> tuple[str, str]:
    unique_events = manifest[["event_id", "watershed_id"]].drop_duplicates().reset_index(drop=True)
    if event_id_arg is None:
        if len(unique_events) != 1:
            options = unique_events["event_id"].tolist()
            raise ValueError(f"Multiple events found in prediction manifest; pass --event-id from {options}")
        row = unique_events.iloc[0]
        return str(row["event_id"]), str(row["watershed_id"])

    matched = unique_events.loc[unique_events["event_id"] == event_id_arg].reset_index(drop=True)
    if matched.empty:
        options = sorted(unique_events["event_id"].astype(str).unique().tolist())
        raise ValueError(f"event_id '{event_id_arg}' is not in prediction manifest. Available: {options}")
    if matched["watershed_id"].nunique() != 1:
        raise ValueError(f"event_id '{event_id_arg}' maps to multiple watersheds in manifest")
    row = matched.iloc[0]
    return str(row["event_id"]), str(row["watershed_id"])


def unpad_patch(padded_patch: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    start_row = (padded_patch.shape[0] - out_rows) // 2
    start_col = (padded_patch.shape[1] - out_cols) // 2
    return padded_patch[start_row : start_row + out_rows, start_col : start_col + out_cols]


def load_true_grid(labels_10m_dir: Path, watershed_id: str, event_id: str) -> np.ndarray:
    label_manifest = pd.read_parquet((labels_10m_dir / "labels_10m_manifest.parquet").resolve())
    label_manifest = label_manifest.copy()
    label_manifest["event_key"] = [
        event_key(ws, eid)
        for ws, eid in zip(label_manifest["watershed_id"], label_manifest["event_id"])
    ]
    key = event_key(watershed_id, event_id)
    matched = label_manifest.loc[label_manifest["event_key"] == key].reset_index(drop=True)
    if matched.empty:
        raise ValueError(f"No labels_10m manifest entry found for event_key '{key}'")
    peak_path = resolve_peak_path(str(matched.loc[0, "path_to_peak_10m"]), labels_10m_dir)
    return np.asarray(np.load(peak_path, mmap_mode="r"), dtype=np.float32)


def collect_valid_cells(
    event_rows: pd.DataFrame,
    prediction_array: np.ndarray,
    true_grid: np.ndarray,
    block_index_grid: np.ndarray,
    block_lookup_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    block_windows, _ = _compute_block_windows(
        block_index_grid=block_index_grid,
        block_lookup_df=block_lookup_df,
        target_rows=prediction_array.shape[1],
        target_cols=prediction_array.shape[2],
    )

    pred_values: list[np.ndarray] = []
    true_values: list[np.ndarray] = []

    for row in event_rows.itertuples(index=False):
        block_index = int(row.block_index)
        window = block_windows[block_index]
        padded_patch = prediction_array[int(row.sample_index)]
        pred_patch = unpad_patch(padded_patch, window.height, window.width)
        true_patch = true_grid[window.row_start : window.row_stop, window.col_start : window.col_stop]
        mask = block_index_grid[window.row_start : window.row_stop, window.col_start : window.col_stop] == block_index
        finite = np.isfinite(true_patch)
        valid = mask & finite
        if not np.any(valid):
            continue
        pred_values.append(pred_patch[valid].astype(np.float32))
        true_values.append(true_patch[valid].astype(np.float32))

    if not pred_values:
        raise ValueError("No valid cells found for event diagnostics")

    return np.concatenate(true_values), np.concatenate(pred_values)


def add_depth_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["depth_bin"] = pd.cut(result["y"], bins=DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)
    return result


def plot_scatter(frame: pd.DataFrame, output_dir: Path, max_points: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sample = frame if len(frame) <= max_points else frame.sample(max_points, random_state=42)
    ax.scatter(sample["y"], sample["y_pred"], s=3, alpha=0.2)

    max_value = float(max(frame["y"].max(), frame["y_pred"].max()))
    ax.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1.5, color="black")
    ax.set_title("Prediction vs Truth (Valid Cells)")
    ax.set_xlabel("True depth")
    ax.set_ylabel("Predicted depth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_scatter_cells.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_depth_bin_metrics(frame: pd.DataFrame, output_dir: Path) -> None:
    records = []
    for label, group in frame.groupby("depth_bin", observed=False):
        if group.empty:
            continue
        error = group["y_pred"] - group["y"]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        mae = float(np.mean(np.abs(error)))
        records.append({"depth_bin": str(label), "count": int(len(group)), "rmse": rmse, "mae": mae})

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(output_dir / "depth_bin_metrics_cells.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    axes[0].bar(metrics_df["depth_bin"], metrics_df["count"], color="#355070")
    axes[0].set_title("Valid Cells by Depth Bin")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(metrics_df["depth_bin"], metrics_df["rmse"], color="#6d597a")
    axes[1].set_title("RMSE by Depth Bin")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(metrics_df["depth_bin"], metrics_df["mae"], color="#b56576")
    axes[2].set_title("MAE by Depth Bin")
    axes[2].set_ylabel("MAE")
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.savefig(output_dir / "depth_bin_metrics_cells.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prediction_dir = args.prediction_dir.resolve()
    labels_10m_dir = args.labels_10m_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((prediction_dir / "summary.json").read_text())
    prediction_path = Path(summary["prediction_path"]).resolve()
    manifest_path = Path(summary["manifest_path"]).resolve()

    manifest = pd.read_parquet(manifest_path)
    event_id, watershed_id = choose_event(manifest, args.event_id)
    event_rows = manifest.loc[(manifest["event_id"] == event_id) & (manifest["watershed_id"] == watershed_id)].copy()

    prediction_array = np.load(prediction_path, mmap_mode="r")
    true_grid = load_true_grid(labels_10m_dir, watershed_id, event_id)
    block_index_grid = np.load((labels_10m_dir / "block_index_10m.npy").resolve())
    block_lookup = pd.read_parquet((labels_10m_dir / "block_index_lookup.parquet").resolve())
    block_lookup = block_lookup.loc[block_lookup["watershed_id"] == watershed_id].reset_index(drop=True)

    y_true, y_pred = collect_valid_cells(
        event_rows=event_rows,
        prediction_array=prediction_array,
        true_grid=true_grid,
        block_index_grid=block_index_grid,
        block_lookup_df=block_lookup,
    )

    frame = pd.DataFrame({"y": y_true, "y_pred": y_pred})
    frame = add_depth_bins(frame)
    frame.to_parquet(output_dir / f"cell_level_predictions_{event_id}.parquet", index=False)

    plot_scatter(frame, output_dir, max_points=args.max_scatter_points)
    plot_depth_bin_metrics(frame, output_dir)

    print(
        "Wrote matrix prediction diagnostics to {} for event {} ({} valid cells)".format(
            output_dir, event_id, len(frame)
        )
    )


if __name__ == "__main__":
    main()
