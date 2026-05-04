import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blockwise_data import event_key
from blockwise_matrix_data import _compute_block_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spatial true/pred/error maps from matrix prediction outputs")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="auto",
        choices=["auto", "depth", "wet"],
        help="How to define white background in prediction panel",
    )
    parser.add_argument(
        "--dry-threshold",
        type=float,
        default=1e-6,
        help="Cells with predicted depth <= threshold are shown as white in the prediction panel",
    )
    parser.add_argument(
        "--wet-prob-threshold",
        type=float,
        default=0.5,
        help="Cells with wet probability < threshold are shown as white when mask-mode is wet/auto",
    )
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


def reconstruct_grid_from_samples(
    event_rows: pd.DataFrame,
    sample_array: np.ndarray,
    block_index_grid: np.ndarray,
    block_lookup_df: pd.DataFrame,
) -> np.ndarray:
    block_windows, _ = _compute_block_windows(
        block_index_grid=block_index_grid,
        block_lookup_df=block_lookup_df,
        target_rows=sample_array.shape[1],
        target_cols=sample_array.shape[2],
    )

    grid = np.full(block_index_grid.shape, np.nan, dtype=np.float32)
    for row in event_rows.itertuples(index=False):
        block_index = int(row.block_index)
        window = block_windows[block_index]
        padded_patch = sample_array[int(row.sample_index)]
        patch = unpad_patch(padded_patch, window.height, window.width)
        mask = block_index_grid[window.row_start : window.row_stop, window.col_start : window.col_stop] == block_index
        target_view = grid[window.row_start : window.row_stop, window.col_start : window.col_stop]
        target_view[mask] = patch[mask]

    return grid


def draw_panel(ax, grid: np.ndarray, title: str, cmap: str, norm) -> None:
    masked = np.ma.masked_invalid(grid)
    local_cmap = mpl.colormaps.get_cmap(cmap).copy()
    local_cmap.set_bad("white")
    image = ax.imshow(masked, cmap=local_cmap, norm=norm, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    return image


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


def main() -> None:
    args = parse_args()

    prediction_dir = args.prediction_dir.resolve()
    labels_10m_dir = args.labels_10m_dir.resolve()

    summary_path = prediction_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    prediction_path = Path(summary["prediction_path"]).resolve()
    manifest_path = Path(summary["manifest_path"]).resolve()
    wet_probability_path = None
    if "wet_probability_path" in summary:
        wet_probability_path = Path(summary["wet_probability_path"]).resolve()

    manifest = pd.read_parquet(manifest_path)
    event_id, watershed_id = choose_event(manifest, args.event_id)
    event_rows = manifest.loc[(manifest["event_id"] == event_id) & (manifest["watershed_id"] == watershed_id)].copy()
    if event_rows.empty:
        raise ValueError("Selected event had zero samples in prediction manifest")

    prediction_array = np.load(prediction_path, mmap_mode="r")
    block_index_grid = np.load((labels_10m_dir / "block_index_10m.npy").resolve())
    block_lookup = pd.read_parquet((labels_10m_dir / "block_index_lookup.parquet").resolve())
    block_lookup = block_lookup.loc[block_lookup["watershed_id"] == watershed_id].reset_index(drop=True)

    pred_grid = reconstruct_grid_from_samples(
        event_rows=event_rows,
        sample_array=prediction_array,
        block_index_grid=block_index_grid,
        block_lookup_df=block_lookup,
    )

    wet_grid = None
    if wet_probability_path is not None and wet_probability_path.exists():
        wet_probability_array = np.load(wet_probability_path, mmap_mode="r")
        wet_grid = reconstruct_grid_from_samples(
            event_rows=event_rows,
            sample_array=wet_probability_array,
            block_index_grid=block_index_grid,
            block_lookup_df=block_lookup,
        )
    true_grid = load_true_grid(labels_10m_dir, watershed_id, event_id)

    valid_mask = block_index_grid >= 0
    true_grid = np.where(np.isfinite(true_grid) & valid_mask, true_grid, np.nan)
    pred_grid = np.where(valid_mask, pred_grid, np.nan)
    error_grid = pred_grid - true_grid

    # Render predicted dry cells as no-data in the prediction panel for visual parity with true-depth masking.
    pred_display_grid = pred_grid.copy()
    if args.mask_mode == "wet" and wet_grid is None:
        raise ValueError("mask-mode 'wet' requested but wet_probability_path was not found in summary.json")

    use_wet_mask = wet_grid is not None and args.mask_mode in {"auto", "wet"}
    if use_wet_mask:
        pred_display_grid[np.isfinite(wet_grid) & (wet_grid < args.wet_prob_threshold)] = np.nan
        mask_mode_used = f"wet(p<{args.wet_prob_threshold})"
    else:
        pred_display_grid[np.isfinite(pred_display_grid) & (pred_display_grid <= args.dry_threshold)] = np.nan
        mask_mode_used = f"depth(<= {args.dry_threshold})"

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_true = true_grid[np.isfinite(true_grid)]
    valid_pred = pred_grid[np.isfinite(pred_grid)]
    if valid_true.size == 0 or valid_pred.size == 0:
        raise ValueError("Could not build valid true/predicted grids for plotting")

    value_max = float(max(np.nanmax(valid_true), np.nanmax(valid_pred)))
    value_norm = mcolors.Normalize(vmin=0.0, vmax=value_max)

    error_abs = float(np.nanmax(np.abs(error_grid)))
    error_norm = mcolors.TwoSlopeNorm(vmin=-error_abs, vcenter=0.0, vmax=error_abs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    draw_panel(axes[0], true_grid, "True Flood Depth", "viridis", value_norm)
    draw_panel(axes[1], pred_display_grid, "Predicted Flood Depth", "viridis", value_norm)
    draw_panel(axes[2], error_grid, "Prediction Error (Pred - True)", "coolwarm", error_norm)

    value_sm = plt.cm.ScalarMappable(norm=value_norm, cmap="viridis")
    value_sm.set_array([])
    error_sm = plt.cm.ScalarMappable(norm=error_norm, cmap="coolwarm")
    error_sm.set_array([])

    fig.colorbar(value_sm, ax=axes[:2], shrink=0.85, label="Flood depth")
    fig.colorbar(error_sm, ax=axes[2], shrink=0.85, label="Prediction error")

    fig.suptitle(f"Spatial Prediction Map for {event_id} ({watershed_id})", fontsize=14)

    fig.savefig(args.output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mask] prediction panel background mode: {mask_mode_used}")
    print(f"Wrote spatial plot to {args.output_path}")


if __name__ == "__main__":
    main()