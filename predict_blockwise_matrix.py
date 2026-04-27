import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from blockwise_data import _build_block_feature_map, _resolve_event_path, _validate_columns, event_key
from blockwise_matrix_data import (
    REQUIRED_BLOCK_LOOKUP_COLUMNS,
    REQUIRED_LABEL_10M_MANIFEST_COLUMNS,
    BlockWindow,
    _compute_block_windows,
)
from blockwise_model import BlockwiseFloodMatrixModel
from train_blockwise_matrix import resolve_device


REQUIRED_EVENT_COLUMNS = {"event_id", "watershed_id", "path_to_X_event", "T", "F"}
REQUIRED_BLOCK_COLUMNS = {"watershed_id", "block_id"}


class MatrixInferenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        event_arrays: Dict[str, np.ndarray],
        block_feature_map: Dict[Tuple[str, str], np.ndarray],
        block_index_grid: np.ndarray,
        block_windows: Dict[int, BlockWindow],
        target_shape: Tuple[int, int],
        peak_grids: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.event_arrays = event_arrays
        self.block_feature_map = block_feature_map
        self.block_index_grid = block_index_grid
        self.block_windows = block_windows
        self.target_rows, self.target_cols = target_shape
        self.peak_grids = peak_grids

    def __len__(self) -> int:
        return len(self.frame)

    def _pad_patch(self, patch: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.target_rows, self.target_cols), dtype=np.float32)
        start_row = (self.target_rows - patch.shape[0]) // 2
        start_col = (self.target_cols - patch.shape[1]) // 2
        padded[start_row : start_row + patch.shape[0], start_col : start_col + patch.shape[1]] = patch
        return padded

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        block_index = int(row["block_index"])
        window = self.block_windows[block_index]
        mask_patch = (
            self.block_index_grid[window.row_start : window.row_stop, window.col_start : window.col_stop] == block_index
        ).astype(np.float32)

        payload = [
            torch.from_numpy(self.event_arrays[row["event_key"]].copy()),
            torch.from_numpy(self.block_feature_map[(row["watershed_id"], row["block_id"])].copy()),
            torch.from_numpy(self._pad_patch(mask_patch)),
        ]

        if self.peak_grids is not None and row["event_key"] in self.peak_grids:
            peak_grid = self.peak_grids[row["event_key"]]
            target_patch = np.asarray(
                peak_grid[window.row_start : window.row_stop, window.col_start : window.col_stop],
                dtype=np.float32,
            )
            target_patch = np.where(np.isfinite(target_patch), target_patch, 0.0).astype(np.float32)
            target_patch *= mask_patch
            payload.append(torch.from_numpy(self._pad_patch(target_patch)))

        return tuple(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict 80x80 block-local depth fields with a trained matrix surrogate")
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--event-ids", nargs="+", default=None, help="Optional subset of event IDs to score")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true", help="Compute masked metrics when target peak rasters exist")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_normalization(path: Path) -> Dict[str, np.ndarray]:
    stats = np.load(path, allow_pickle=True)
    feature_columns = [str(value) for value in stats["block_feature_columns"].tolist()]
    return {
        "event_mean": stats["event_mean"].astype(np.float32),
        "event_std": stats["event_std"].astype(np.float32),
        "block_mean": stats["block_mean"].astype(np.float32),
        "block_std": stats["block_std"].astype(np.float32),
        "feature_columns": feature_columns,
    }


def load_events(events_csv: Path, base_dir: Path, event_ids: Optional[list[str]]) -> pd.DataFrame:
    events_df = pd.read_csv(events_csv)
    _validate_columns(events_df, REQUIRED_EVENT_COLUMNS, "events.csv")
    if event_ids:
        requested = set(event_ids)
        events_df = events_df.loc[events_df["event_id"].isin(requested)].reset_index(drop=True)
        if events_df.empty:
            raise ValueError("No events matched --event-ids")

    events_df = events_df.copy()
    events_df["event_key"] = [
        event_key(watershed_id, event_id)
        for watershed_id, event_id in zip(events_df["watershed_id"], events_df["event_id"])
    ]
    events_df["resolved_event_path"] = [
        str(_resolve_event_path(raw_path, base_dir, events_csv, watershed_id, event_id))
        for raw_path, watershed_id, event_id in zip(
            events_df["path_to_X_event"],
            events_df["watershed_id"],
            events_df["event_id"],
        )
    ]
    return events_df


def load_normalized_events(events_df: pd.DataFrame, event_mean: np.ndarray, event_std: np.ndarray) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    expected_shape = None
    for row in events_df.itertuples(index=False):
        array = np.load(row.resolved_event_path).astype(np.float32)
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("All inference events must share one common shape")
        arrays[row.event_key] = ((array - event_mean) / event_std).astype(np.float32)
    return arrays


def load_peak_grids(labels_10m_dir: Path, event_frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    manifest_path = labels_10m_dir / "labels_10m_manifest.parquet"
    manifest_df = pd.read_parquet(manifest_path)
    _validate_columns(manifest_df, REQUIRED_LABEL_10M_MANIFEST_COLUMNS, "labels_10m_manifest.parquet")
    manifest_df = manifest_df.copy()
    manifest_df["event_key"] = [
        event_key(watershed_id, event_id)
        for watershed_id, event_id in zip(manifest_df["watershed_id"], manifest_df["event_id"])
    ]
    manifest_df = manifest_df.loc[manifest_df["event_key"].isin(event_frame["event_key"])].reset_index(drop=True)

    grids: Dict[str, np.ndarray] = {}
    for row in manifest_df.itertuples(index=False):
        peak_path = Path(row.path_to_peak_10m)
        if not peak_path.exists():
            peak_path = (labels_10m_dir / peak_path).resolve()
        grids[row.event_key] = np.load(peak_path, mmap_mode="r")
    return grids


def build_inference_frame(events_df: pd.DataFrame, blocks_df: pd.DataFrame, block_lookup_df: pd.DataFrame) -> pd.DataFrame:
    return events_df[["event_id", "watershed_id", "event_key"]].merge(
        block_lookup_df[["watershed_id", "block_id", "block_index"]],
        on="watershed_id",
        how="inner",
        validate="many_to_many",
    ).merge(
        blocks_df[["watershed_id", "block_id"]],
        on=["watershed_id", "block_id"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["watershed_id", "event_id", "block_id"]).reset_index(drop=True)


def update_masked_metrics(accumulator: Dict[str, float], predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> None:
    valid = masks > 0.5
    pred = predictions[valid].astype(np.float64)
    targ = targets[valid].astype(np.float64)
    if len(pred) == 0:
        return
    errors = pred - targ
    accumulator["count"] += float(len(pred))
    accumulator["sum_abs"] += float(np.abs(errors).sum())
    accumulator["sum_sq"] += float((errors ** 2).sum())
    accumulator["sum_y"] += float(targ.sum())
    accumulator["sum_y_sq"] += float((targ ** 2).sum())


def finalize_masked_metrics(accumulator: Dict[str, float]) -> Dict[str, float]:
    count = accumulator["count"]
    if count <= 0:
        raise ValueError("No valid cells were available for evaluation")
    mse = accumulator["sum_sq"] / count
    mae = accumulator["sum_abs"] / count
    rmse = float(np.sqrt(mse))
    mean_y = accumulator["sum_y"] / count
    ss_tot = accumulator["sum_y_sq"] - (count * mean_y * mean_y)
    r2 = float(1.0 - accumulator["sum_sq"] / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": float(mse), "mae": float(mae), "rmse": rmse, "r2": r2}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events_df = load_events(args.events_csv.resolve(), args.base_dir.resolve(), args.event_ids)
    blocks_df = pd.read_parquet(args.blocks_parquet.resolve())
    _validate_columns(blocks_df, REQUIRED_BLOCK_COLUMNS, "blocks.parquet")

    labels_10m_dir = args.labels_10m_dir.resolve()
    block_lookup_df = pd.read_parquet(labels_10m_dir / "block_index_lookup.parquet")
    _validate_columns(block_lookup_df, REQUIRED_BLOCK_LOOKUP_COLUMNS, "block_index_lookup.parquet")
    block_index_grid = np.load(labels_10m_dir / "block_index_10m.npy")

    norm = load_normalization(args.normalization_stats.resolve())
    missing_features = [column for column in norm["feature_columns"] if column not in blocks_df.columns]
    if missing_features:
        raise ValueError(f"blocks.parquet is missing feature columns required by normalization stats: {missing_features}")

    inference_frame = build_inference_frame(events_df, blocks_df, block_lookup_df)
    if inference_frame.empty:
        raise ValueError("Inference frame is empty after joining events and blocks")

    event_arrays = load_normalized_events(events_df, norm["event_mean"], norm["event_std"])
    block_feature_map = _build_block_feature_map(
        blocks_df[["watershed_id", "block_id", *norm["feature_columns"]]].copy(),
        norm["feature_columns"],
        norm["block_mean"],
        norm["block_std"],
    )

    checkpoint = load_checkpoint(args.checkpoint.resolve(), device)
    if "target_shape" in checkpoint:
        final_target_shape = tuple(int(value) for value in checkpoint["target_shape"])
    else:
        final_target_shape = (
            int(checkpoint["model_config"]["target_rows"]),
            int(checkpoint["model_config"]["target_cols"]),
        )

    block_windows, final_target_shape = _compute_block_windows(
        block_index_grid=block_index_grid,
        block_lookup_df=block_lookup_df,
        target_rows=final_target_shape[0],
        target_cols=final_target_shape[1],
    )

    peak_grids = None
    if args.evaluate:
        peak_grids = load_peak_grids(labels_10m_dir, events_df[["event_key"]].drop_duplicates())
        available_event_keys = set(peak_grids)
        requested_event_keys = set(events_df["event_key"])
        if requested_event_keys - available_event_keys:
            missing = sorted(requested_event_keys - available_event_keys)
            raise ValueError(f"Evaluation requested, but target peak rasters are missing for event keys: {missing}")

    dataset = MatrixInferenceDataset(
        frame=inference_frame,
        event_arrays=event_arrays,
        block_feature_map=block_feature_map,
        block_index_grid=block_index_grid,
        block_windows=block_windows,
        target_shape=final_target_shape,
        peak_grids=peak_grids,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BlockwiseFloodMatrixModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_path = output_dir / "predictions.npy"
    prediction_array = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), final_target_shape[0], final_target_shape[1]),
    )

    metrics_accumulator = {"count": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_y": 0.0, "sum_y_sq": 0.0}
    sample_offset = 0

    with torch.no_grad():
        for batch in loader:
            if args.evaluate:
                event_tensor, block_features, block_mask, target_map = batch
                target_map_np = target_map.numpy()
            else:
                event_tensor, block_features, block_mask = batch
                target_map_np = None

            prediction_map = model(
                event_tensor.to(device),
                block_features.to(device),
                block_mask.to(device),
            ).cpu().numpy().astype(np.float32)

            batch_size = prediction_map.shape[0]
            prediction_array[sample_offset : sample_offset + batch_size] = prediction_map
            if args.evaluate and target_map_np is not None:
                update_masked_metrics(metrics_accumulator, prediction_map, target_map_np, block_mask.numpy())
            sample_offset += batch_size

    prediction_array.flush()

    manifest = inference_frame.copy()
    manifest.insert(0, "sample_index", np.arange(len(manifest), dtype=np.int64))
    manifest.to_parquet(output_dir / "prediction_manifest.parquet", index=False)

    summary = {
        "prediction_path": str(prediction_path),
        "manifest_path": str(output_dir / "prediction_manifest.parquet"),
        "num_samples": len(dataset),
        "target_shape": list(final_target_shape),
        "evaluated": bool(args.evaluate),
    }

    if args.evaluate:
        metrics = finalize_masked_metrics(metrics_accumulator)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        summary["metrics_path"] = str(output_dir / "metrics.json")
        print(
            "[Metrics] rmse={rmse:.6f} mae={mae:.6f} r2={r2:.6f}".format(
                rmse=metrics["rmse"], mae=metrics["mae"], r2=metrics["r2"]
            )
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[Output] wrote {} samples to {}".format(len(dataset), output_dir))


if __name__ == "__main__":
    main()