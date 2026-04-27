from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from blockwise_data import (
    BlockwiseSplit,
    NormalizationStats,
    _build_block_feature_map,
    _fit_block_normalization,
    _fit_event_normalization,
    _load_and_validate_event_arrays,
    _normalize_event_arrays,
    _resolve_event_path,
    _validate_columns,
    event_key,
    split_samples_by_event,
)


REQUIRED_LABEL_10M_MANIFEST_COLUMNS = {"event_id", "watershed_id", "path_to_peak_10m", "rows", "cols"}
REQUIRED_BLOCK_LOOKUP_COLUMNS = {"watershed_id", "block_id", "block_index", "n_cells_10m"}


@dataclass
class BlockWindow:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    height: int
    width: int


@dataclass
class BlockwiseMatrixDataBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    feature_columns: List[str]
    event_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    normalization: NormalizationStats
    splits: BlockwiseSplit


class BlockwiseFloodMatrixDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        event_arrays: Dict[str, np.ndarray],
        block_feature_map: Dict[Tuple[str, str], np.ndarray],
        peak_grids: Dict[str, np.ndarray],
        block_index_grid: np.ndarray,
        block_windows: Dict[int, BlockWindow],
        target_shape: Tuple[int, int],
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.event_arrays = event_arrays
        self.block_feature_map = block_feature_map
        self.peak_grids = peak_grids
        self.block_index_grid = block_index_grid
        self.block_windows = block_windows
        self.target_rows, self.target_cols = target_shape

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_patch(self, patch: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.target_rows, self.target_cols), dtype=np.float32)
        start_row = (self.target_rows - patch.shape[0]) // 2
        start_col = (self.target_cols - patch.shape[1]) // 2
        padded[start_row : start_row + patch.shape[0], start_col : start_col + patch.shape[1]] = patch
        return padded

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        sample_key = (row["watershed_id"], row["block_id"])
        event_tensor = self.event_arrays[row["event_key"]]
        block_features = self.block_feature_map[sample_key]
        peak_grid = self.peak_grids[row["event_key"]]
        block_index = int(row["block_index"])
        window = self.block_windows[block_index]

        target_patch = np.asarray(
            peak_grid[window.row_start : window.row_stop, window.col_start : window.col_stop],
            dtype=np.float32,
        )
        mask_patch = (
            self.block_index_grid[window.row_start : window.row_stop, window.col_start : window.col_stop] == block_index
        ).astype(np.float32)
        target_patch = np.where(np.isfinite(target_patch), target_patch, 0.0).astype(np.float32)
        target_patch *= mask_patch

        return (
            torch.from_numpy(event_tensor.copy()),
            torch.from_numpy(block_features.copy()),
            torch.from_numpy(self._pad_patch(mask_patch)),
            torch.from_numpy(self._pad_patch(target_patch)),
        )


def _resolve_existing_path(raw_path: str, base_dir: Path, anchor_dir: Path) -> Path:
    candidate = Path(raw_path)
    candidates = [candidate, base_dir / candidate, anchor_dir / candidate]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve path '{raw_path}'")


def _load_peak_grids(manifest_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    grids: Dict[str, np.ndarray] = {}
    for row in manifest_df.itertuples(index=False):
        grid = np.load(row.resolved_peak_path, mmap_mode="r")
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D peak raster, got shape {grid.shape} at {row.resolved_peak_path}")
        grids[row.event_key] = grid
    return grids


def _compute_block_windows(
    block_index_grid: np.ndarray,
    block_lookup_df: pd.DataFrame,
    target_rows: Optional[int],
    target_cols: Optional[int],
) -> Tuple[Dict[int, BlockWindow], Tuple[int, int]]:
    valid_cells = block_index_grid >= 0
    rows, cols = np.nonzero(valid_cells)
    indices = block_index_grid[valid_cells].astype(np.int64)

    if len(indices) == 0:
        raise ValueError("block_index_10m.npy contains no valid block cells")

    n_blocks = int(block_lookup_df["block_index"].max()) + 1
    row_min = np.full(n_blocks, block_index_grid.shape[0], dtype=np.int32)
    row_max = np.full(n_blocks, -1, dtype=np.int32)
    col_min = np.full(n_blocks, block_index_grid.shape[1], dtype=np.int32)
    col_max = np.full(n_blocks, -1, dtype=np.int32)

    np.minimum.at(row_min, indices, rows)
    np.maximum.at(row_max, indices, rows)
    np.minimum.at(col_min, indices, cols)
    np.maximum.at(col_max, indices, cols)

    inferred_rows = 0
    inferred_cols = 0
    windows: Dict[int, BlockWindow] = {}
    for row in block_lookup_df.itertuples(index=False):
        block_index = int(row.block_index)
        if row_max[block_index] < 0:
            raise ValueError(f"Block index {block_index} has zero cells in block_index_10m.npy")

        height = int(row_max[block_index] - row_min[block_index] + 1)
        width = int(col_max[block_index] - col_min[block_index] + 1)
        inferred_rows = max(inferred_rows, height)
        inferred_cols = max(inferred_cols, width)
        windows[block_index] = BlockWindow(
            row_start=int(row_min[block_index]),
            row_stop=int(row_max[block_index] + 1),
            col_start=int(col_min[block_index]),
            col_stop=int(col_max[block_index] + 1),
            height=height,
            width=width,
        )

    final_rows = inferred_rows if target_rows is None else target_rows
    final_cols = inferred_cols if target_cols is None else target_cols

    too_large = [
        block_index
        for block_index, window in windows.items()
        if window.height > final_rows or window.width > final_cols
    ]
    if too_large:
        preview = too_large[:5]
        raise ValueError(
            f"Some blocks exceed target shape {final_rows}x{final_cols}; examples block_index={preview}"
        )

    return windows, (final_rows, final_cols)


def prepare_blockwise_matrix_datasets(
    events_csv: Path,
    blocks_parquet: Path,
    labels_10m_dir: Path,
    base_dir: Path,
    feature_columns: Optional[Sequence[str]],
    test_events: Optional[Sequence[str]],
    val_fraction: float,
    seed: int,
    target_rows: Optional[int] = 80,
    target_cols: Optional[int] = 80,
) -> BlockwiseMatrixDataBundle:
    events_df = pd.read_csv(events_csv)
    blocks_df = pd.read_parquet(blocks_parquet)
    labels_10m_dir = labels_10m_dir.resolve()
    manifest_path = labels_10m_dir / "labels_10m_manifest.parquet"
    block_lookup_path = labels_10m_dir / "block_index_lookup.parquet"
    block_index_path = labels_10m_dir / "block_index_10m.npy"

    manifest_df = pd.read_parquet(manifest_path)
    block_lookup_df = pd.read_parquet(block_lookup_path)
    block_index_grid = np.load(block_index_path)

    _validate_columns(events_df, {"event_id", "watershed_id", "path_to_X_event", "T", "F"}, "events.csv")
    _validate_columns(blocks_df, {"watershed_id", "block_id"}, "blocks.parquet")
    _validate_columns(manifest_df, REQUIRED_LABEL_10M_MANIFEST_COLUMNS, "labels_10m_manifest.parquet")
    _validate_columns(block_lookup_df, REQUIRED_BLOCK_LOOKUP_COLUMNS, "block_index_lookup.parquet")

    events_df = events_df.copy()
    events_df["event_key"] = [
        event_key(watershed_id, event_id)
        for watershed_id, event_id in zip(events_df["watershed_id"], events_df["event_id"])
    ]
    events_df["resolved_event_path"] = [
        str(
            _resolve_event_path(
                raw_path=raw_path,
                base_dir=base_dir,
                events_csv_path=events_csv,
                watershed_id=watershed_id,
                event_id=event_id,
            )
        )
        for raw_path, watershed_id, event_id in zip(
            events_df["path_to_X_event"],
            events_df["watershed_id"],
            events_df["event_id"],
        )
    ]

    manifest_df = manifest_df.copy()
    manifest_df["event_key"] = [
        event_key(watershed_id, event_id)
        for watershed_id, event_id in zip(manifest_df["watershed_id"], manifest_df["event_id"])
    ]
    manifest_df["resolved_peak_path"] = [
        str(_resolve_existing_path(raw_path, labels_10m_dir, manifest_path.parent))
        for raw_path in manifest_df["path_to_peak_10m"]
    ]

    if feature_columns is None:
        feature_columns = [column for column in blocks_df.columns if column not in {"watershed_id", "block_id"}]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No block feature columns were selected")

    missing_features = [column for column in feature_columns if column not in blocks_df.columns]
    if missing_features:
        raise ValueError(f"Requested block feature columns are missing from blocks.parquet: {missing_features}")

    block_features_df = block_lookup_df.merge(
        blocks_df[["watershed_id", "block_id", *feature_columns]],
        on=["watershed_id", "block_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(block_features_df) != len(block_lookup_df):
        raise ValueError(
            "Join coverage mismatch between block_index_lookup.parquet and blocks.parquet"
        )

    event_frame = manifest_df.merge(
        events_df[["watershed_id", "event_id", "event_key", "resolved_event_path", "T", "F"]],
        on=["watershed_id", "event_id", "event_key"],
        how="inner",
        validate="one_to_one",
    )
    if len(event_frame) != len(manifest_df):
        raise ValueError("Join coverage mismatch between labels_10m_manifest.parquet and events.csv")

    samples_df = event_frame.assign(_join=1).merge(
        block_features_df.assign(_join=1),
        on=["watershed_id", "_join"],
        how="inner",
        validate="one_to_many",
    ).drop(columns="_join")

    for column in feature_columns:
        samples_df[column] = pd.to_numeric(samples_df[column], errors="coerce")
    if samples_df[feature_columns].isna().any().any():
        raise ValueError("blocks.parquet contains non-numeric or missing feature values in selected feature columns")

    splits = split_samples_by_event(
        merged_df=samples_df,
        test_events=test_events,
        val_fraction=val_fraction,
        seed=seed,
    )

    events_for_loading = (
        pd.concat(
            [
                splits.train_df[["event_key", "resolved_event_path", "T", "F"]],
                splits.val_df[["event_key", "resolved_event_path", "T", "F"]],
                splits.test_df[["event_key", "resolved_event_path", "T", "F"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["event_key"])
        .reset_index(drop=True)
    )
    event_arrays = _load_and_validate_event_arrays(events_for_loading)
    peak_manifest = event_frame[["event_key", "resolved_peak_path"]].drop_duplicates(subset=["event_key"]).reset_index(drop=True)
    peak_grids = _load_peak_grids(peak_manifest)

    train_event_keys = sorted(splits.train_df["event_key"].unique().tolist())
    event_mean, event_std = _fit_event_normalization(train_event_keys, event_arrays)
    block_mean, block_std = _fit_block_normalization(splits.train_df, feature_columns)

    normalized_events = _normalize_event_arrays(event_arrays, event_mean, event_std)
    normalized_blocks = _build_block_feature_map(block_features_df, feature_columns, block_mean, block_std)
    block_windows, final_target_shape = _compute_block_windows(
        block_index_grid=block_index_grid,
        block_lookup_df=block_lookup_df,
        target_rows=target_rows,
        target_cols=target_cols,
    )

    sample_event_shape = next(iter(normalized_events.values())).shape

    return BlockwiseMatrixDataBundle(
        train_dataset=BlockwiseFloodMatrixDataset(
            splits.train_df,
            normalized_events,
            normalized_blocks,
            peak_grids,
            block_index_grid,
            block_windows,
            final_target_shape,
        ),
        val_dataset=BlockwiseFloodMatrixDataset(
            splits.val_df,
            normalized_events,
            normalized_blocks,
            peak_grids,
            block_index_grid,
            block_windows,
            final_target_shape,
        ),
        test_dataset=BlockwiseFloodMatrixDataset(
            splits.test_df,
            normalized_events,
            normalized_blocks,
            peak_grids,
            block_index_grid,
            block_windows,
            final_target_shape,
        ),
        feature_columns=feature_columns,
        event_shape=sample_event_shape,
        target_shape=final_target_shape,
        normalization=NormalizationStats(
            event_mean=event_mean,
            event_std=event_std,
            block_mean=block_mean,
            block_std=block_std,
        ),
        splits=splits,
    )