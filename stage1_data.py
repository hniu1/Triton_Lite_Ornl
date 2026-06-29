"""Streaming data layer for the timestamp-conditioned Stage-1 surrogate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import netCDF4 as nc4
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

from blockwise_data import (
    _build_block_feature_map,
    _fit_block_normalization,
    _fit_event_normalization,
    _load_and_validate_event_arrays,
    _normalize_event_arrays,
    _resolve_event_path,
    event_key,
)
from blockwise_matrix_data import BlockWindow, _compute_block_windows


SampleKey = Tuple[int, int, int]  # event row, timestamp, block position


@dataclass
class Stage1Normalization:
    event_mean: np.ndarray
    event_std: np.ndarray
    block_mean: np.ndarray
    block_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray


@dataclass
class Stage1DataBundle:
    train_dataset: "Stage1TimestampDataset"
    val_dataset: "Stage1TimestampDataset"
    test_dataset: "Stage1TimestampDataset"
    train_sampler: "SpatiotemporalBatchSampler"
    val_sampler: "SpatiotemporalBatchSampler"
    test_sampler: "SpatiotemporalBatchSampler"
    normalization: Stage1Normalization
    feature_columns: List[str]
    event_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    static_channels: int
    component_semantics: str
    variable_names: Dict[str, str]
    split_events: Dict[str, List[str]]


def _center_pad(patch: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float32)
    row0 = (rows - patch.shape[0]) // 2
    col0 = (cols - patch.shape[1]) // 2
    out[row0 : row0 + patch.shape[0], col0 : col0 + patch.shape[1]] = patch
    return out


def _resolve_existing(raw: str, anchors: Sequence[Path]) -> Path:
    candidate = Path(raw)
    for path in [candidate, *(anchor / candidate for anchor in anchors)]:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve path: {raw}")


def _split_event_ids(
    event_ids: Sequence[str],
    test_events: Optional[Sequence[str]],
    val_fraction: float,
    seed: int,
) -> Dict[str, List[str]]:
    event_ids = sorted(set(str(value) for value in event_ids))
    if len(event_ids) < 3:
        raise ValueError("At least three dynamic events are required")
    if test_events:
        requested = set(test_events)
        missing = requested - set(event_ids)
        if missing:
            raise ValueError(f"Requested test events are unavailable: {sorted(missing)}")
        test = sorted(requested)
        remaining = [value for value in event_ids if value not in requested]
    else:
        remaining, test = train_test_split(event_ids, test_size=0.2, random_state=seed)
    train, val = train_test_split(remaining, test_size=val_fraction, random_state=seed)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


class Stage1TimestampDataset(Dataset):
    """Dataset indexed by explicit ``(event, time, block)`` sample keys.

    netCDF handles are opened lazily per DataLoader worker. Samples in a batch
    are deliberately grouped by event/time and spatial block order so HDF5's
    decompressed chunk cache can be reused.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        event_arrays: Dict[str, np.ndarray],
        block_rows: pd.DataFrame,
        block_feature_map: Dict[Tuple[str, str], np.ndarray],
        block_index_grid: np.ndarray,
        block_windows: Dict[int, BlockWindow],
        static_features_path: Path,
        static_masks_path: Path,
        static_mean: np.ndarray,
        static_std: np.ndarray,
        variable_names: Dict[str, str],
        target_shape: Tuple[int, int],
        wet_threshold: float,
        netcdf_chunk_cache_mb: int = 256,
    ) -> None:
        self.events = events.reset_index(drop=True)
        self.event_arrays = event_arrays
        self.block_rows = block_rows.reset_index(drop=True)
        self.block_feature_map = block_feature_map
        self.block_index_grid = block_index_grid
        self.block_windows = block_windows
        self.static_features_path = Path(static_features_path)
        self.static_masks_path = Path(static_masks_path)
        self.static_mean = static_mean.astype(np.float32)
        self.static_std = static_std.astype(np.float32)
        self.variable_names = variable_names
        self.target_rows, self.target_cols = target_shape
        self.wet_threshold = float(wet_threshold)
        self.netcdf_chunk_cache_bytes = int(netcdf_chunk_cache_mb * 1024 * 1024)
        self._handles: Dict[str, nc4.Dataset] = {}
        self._static_features = None
        self._static_masks = None

    def __len__(self) -> int:
        return len(self.events) * int(self.events["n_times"].max()) * len(self.block_rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_static_features"] = None
        state["_static_masks"] = None
        return state

    def _ensure_static(self) -> None:
        if self._static_features is None:
            self._static_features = np.load(self.static_features_path, mmap_mode="r")
            self._static_masks = np.load(self.static_masks_path, mmap_mode="r")

    def _handle(self, path: str) -> nc4.Dataset:
        if path not in self._handles:
            ds = nc4.Dataset(path, "r")
            for name in (
                self.variable_names["depth"],
                self.variable_names["component_x"],
                self.variable_names["component_y"],
            ):
                variable = ds.variables[name]
                nelems = max(1, self.netcdf_chunk_cache_bytes // max(1, variable.dtype.itemsize))
                variable.set_var_chunk_cache(self.netcdf_chunk_cache_bytes, nelems, 0.75)
            self._handles[path] = ds
        return self._handles[path]

    @staticmethod
    def _values(array) -> np.ndarray:
        if np.ma.isMaskedArray(array):
            values = array.filled(0.0)
        else:
            values = np.asarray(array)
        values = np.asarray(values, dtype=np.float32)
        values[~np.isfinite(values)] = 0.0
        return values

    def __getitem__(self, key: SampleKey):
        event_position, time_index, block_position = (int(value) for value in key)
        event = self.events.iloc[event_position]
        block = self.block_rows.iloc[block_position]
        block_index = int(block["block_index"])
        window = self.block_windows[block_index]
        ds = self._handle(str(event["resolved_netcdf_path"]))

        slices = (
            time_index,
            slice(window.row_start, window.row_stop),
            slice(window.col_start, window.col_stop),
        )
        depth = self._values(ds.variables[self.variable_names["depth"]][slices])
        component_x = self._values(ds.variables[self.variable_names["component_x"]][slices])
        component_y = self._values(ds.variables[self.variable_names["component_y"]][slices])

        mask_patch = (
            self.block_index_grid[
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ]
            == block_index
        ).astype(np.float32)
        depth *= mask_patch
        component_x *= mask_patch
        component_y *= mask_patch

        self._ensure_static()
        static = np.asarray(self._static_features[block_index], dtype=np.float32)
        static = (static - self.static_mean[:, None, None]) / self.static_std[:, None, None]
        mask = np.asarray(self._static_masks[block_index], dtype=np.float32)
        static *= mask[None, :, :]

        key_name = str(event["event_key"])
        block_key = (str(block["watershed_id"]), str(block["block_id"]))
        time_fraction = float(time_index) / max(int(event["n_times"]) - 1, 1)
        time_hours = float(event["time_start"]) + time_index * float(event["time_step"])

        return {
            "event": torch.from_numpy(self.event_arrays[key_name].copy()),
            "time_index": torch.tensor(time_index, dtype=torch.long),
            "time_features": torch.tensor(
                [
                    time_fraction,
                    math.sin(2.0 * math.pi * time_fraction),
                    math.cos(2.0 * math.pi * time_fraction),
                    time_hours / max(float(event["time_end"]), 1.0),
                ],
                dtype=torch.float32,
            ),
            "block_features": torch.from_numpy(self.block_feature_map[block_key].copy()),
            "static": torch.from_numpy(static.copy()),
            "mask": torch.from_numpy(mask.copy()),
            "depth": torch.from_numpy(_center_pad(depth, self.target_rows, self.target_cols)),
            "component_x": torch.from_numpy(
                _center_pad(component_x, self.target_rows, self.target_cols)
            ),
            "component_y": torch.from_numpy(
                _center_pad(component_y, self.target_rows, self.target_cols)
            ),
            "event_id": str(event["event_id"]),
            "block_id": str(block["block_id"]),
        }


class SpatiotemporalBatchSampler(Sampler[List[SampleKey]]):
    """Yield same-event, same-time, spatially local block batches."""

    def __init__(
        self,
        n_events: int,
        n_times: Sequence[int],
        n_blocks: int,
        batch_size: int,
        batches_per_epoch: Optional[int],
        time_stride: int,
        seed: int,
        shuffle: bool,
        time_weights: Optional[Sequence[np.ndarray]] = None,
        block_start_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.n_events = int(n_events)
        self.n_times = [int(value) for value in n_times]
        self.n_blocks = int(n_blocks)
        self.batch_size = int(batch_size)
        self.time_stride = max(1, int(time_stride))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.time_weights = time_weights
        self.block_start_weights = block_start_weights
        self.epoch = 0
        if batches_per_epoch is None:
            total_time_groups = sum(
                len(range(0, event_times, self.time_stride)) for event_times in self.n_times
            )
            batches_per_spatial_sweep = math.ceil(self.n_blocks / self.batch_size)
            self.batches_per_epoch = total_time_groups * batches_per_spatial_sweep
        else:
            self.batches_per_epoch = int(batches_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[SampleKey]]:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        if self.shuffle:
            for _ in range(self.batches_per_epoch):
                event_position = int(rng.integers(0, self.n_events))
                valid_times = np.arange(
                    0, self.n_times[event_position], self.time_stride, dtype=np.int64
                )
                if self.time_weights is None:
                    time_index = int(rng.choice(valid_times))
                else:
                    probabilities = self.time_weights[event_position][valid_times]
                    probabilities = probabilities / probabilities.sum()
                    time_index = int(rng.choice(valid_times, p=probabilities))
                max_starts = max(1, self.n_blocks - self.batch_size + 1)
                if self.block_start_weights is None:
                    start = int(rng.integers(0, max_starts))
                else:
                    probabilities = self.block_start_weights[:max_starts]
                    probabilities = probabilities / probabilities.sum()
                    start = int(rng.choice(max_starts, p=probabilities))
                yield [
                    (event_position, time_index, block_position)
                    for block_position in range(start, min(start + self.batch_size, self.n_blocks))
                ]
            return

        candidates: List[Tuple[int, int, int]] = []
        for event_position in range(self.n_events):
            for time_index in range(0, self.n_times[event_position], self.time_stride):
                for start in range(0, self.n_blocks, self.batch_size):
                    candidates.append((event_position, time_index, start))
        if not candidates:
            return
        if self.batches_per_epoch >= len(candidates):
            selected = range(len(candidates))
        else:
            selected = np.linspace(
                0, len(candidates) - 1, self.batches_per_epoch, dtype=np.int64
            ).tolist()
        for candidate_index in selected:
            event_position, time_index, start = candidates[int(candidate_index)]
            yield [
                (event_position, time_index, block_position)
                for block_position in range(start, min(start + self.batch_size, self.n_blocks))
            ]


def prepare_stage1_data(
    manifest_dir: Path,
    events_csv: Path,
    blocks_parquet: Path,
    labels_10m_dir: Path,
    static_rasters_dir: Path,
    base_dir: Path,
    test_events: Optional[Sequence[str]],
    val_fraction: float,
    seed: int,
    batch_size: int,
    train_batches_per_epoch: int,
    eval_batches: int,
    train_time_stride: int,
    eval_time_stride: int,
    wet_threshold: float,
    feature_columns: Optional[Sequence[str]] = None,
    netcdf_chunk_cache_mb: int = 256,
) -> Stage1DataBundle:
    manifest_dir = manifest_dir.resolve()
    manifest = pd.read_parquet(manifest_dir / "dynamic_manifest.parquet")
    metadata = json.loads((manifest_dir / "dynamic_metadata.json").read_text())
    events = pd.read_csv(events_csv.resolve())
    blocks = pd.read_parquet(blocks_parquet.resolve())
    block_lookup = pd.read_parquet(labels_10m_dir.resolve() / "block_index_lookup.parquet")
    block_index_grid = np.load(labels_10m_dir.resolve() / "block_index_10m.npy", mmap_mode="r")
    static_lookup = pd.read_parquet(static_rasters_dir.resolve() / "block_static_lookup.parquet")

    if not block_lookup.equals(static_lookup[block_lookup.columns]):
        raise ValueError("Dynamic block lookup and static-raster lookup are not identical")

    if feature_columns is None:
        feature_columns = [
            column for column in blocks.columns if column not in {"watershed_id", "block_id"}
        ]
    feature_columns = list(feature_columns)
    block_rows = block_lookup.merge(
        blocks[["watershed_id", "block_id", *feature_columns]],
        on=["watershed_id", "block_id"],
        validate="one_to_one",
    ).sort_values("block_index").reset_index(drop=True)

    manifest = manifest.merge(
        events[["event_id", "watershed_id", "path_to_X_event", "T", "F"]],
        on=["event_id", "watershed_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_events"),
    )
    manifest["event_key"] = [
        event_key(watershed_id, event_id)
        for watershed_id, event_id in zip(manifest["watershed_id"], manifest["event_id"])
    ]
    manifest["resolved_event_path"] = [
        str(
            _resolve_event_path(
                raw_path,
                base_dir.resolve(),
                events_csv.resolve(),
                watershed_id,
                event_id,
            )
        )
        for raw_path, watershed_id, event_id in zip(
            manifest["path_to_X_event_events"].fillna(manifest["path_to_X_event"]),
            manifest["watershed_id"],
            manifest["event_id"],
        )
    ]
    manifest["resolved_netcdf_path"] = [
        str(_resolve_existing(raw, [manifest_dir, base_dir.resolve()]))
        for raw in manifest["path_to_netcdf"]
    ]

    split_events = _split_event_ids(
        manifest["event_id"].tolist(), test_events, val_fraction, seed
    )
    split_frames = {
        name: manifest.loc[manifest["event_id"].isin(ids)].reset_index(drop=True)
        for name, ids in split_events.items()
    }

    event_load_frame = manifest[
        ["event_key", "resolved_event_path", "T", "F"]
    ].drop_duplicates("event_key")
    event_arrays = _load_and_validate_event_arrays(event_load_frame)
    train_keys = split_frames["train"]["event_key"].tolist()
    event_mean, event_std = _fit_event_normalization(train_keys, event_arrays)
    normalized_events = _normalize_event_arrays(event_arrays, event_mean, event_std)

    train_block_frame = block_rows.copy()
    block_mean, block_std = _fit_block_normalization(train_block_frame, feature_columns)
    block_feature_map = _build_block_feature_map(
        block_rows, feature_columns, block_mean, block_std
    )

    stats = json.loads(
        (static_rasters_dir.resolve() / "block_static_feature_stats.json").read_text()
    )
    static_mean = np.asarray(stats["mean"], dtype=np.float32)
    static_std = np.asarray(stats["std"], dtype=np.float32)
    static_std[static_std < 1e-6] = 1.0
    static_path = static_rasters_dir.resolve() / "block_static_features.npy"
    static_masks_path = static_rasters_dir.resolve() / "block_static_masks.npy"
    static_shape = np.load(static_path, mmap_mode="r").shape

    # Bias training toward active portions of the hydrograph while retaining a
    # 25% uniform component so rising limbs, recession, and quiet periods remain
    # represented. This changes sampling frequency only; it does not expose
    # future forcing to the causal encoder.
    event_time_weights: Dict[str, np.ndarray] = {}
    for key, array in event_arrays.items():
        instantaneous = np.maximum(array, 0.0).sum(axis=1, dtype=np.float64)
        kernel = np.ones(49, dtype=np.float64) / 49.0
        smoothed = np.convolve(instantaneous, kernel, mode="same")
        if smoothed.sum() <= 0:
            event_time_weights[key] = np.ones(array.shape[0], dtype=np.float64)
        else:
            event_time_weights[key] = (
                0.25 / array.shape[0] + 0.75 * smoothed / smoothed.sum()
            )

    # Prefer blocks with larger flow accumulation, again mixed with a uniform
    # component. Computing this once from the memory-mapped static tensor is far
    # cheaper than scanning dynamic labels for wet-cell occupancy.
    feature_names_path = static_rasters_dir.resolve() / "block_static_feature_names.json"
    static_feature_names = json.loads(feature_names_path.read_text())["feature_names"]
    raw_static = np.load(static_path, mmap_mode="r")
    if "flow_acc" in static_feature_names:
        flow_index = static_feature_names.index("flow_acc")
        block_activity = np.log1p(
            np.maximum(raw_static[:, flow_index], 0.0).max(axis=(1, 2)).astype(np.float64)
        )
        block_activity = block_activity + 1e-12
        block_probabilities = (
            0.25 / len(block_activity) + 0.75 * block_activity / block_activity.sum()
        )
    else:
        block_probabilities = np.full(len(block_rows), 1.0 / len(block_rows))
    batch_kernel = np.ones(batch_size, dtype=np.float64)
    block_start_weights = np.convolve(block_probabilities, batch_kernel, mode="valid")

    block_windows, target_shape = _compute_block_windows(
        block_index_grid,
        block_lookup,
        target_rows=int(static_shape[2]),
        target_cols=int(static_shape[3]),
    )

    normalization = Stage1Normalization(
        event_mean=event_mean,
        event_std=event_std,
        block_mean=block_mean,
        block_std=block_std,
        static_mean=static_mean,
        static_std=static_std,
    )

    datasets = {}
    samplers = {}
    for name in ("train", "val", "test"):
        frame = split_frames[name]
        datasets[name] = Stage1TimestampDataset(
            events=frame,
            event_arrays=normalized_events,
            block_rows=block_rows,
            block_feature_map=block_feature_map,
            block_index_grid=block_index_grid,
            block_windows=block_windows,
            static_features_path=static_path,
            static_masks_path=static_masks_path,
            static_mean=static_mean,
            static_std=static_std,
            variable_names=metadata["variable_names"],
            target_shape=target_shape,
            wet_threshold=wet_threshold,
            netcdf_chunk_cache_mb=netcdf_chunk_cache_mb,
        )
        samplers[name] = SpatiotemporalBatchSampler(
            n_events=len(frame),
            n_times=frame["n_times"].tolist(),
            n_blocks=len(block_rows),
            batch_size=batch_size,
            batches_per_epoch=train_batches_per_epoch if name == "train" else eval_batches,
            time_stride=train_time_stride if name == "train" else eval_time_stride,
            seed=seed + {"train": 0, "val": 1000, "test": 2000}[name],
            shuffle=name == "train",
            time_weights=(
                [event_time_weights[key] for key in frame["event_key"]]
                if name == "train"
                else None
            ),
            block_start_weights=block_start_weights if name == "train" else None,
        )

    sample_event_shape = next(iter(normalized_events.values())).shape
    return Stage1DataBundle(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        train_sampler=samplers["train"],
        val_sampler=samplers["val"],
        test_sampler=samplers["test"],
        normalization=normalization,
        feature_columns=feature_columns,
        event_shape=sample_event_shape,
        target_shape=target_shape,
        static_channels=int(static_shape[1]),
        component_semantics=str(metadata["component_semantics"]),
        variable_names=dict(metadata["variable_names"]),
        split_events=split_events,
    )
