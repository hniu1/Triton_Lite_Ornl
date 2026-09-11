"""Streaming data layer for the timestamp-conditioned Stage-1 surrogate."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
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
        max_open_netcdf_handles: int = 8,
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
        self.max_open_netcdf_handles = max(1, int(max_open_netcdf_handles))
        self._handles: OrderedDict[str, nc4.Dataset] = OrderedDict()
        self._static_features = None
        self._static_masks = None

    def __len__(self) -> int:
        return len(self.events) * int(self.events["n_times"].max()) * len(self.block_rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_static_features"] = None
        state["_static_masks"] = None
        return state

    def _ensure_static(self) -> None:
        if self._static_features is None:
            self._static_features = np.load(self.static_features_path, mmap_mode="r")
            self._static_masks = np.load(self.static_masks_path, mmap_mode="r")

    def _handle(self, path: str) -> nc4.Dataset:
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        while len(self._handles) >= self.max_open_netcdf_handles:
            _, old_ds = self._handles.popitem(last=False)
            old_ds.close()
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
        return ds

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
        minimum_time_index: int = 0,
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
        self.minimum_time_index = max(0, int(minimum_time_index))
        self.epoch = 0
        if batches_per_epoch is None:
            total_time_groups = sum(
                sum(
                    time_index >= self.minimum_time_index
                    for time_index in range(0, event_times, self.time_stride)
                )
                for event_times in self.n_times
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
                valid_times = valid_times[valid_times >= self.minimum_time_index]
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
                if time_index < self.minimum_time_index:
                    continue
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


class LabelAwareBatchSampler(Sampler[List[SampleKey]]):
    """Sample batches around labeled anchors with category and phase quotas."""

    def __init__(
        self,
        candidates: pd.DataFrame,
        event_ids: Sequence[str],
        n_times: Sequence[int],
        n_blocks: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
        category_fractions: Dict[str, float],
        phase_fractions: Dict[str, float],
    ) -> None:
        required = {"event_id", "time_index", "anchor_block", "category", "phase"}
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(f"Sampling index is missing columns: {sorted(missing)}")
        self.event_ids = [str(value) for value in event_ids]
        event_positions = {event_id: i for i, event_id in enumerate(self.event_ids)}
        frame = candidates.loc[candidates["event_id"].isin(event_positions)].copy()
        frame["event_position"] = frame["event_id"].map(event_positions).astype(np.int32)
        valid = (
            (frame["anchor_block"] >= 0)
            & (frame["anchor_block"] < int(n_blocks))
            & (frame["time_index"] >= 0)
        )
        for event_position, n_event_times in enumerate(n_times):
            invalid_time = (frame["event_position"] == event_position) & (
                frame["time_index"] >= int(n_event_times)
            )
            valid &= ~invalid_time
        frame = frame.loc[valid].reset_index(drop=True)
        if frame.empty:
            raise ValueError("Sampling index has no candidates for the training events")

        self.n_blocks = int(n_blocks)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.category_fractions = self._validate_fractions(
            category_fractions, "category"
        )
        self.phase_fractions = self._validate_fractions(phase_fractions, "phase")
        self.pools: Dict[Tuple[str, str, int], np.ndarray] = {}
        for key, group in frame.groupby(
            ["category", "phase", "event_position"], observed=True, sort=False
        ):
            self.pools[(str(key[0]), str(key[1]), int(key[2]))] = group[
                ["time_index", "anchor_block"]
            ].to_numpy(dtype=np.int32)
        self.available_categories = sorted({key[0] for key in self.pools})

    @staticmethod
    def _validate_fractions(values: Dict[str, float], name: str) -> Dict[str, float]:
        if any(float(value) < 0 for value in values.values()):
            raise ValueError(f"{name.title()} fractions cannot be negative")
        result = {str(key): float(value) for key, value in values.items() if value > 0}
        if not result:
            raise ValueError(f"At least one positive {name} fraction is required")
        total = sum(result.values())
        return {key: value / total for key, value in result.items()}

    @staticmethod
    def _choice(rng, names: Sequence[str], fractions: Dict[str, float]) -> str:
        weights = np.asarray([fractions.get(name, 0.0) for name in names], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(names), dtype=np.float64)
        weights /= weights.sum()
        return str(rng.choice(names, p=weights))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[SampleKey]]:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        for _ in range(self.batches_per_epoch):
            category = self._choice(
                rng, self.available_categories, self.category_fractions
            )
            phases = sorted({key[1] for key in self.pools if key[0] == category})
            phase = self._choice(rng, phases, self.phase_fractions)
            event_positions = sorted(
                key[2]
                for key in self.pools
                if key[0] == category and key[1] == phase
            )
            event_position = int(rng.choice(event_positions))
            pool = self.pools[(category, phase, event_position)]
            time_index, anchor_block = pool[int(rng.integers(0, len(pool)))]
            start = min(
                max(0, int(anchor_block) - self.batch_size // 2),
                max(0, self.n_blocks - self.batch_size),
            )
            yield [
                (event_position, int(time_index), block_position)
                for block_position in range(start, min(start + self.batch_size, self.n_blocks))
            ]


class BalancedLabelBatchSampler(Sampler[List[SampleKey]]):
    """Build unique, same-event/time batches from individually labeled blocks."""

    def __init__(
        self,
        candidates: pd.DataFrame,
        event_ids: Sequence[str],
        n_times: Sequence[int],
        n_blocks: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
        category_fractions: Dict[str, float],
        phase_fractions: Dict[str, float],
        target_wet_cell_fraction: float,
        strict_category_quotas: bool = False,
    ) -> None:
        required = {
            "event_id",
            "time_index",
            "anchor_block",
            "category",
            "phase",
            "wet_fraction",
        }
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(f"Sampling index is missing columns: {sorted(missing)}")
        if not 0 <= target_wet_cell_fraction <= 1:
            raise ValueError("target_wet_cell_fraction must be between 0 and 1")
        self.event_ids = [str(value) for value in event_ids]
        event_positions = {event_id: i for i, event_id in enumerate(self.event_ids)}
        frame = candidates.loc[candidates["event_id"].isin(event_positions)].copy()
        frame["event_position"] = frame["event_id"].map(event_positions).astype(np.int32)
        frame = frame.drop_duplicates(["event_position", "time_index", "anchor_block"])
        valid = (
            (frame["anchor_block"] >= 0)
            & (frame["anchor_block"] < int(n_blocks))
            & (frame["time_index"] >= 0)
        )
        for event_position, n_event_times in enumerate(n_times):
            valid &= ~(
                (frame["event_position"] == event_position)
                & (frame["time_index"] >= int(n_event_times))
            )
        frame = frame.loc[valid].reset_index(drop=True)
        if frame.empty:
            raise ValueError("Sampling index has no candidates for the training events")

        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.target_wet_cell_fraction = float(target_wet_cell_fraction)
        self.strict_category_quotas = bool(strict_category_quotas)
        self.category_fractions = LabelAwareBatchSampler._validate_fractions(
            category_fractions, "category"
        )
        self.phase_fractions = LabelAwareBatchSampler._validate_fractions(
            phase_fractions, "phase"
        )
        self.category_counts = self._quota_counts(
            self.category_fractions, self.batch_size
        )
        self.groups: Dict[Tuple[int, int], Dict[str, object]] = {}
        self.pools: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
        for (event_position, time_index), group in frame.groupby(
            ["event_position", "time_index"], observed=True, sort=False
        ):
            if len(group) < self.batch_size:
                continue
            wet_fractions = group["wet_fraction"].to_numpy(dtype=np.float32)
            categories = group["category"].astype(str).to_numpy()
            if self.strict_category_quotas:
                quota_indices = []
                feasible = True
                for category, requested in self.category_counts.items():
                    available = np.flatnonzero(categories == category)
                    if len(available) < requested:
                        feasible = False
                        break
                    if requested:
                        order = available[np.argsort(-wet_fractions[available])]
                        quota_indices.extend(order[:requested].tolist())
                if not feasible:
                    continue
                maximum_possible = float(wet_fractions[quota_indices].mean())
            else:
                maximum_possible = np.partition(wet_fractions, -self.batch_size)[
                    -self.batch_size:
                ].mean()
            if maximum_possible + 1e-8 < self.target_wet_cell_fraction:
                continue
            phases = group["phase"].astype(str).mode()
            phase = str(phases.iloc[0])
            key = (int(event_position), int(time_index))
            self.groups[key] = {
                "blocks": group["anchor_block"].to_numpy(dtype=np.int32),
                "categories": categories,
                "wet_fractions": wet_fractions,
                "phase": phase,
            }
            self.pools.setdefault(phase, {}).setdefault(int(event_position), []).append(key)
        if not self.groups:
            raise ValueError(
                "Sampling index has no event/time groups capable of meeting the requested "
                "batch size and wet-cell target; rebuild M4 with more candidates per event"
            )
        self.available_phases = sorted(self.pools)

    @staticmethod
    def _quota_counts(fractions: Dict[str, float], batch_size: int) -> Dict[str, int]:
        names = list(fractions)
        raw = np.asarray([fractions[name] * batch_size for name in names])
        counts = np.floor(raw).astype(int)
        remainder = batch_size - int(counts.sum())
        order = np.argsort(-(raw - counts))
        for index in order[:remainder]:
            counts[index] += 1
        return {name: int(value) for name, value in zip(names, counts)}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _select_group(self, rng):
        phase = LabelAwareBatchSampler._choice(
            rng, self.available_phases, self.phase_fractions
        )
        event_positions = sorted(self.pools[phase])
        event_position = int(rng.choice(event_positions))
        keys = self.pools[phase][event_position]
        return keys[int(rng.integers(0, len(keys)))]

    def _select_blocks(self, rng, group: Dict[str, object]) -> np.ndarray:
        blocks = group["blocks"]
        categories = group["categories"]
        wet_fractions = group["wet_fractions"]
        selected: List[int] = []
        selected_set = set()
        for category, requested in self.category_counts.items():
            available = np.flatnonzero(categories == category)
            if len(available) == 0 or requested == 0:
                continue
            count = min(requested, len(available))
            chosen = rng.choice(available, size=count, replace=False).tolist()
            selected.extend(int(value) for value in chosen)
            selected_set.update(int(value) for value in chosen)
        if len(selected) < self.batch_size:
            if self.strict_category_quotas:
                raise RuntimeError("Strict category quotas became infeasible after initialization")
            available = np.asarray(
                [index for index in range(len(blocks)) if index not in selected_set],
                dtype=np.int64,
            )
            # Prefer informative cells when a requested category is unavailable.
            jitter = rng.random(len(available)) * 1e-6
            order = available[np.argsort(-(wet_fractions[available] + jitter))]
            needed = self.batch_size - len(selected)
            selected.extend(int(value) for value in order[:needed])

        selected = selected[: self.batch_size]
        selected_set = set(selected)
        current = float(wet_fractions[selected].mean())
        if current + 1e-8 < self.target_wet_cell_fraction:
            if self.strict_category_quotas:
                for category in self.category_counts:
                    low = sorted(
                        [index for index in selected if categories[index] == category],
                        key=lambda index: wet_fractions[index],
                    )
                    high = sorted(
                        [index for index in range(len(blocks)) if index not in selected_set and categories[index] == category],
                        key=lambda index: wet_fractions[index], reverse=True,
                    )
                    for low_index, high_index in zip(low, high):
                        if wet_fractions[high_index] <= wet_fractions[low_index]:
                            break
                        position = selected.index(low_index)
                        selected[position] = high_index
                        selected_set.remove(low_index)
                        selected_set.add(high_index)
                        current += float(
                            (wet_fractions[high_index] - wet_fractions[low_index])
                            / self.batch_size
                        )
                        if current + 1e-8 >= self.target_wet_cell_fraction:
                            break
                    if current + 1e-8 >= self.target_wet_cell_fraction:
                        break
            else:
                unselected = [index for index in range(len(blocks)) if index not in selected_set]
                high = sorted(unselected, key=lambda index: wet_fractions[index], reverse=True)
                low = sorted(selected, key=lambda index: wet_fractions[index])
                for low_index, high_index in zip(low, high):
                    if wet_fractions[high_index] <= wet_fractions[low_index]:
                        break
                    position = selected.index(low_index)
                    selected[position] = high_index
                    current += float(
                        (wet_fractions[high_index] - wet_fractions[low_index])
                        / self.batch_size
                    )
                    if current + 1e-8 >= self.target_wet_cell_fraction:
                        break
        if current + 1e-8 < self.target_wet_cell_fraction:
            raise RuntimeError("Sampler failed to reach its configured wet-cell target")
        return np.sort(blocks[np.asarray(selected, dtype=np.int64)])

    def __iter__(self) -> Iterator[List[SampleKey]]:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        for _ in range(self.batches_per_epoch):
            event_position, time_index = self._select_group(rng)
            blocks = self._select_blocks(rng, self.groups[(event_position, time_index)])
            yield [
                (int(event_position), int(time_index), int(block_position))
                for block_position in blocks
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
    max_open_netcdf_handles: int = 8,
    sampling_index_dir: Optional[Path] = None,
    sampling_category_fractions: Optional[Dict[str, float]] = None,
    sampling_phase_fractions: Optional[Dict[str, float]] = None,
    sampling_mode: str = "anchor",
    sampling_target_wet_cell_fraction: float = 0.0,
    sampling_strict_category_quotas: bool = False,
    minimum_time_index: int = 0,
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
            max_open_netcdf_handles=max_open_netcdf_handles,
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
            minimum_time_index=minimum_time_index,
        )

    if sampling_index_dir is not None:
        sampling_index_dir = sampling_index_dir.resolve()
        candidates = pd.read_parquet(sampling_index_dir / "sampling_candidates.parquet")
        candidates = candidates.loc[
            candidates["time_index"] >= int(minimum_time_index)
        ].reset_index(drop=True)
        sampling_metadata = json.loads(
            (sampling_index_dir / "sampling_metadata.json").read_text()
        )
        if not np.isclose(float(sampling_metadata["wet_threshold"]), wet_threshold):
            raise ValueError(
                "Sampling-index wet threshold does not match training wet threshold: "
                f"{sampling_metadata['wet_threshold']} != {wet_threshold}"
            )
        sampler_kwargs = {
            "candidates": candidates,
            "event_ids": split_frames["train"]["event_id"].tolist(),
            "n_times": split_frames["train"]["n_times"].tolist(),
            "n_blocks": len(block_rows),
            "batch_size": batch_size,
            "batches_per_epoch": train_batches_per_epoch,
            "seed": seed,
            "category_fractions": sampling_category_fractions
            or {"dry": 0.15, "boundary": 0.25, "wet": 0.40, "deep": 0.20},
            "phase_fractions": sampling_phase_fractions
            or {"quiet": 0.15, "rising": 0.30, "peak": 0.30, "recession": 0.25},
        }
        if sampling_mode == "anchor":
            samplers["train"] = LabelAwareBatchSampler(**sampler_kwargs)
        elif sampling_mode == "balanced_batch":
            samplers["train"] = BalancedLabelBatchSampler(
                **sampler_kwargs,
                target_wet_cell_fraction=sampling_target_wet_cell_fraction,
                strict_category_quotas=sampling_strict_category_quotas,
            )
        else:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

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
