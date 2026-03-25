from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


REQUIRED_EVENT_COLUMNS = {"event_id", "watershed_id", "path_to_X_event", "T", "F"}
REQUIRED_BLOCK_COLUMNS = {"watershed_id", "block_id"}
REQUIRED_LABEL_COLUMNS = {"event_id", "watershed_id", "block_id", "y"}


@dataclass
class NormalizationStats:
    event_mean: np.ndarray
    event_std: np.ndarray
    block_mean: np.ndarray
    block_std: np.ndarray


@dataclass
class BlockwiseSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class BlockwiseDataBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    feature_columns: List[str]
    event_shape: Tuple[int, int]
    normalization: NormalizationStats
    splits: BlockwiseSplit


class BlockwiseFloodDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        event_arrays: Dict[str, np.ndarray],
        block_feature_map: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.event_arrays = event_arrays
        self.block_feature_map = block_feature_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        sample_key = (row["watershed_id"], row["block_id"])
        event_tensor = self.event_arrays[row["event_key"]]
        block_features = self.block_feature_map[sample_key]
        target = np.float32(row["y"])

        return (
            torch.from_numpy(event_tensor.copy()),
            torch.from_numpy(block_features.copy()),
            torch.tensor(target, dtype=torch.float32),
        )


def event_key(watershed_id: str, event_id: str) -> str:
    return f"{watershed_id}::{event_id}"


def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _resolve_event_path(
    raw_path: str,
    base_dir: Path,
    events_csv_path: Path,
    watershed_id: str,
    event_id: str,
) -> Path:
    candidate = Path(raw_path)
    candidates = [
        candidate,
        base_dir / candidate,
        events_csv_path.parent / candidate,
        events_csv_path.parent / "events" / watershed_id / event_id / "X_event.npy",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not resolve X_event path '{raw_path}' for watershed_id='{watershed_id}', event_id='{event_id}'"
    )


def _load_and_validate_event_arrays(events_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    expected_shape: Optional[Tuple[int, int]] = None

    for row in events_df.itertuples(index=False):
        array = np.load(row.resolved_event_path).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D X_event array, got shape {array.shape} at {row.resolved_event_path}")

        if int(row.T) != array.shape[0] or int(row.F) != array.shape[1]:
            raise ValueError(
                f"events.csv shape mismatch for {row.event_key}: table says (T={row.T}, F={row.F}) "
                f"but array is {array.shape}"
            )

        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"All events must share one common shape for batching. Saw {array.shape} and {expected_shape}."
            )

        arrays[row.event_key] = array

    return arrays


def _fit_event_normalization(train_event_keys: Sequence[str], event_arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    train_arrays = [event_arrays[key] for key in train_event_keys]
    stacked = np.concatenate(train_arrays, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _fit_block_normalization(train_df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    unique_blocks = train_df[["watershed_id", "block_id", *feature_columns]].drop_duplicates(
        subset=["watershed_id", "block_id"]
    )
    values = unique_blocks.loc[:, feature_columns].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _normalize_event_arrays(
    event_arrays: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> Dict[str, np.ndarray]:
    normalized: Dict[str, np.ndarray] = {}
    for key, value in event_arrays.items():
        normalized[key] = ((value - mean) / std).astype(np.float32)
    return normalized


def _build_block_feature_map(
    blocks_df: pd.DataFrame,
    feature_columns: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> Dict[Tuple[str, str], np.ndarray]:
    mapping: Dict[Tuple[str, str], np.ndarray] = {}
    for row in blocks_df.itertuples(index=False):
        key = (row.watershed_id, row.block_id)
        vector = np.asarray([getattr(row, column) for column in feature_columns], dtype=np.float32)
        mapping[key] = ((vector - mean) / std).astype(np.float32)
    return mapping


def _normalize_event_identifier(identifier: str) -> str:
    value = str(identifier).strip()
    if not value:
        raise ValueError("Encountered empty event identifier in explicit split list")
    return value


def _select_explicit_event_keys(all_event_keys: Sequence[str], explicit_identifiers: Sequence[str]) -> List[str]:
    explicit_identifiers = [_normalize_event_identifier(value) for value in explicit_identifiers]
    selected: List[str] = []
    available_by_event_id: Dict[str, List[str]] = {}
    for key in all_event_keys:
        _, event_id = key.split("::", 1)
        available_by_event_id.setdefault(event_id, []).append(key)

    missing: List[str] = []
    for identifier in explicit_identifiers:
        if "::" in identifier:
            if identifier not in all_event_keys:
                missing.append(identifier)
            else:
                selected.append(identifier)
            continue

        matches = available_by_event_id.get(identifier, [])
        if not matches:
            missing.append(identifier)
            continue
        selected.extend(matches)

    if missing:
        raise ValueError(f"Requested event identifiers were not found: {missing}")

    return sorted(set(selected))


def split_samples_by_event(
    merged_df: pd.DataFrame,
    test_events: Optional[Sequence[str]],
    val_fraction: float,
    seed: int,
) -> BlockwiseSplit:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    all_event_keys = sorted(merged_df["event_key"].unique().tolist())
    if len(all_event_keys) < 3:
        raise ValueError("Need at least 3 unique events to make train/val/test splits")

    if test_events:
        test_keys = _select_explicit_event_keys(all_event_keys, test_events)
        remaining = [key for key in all_event_keys if key not in test_keys]
    else:
        remaining, test_keys = train_test_split(all_event_keys, test_size=0.2, random_state=seed, shuffle=True)

    if len(test_keys) == 0:
        raise ValueError("Test split is empty")
    if len(remaining) < 2:
        raise ValueError("Not enough events left after test split to create train and validation sets")

    train_keys, val_keys = train_test_split(remaining, test_size=val_fraction, random_state=seed, shuffle=True)
    if len(train_keys) == 0 or len(val_keys) == 0:
        raise ValueError("Train or validation split is empty")

    return BlockwiseSplit(
        train_df=merged_df.loc[merged_df["event_key"].isin(train_keys)].reset_index(drop=True),
        val_df=merged_df.loc[merged_df["event_key"].isin(val_keys)].reset_index(drop=True),
        test_df=merged_df.loc[merged_df["event_key"].isin(test_keys)].reset_index(drop=True),
    )


def load_blockwise_training_frame(
    events_csv: Path,
    blocks_parquet: Path,
    labels_parquet: Path,
    base_dir: Path,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    events_df = pd.read_csv(events_csv)
    blocks_df = pd.read_parquet(blocks_parquet)
    labels_df = pd.read_parquet(labels_parquet)

    _validate_columns(events_df, REQUIRED_EVENT_COLUMNS, "events.csv")
    _validate_columns(blocks_df, REQUIRED_BLOCK_COLUMNS, "blocks.parquet")
    _validate_columns(labels_df, REQUIRED_LABEL_COLUMNS, "labels.parquet")

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

    if feature_columns is None:
        feature_columns = [
            column
            for column in blocks_df.columns
            if column not in {"watershed_id", "block_id"}
        ]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No block feature columns were selected")

    missing_features = [column for column in feature_columns if column not in blocks_df.columns]
    if missing_features:
        raise ValueError(f"Requested block feature columns are missing from blocks.parquet: {missing_features}")

    merge_columns = ["watershed_id", "block_id", *feature_columns]
    merged_df = labels_df.merge(
        events_df[["watershed_id", "event_id", "event_key", "resolved_event_path", "T", "F"]],
        on=["watershed_id", "event_id"],
        how="inner",
        validate="many_to_one",
    ).merge(
        blocks_df[merge_columns],
        on=["watershed_id", "block_id"],
        how="inner",
        validate="many_to_one",
    )

    if len(merged_df) != len(labels_df):
        raise ValueError(
            f"Join coverage mismatch: labels has {len(labels_df)} rows but merged training frame has {len(merged_df)} rows"
        )

    merged_df["y"] = pd.to_numeric(merged_df["y"], errors="coerce")
    for column in feature_columns:
        merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce")

    if merged_df["y"].isna().any():
        raise ValueError("labels.parquet contains non-numeric or missing y values after join")
    if merged_df[feature_columns].isna().any().any():
        raise ValueError("blocks.parquet contains non-numeric or missing feature values in selected feature columns")

    return merged_df.sort_values(["watershed_id", "event_id", "block_id"]).reset_index(drop=True), blocks_df, feature_columns


def prepare_blockwise_datasets(
    events_csv: Path,
    blocks_parquet: Path,
    labels_parquet: Path,
    base_dir: Path,
    feature_columns: Optional[Sequence[str]],
    test_events: Optional[Sequence[str]],
    val_fraction: float,
    seed: int,
) -> BlockwiseDataBundle:
    merged_df, blocks_df, feature_columns = load_blockwise_training_frame(
        events_csv=events_csv,
        blocks_parquet=blocks_parquet,
        labels_parquet=labels_parquet,
        base_dir=base_dir,
        feature_columns=feature_columns,
    )

    splits = split_samples_by_event(
        merged_df=merged_df,
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

    train_event_keys = sorted(splits.train_df["event_key"].unique().tolist())
    event_mean, event_std = _fit_event_normalization(train_event_keys, event_arrays)
    block_mean, block_std = _fit_block_normalization(splits.train_df, feature_columns)

    normalized_events = _normalize_event_arrays(event_arrays, event_mean, event_std)
    normalized_blocks = _build_block_feature_map(blocks_df, feature_columns, block_mean, block_std)

    sample_event_shape = next(iter(normalized_events.values())).shape

    return BlockwiseDataBundle(
        train_dataset=BlockwiseFloodDataset(splits.train_df, normalized_events, normalized_blocks),
        val_dataset=BlockwiseFloodDataset(splits.val_df, normalized_events, normalized_blocks),
        test_dataset=BlockwiseFloodDataset(splits.test_df, normalized_events, normalized_blocks),
        feature_columns=feature_columns,
        event_shape=sample_event_shape,
        normalization=NormalizationStats(
            event_mean=event_mean,
            event_std=event_std,
            block_mean=block_mean,
            block_std=block_std,
        ),
        splits=splits,
    )
