from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_EVENT_COLUMNS = {"event_id", "watershed_id", "path_to_X_event", "T", "F"}
REQUIRED_BLOCK_COLUMNS = {"watershed_id", "block_id"}


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
