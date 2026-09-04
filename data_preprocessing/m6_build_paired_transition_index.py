#!/usr/bin/env python3
"""Attach exact same-patch hydraulic transition labels to M4 candidates.

Unlike M5's event/time matching heuristic, this stage reads the previous
TRITON depth patch for every current candidate. Current patch statistics are
already present in M4, so only one additional netCDF patch read is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import netCDF4 as nc4
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_matrix_data import _compute_block_windows
from data_preprocessing.m4_build_stage1_sampling_index import resolve_path, values


LOGGER = logging.getLogger("m6_paired_transition_index")
REGIMES = ("stable", "filling", "draining", "rapid_filling", "rapid_draining")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--static-rasters-dir", type=Path, required=True)
    parser.add_argument("--input-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--event-ids", nargs="+", default=None)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--stable-storage-threshold", type=float, default=0.01)
    parser.add_argument("--stable-extent-threshold", type=float, default=0.01)
    parser.add_argument("--rapid-storage-threshold", type=float, default=0.05)
    parser.add_argument("--rapid-extent-threshold", type=float, default=0.05)
    parser.add_argument("--netcdf-chunk-cache-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def patch_statistics(patch: np.ndarray, mask: np.ndarray, wet_threshold: float):
    valid_depth = patch[mask]
    wet = valid_depth >= wet_threshold
    wet_count = int(wet.sum())
    wet_fraction = wet_count / max(len(valid_depth), 1)
    mean_wet_depth = float(valid_depth[wet].mean()) if wet_count else 0.0
    p90_wet_depth = float(np.quantile(valid_depth[wet], 0.90)) if wet_count else 0.0
    max_depth = float(valid_depth.max(initial=0.0))
    mean_cell_depth = float(valid_depth.sum(dtype=np.float64) / max(len(valid_depth), 1))
    return wet_fraction, mean_wet_depth, p90_wet_depth, max_depth, mean_cell_depth


def classify_regime(
    extent_delta: np.ndarray,
    storage_delta: np.ndarray,
    stable_extent_threshold: float,
    stable_storage_threshold: float,
    rapid_extent_threshold: float,
    rapid_storage_threshold: float,
):
    stable = (np.abs(extent_delta) < stable_extent_threshold) & (
        np.abs(storage_delta) < stable_storage_threshold
    )
    rapid = (np.abs(extent_delta) >= rapid_extent_threshold) | (
        np.abs(storage_delta) >= rapid_storage_threshold
    )
    # Mean cell depth is a storage-per-area proxy in metres. A five-centimetre
    # scale converts extent change to a comparable directional contribution.
    direction = storage_delta + 0.05 * extent_delta
    labels = np.full(len(extent_delta), "stable", dtype=object)
    labels[(~stable) & (~rapid) & (direction >= 0)] = "filling"
    labels[(~stable) & (~rapid) & (direction < 0)] = "draining"
    labels[rapid & (direction >= 0)] = "rapid_filling"
    labels[rapid & (direction < 0)] = "rapid_draining"
    activity = np.maximum(
        np.abs(storage_delta) / max(rapid_storage_threshold, 1e-12),
        np.abs(extent_delta) / max(rapid_extent_threshold, 1e-12),
    )
    return labels, direction, activity


def validate_args(args):
    thresholds = (
        args.wet_threshold,
        args.stable_storage_threshold,
        args.stable_extent_threshold,
        args.rapid_storage_threshold,
        args.rapid_extent_threshold,
    )
    if any(value <= 0 for value in thresholds):
        raise ValueError("Wet, stable, and rapid thresholds must be positive")
    if args.stable_storage_threshold >= args.rapid_storage_threshold:
        raise ValueError("Stable storage threshold must be below rapid threshold")
    if args.stable_extent_threshold >= args.rapid_extent_threshold:
        raise ValueError("Stable extent threshold must be below rapid threshold")


def main():
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir = args.output_dir.resolve()
    output_path = output_dir / "sampling_candidates.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = args.manifest_dir.resolve()
    manifest = pd.read_parquet(manifest_dir / "dynamic_manifest.parquet")
    metadata = json.loads((manifest_dir / "dynamic_metadata.json").read_text())
    candidates = pd.read_parquet(args.input_candidates.resolve())
    required = {
        "event_id",
        "time_index",
        "anchor_block",
        "wet_fraction",
        "mean_wet_depth",
        "wet_depth_p90",
        "max_depth",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Input candidate index is missing: {sorted(missing)}")
    candidates["event_id"] = candidates["event_id"].astype(str)
    if args.event_ids:
        requested = {str(value) for value in args.event_ids}
        candidates = candidates.loc[candidates["event_id"].isin(requested)].copy()
        manifest = manifest.loc[manifest["event_id"].astype(str).isin(requested)].copy()
    available = set(manifest["event_id"].astype(str))
    missing_events = set(candidates["event_id"]) - available
    if missing_events:
        raise ValueError(f"Candidates reference missing events: {sorted(missing_events)}")
    if candidates.empty:
        raise ValueError("No candidates remain after event filtering")

    labels_dir = args.labels_10m_dir.resolve()
    static_dir = args.static_rasters_dir.resolve()
    lookup = pd.read_parquet(labels_dir / "block_index_lookup.parquet").sort_values(
        "block_index"
    )
    block_grid = np.load(labels_dir / "block_index_10m.npy", mmap_mode="r")
    static_shape = np.load(static_dir / "block_static_features.npy", mmap_mode="r").shape
    windows, _ = _compute_block_windows(
        block_grid,
        lookup,
        target_rows=int(static_shape[2]),
        target_cols=int(static_shape[3]),
    )
    if candidates["anchor_block"].min() < 0 or candidates["anchor_block"].max() >= len(windows):
        raise ValueError("Candidate anchor block is outside the block lookup")

    result_frames = []
    cache_bytes = int(args.netcdf_chunk_cache_mb * 1024 * 1024)
    depth_name = metadata["variable_names"]["depth"]
    anchors: Sequence[Path] = (args.base_dir.resolve(), manifest_dir)
    manifest_lookup = manifest.set_index(manifest["event_id"].astype(str), drop=False)
    for event_id, event_candidates in candidates.groupby("event_id", sort=True):
        event = manifest_lookup.loc[event_id]
        netcdf_path = resolve_path(str(event["path_to_netcdf"]), anchors)
        frame = event_candidates.copy()
        frame["original_order"] = np.arange(len(frame), dtype=np.int64)
        with nc4.Dataset(netcdf_path, "r") as dataset:
            variable = dataset.variables[depth_name]
            nelems = max(1, cache_bytes // max(variable.dtype.itemsize, 1))
            variable.set_var_chunk_cache(cache_bytes, nelems, 0.75)
            chunks = variable.chunking()
            if isinstance(chunks, (list, tuple)) and len(chunks) == 3:
                frame["_previous_time_chunk"] = np.maximum(
                    frame["time_index"].to_numpy(dtype=np.int64) - 1, 0
                ) // int(chunks[0])
                frame["_row_chunk"] = frame["anchor_block"].map(
                    lambda value: windows[int(value)].row_start // int(chunks[1])
                )
                frame["_col_chunk"] = frame["anchor_block"].map(
                    lambda value: windows[int(value)].col_start // int(chunks[2])
                )
                frame = frame.sort_values(
                    ["_previous_time_chunk", "_row_chunk", "_col_chunk", "time_index"]
                )
            LOGGER.info("%s: reading %d previous patches", event_id, len(frame))
            previous_stats = {}
            for candidate in frame.itertuples(index=False):
                time_index = int(candidate.time_index)
                block_index = int(candidate.anchor_block)
                if time_index == 0:
                    stats = (
                        float(candidate.wet_fraction),
                        float(candidate.mean_wet_depth),
                        float(candidate.wet_depth_p90),
                        float(candidate.max_depth),
                        float(candidate.wet_fraction) * float(candidate.mean_wet_depth),
                    )
                else:
                    window = windows[block_index]
                    patch = values(
                        variable[
                            time_index - 1,
                            window.row_start : window.row_stop,
                            window.col_start : window.col_stop,
                        ]
                    )
                    mask = (
                        block_grid[
                            window.row_start : window.row_stop,
                            window.col_start : window.col_stop,
                        ]
                        == block_index
                    )
                    stats = patch_statistics(patch, mask, args.wet_threshold)
                previous_stats[int(candidate.original_order)] = stats

        frame = frame.sort_values("original_order").reset_index(drop=True)
        ordered = np.asarray(
            [previous_stats[index] for index in frame["original_order"]],
            dtype=np.float64,
        )
        frame["previous_wet_fraction"] = ordered[:, 0].astype(np.float32)
        frame["previous_mean_wet_depth"] = ordered[:, 1].astype(np.float32)
        frame["previous_wet_depth_p90"] = ordered[:, 2].astype(np.float32)
        frame["previous_max_depth"] = ordered[:, 3].astype(np.float32)
        frame["previous_mean_cell_depth"] = ordered[:, 4].astype(np.float32)
        frame["mean_cell_depth"] = (
            frame["wet_fraction"].astype(np.float64)
            * frame["mean_wet_depth"].astype(np.float64)
        ).astype(np.float32)
        frame["local_extent_delta"] = (
            frame["wet_fraction"] - frame["previous_wet_fraction"]
        ).astype(np.float32)
        frame["local_storage_delta"] = (
            frame["mean_cell_depth"] - frame["previous_mean_cell_depth"]
        ).astype(np.float32)
        frame["local_p90_depth_delta"] = (
            frame["wet_depth_p90"] - frame["previous_wet_depth_p90"]
        ).astype(np.float32)
        labels, direction, activity = classify_regime(
            frame["local_extent_delta"].to_numpy(dtype=np.float64),
            frame["local_storage_delta"].to_numpy(dtype=np.float64),
            args.stable_extent_threshold,
            args.stable_storage_threshold,
            args.rapid_extent_threshold,
            args.rapid_storage_threshold,
        )
        frame["local_transition_regime"] = labels
        frame["local_transition_direction"] = direction.astype(np.float32)
        frame["local_transition_activity"] = activity.astype(np.float32)
        temporary = [
            name
            for name in frame.columns
            if name.startswith("_") or name == "original_order"
        ]
        result_frames.append(frame.drop(columns=temporary))

    result = pd.concat(result_frames, ignore_index=True)
    result.to_parquet(output_path, index=False)
    summary = (
        result.groupby(["event_id", "local_transition_regime"], observed=True)
        .agg(
            candidates=("local_transition_regime", "size"),
            mean_activity=("local_transition_activity", "mean"),
            mean_extent_delta=("local_extent_delta", "mean"),
            mean_storage_delta=("local_storage_delta", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "transition_sampling_summary.csv", index=False)
    payload = {
        "format_version": 1,
        "sampling_strategy": "exact_same_patch_transition_candidates",
        "source_candidates": str(args.input_candidates.resolve()),
        "n_candidates": int(len(result)),
        "n_events": int(result["event_id"].nunique()),
        "event_ids": sorted(result["event_id"].unique().tolist()),
        "wet_threshold": args.wet_threshold,
        "stable_storage_threshold": args.stable_storage_threshold,
        "stable_extent_threshold": args.stable_extent_threshold,
        "rapid_storage_threshold": args.rapid_storage_threshold,
        "rapid_extent_threshold": args.rapid_extent_threshold,
        "regime_counts": {
            str(key): int(value)
            for key, value in result["local_transition_regime"].value_counts().items()
        },
    }
    (output_dir / "sampling_metadata.json").write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote %d exact paired candidates to %s", len(result), output_dir)


if __name__ == "__main__":
    main()
