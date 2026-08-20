#!/usr/bin/env python3
"""Build a compact, label-aware candidate pool for Stage-1 training.

Each row describes one actual TRITON depth patch.  The patch is used as the
anchor of a spatially local training batch and is categorized as dry,
boundary, wet, or deep.  This gives the training sampler direct information
about dynamic labels without exhaustively scanning every event/time/block.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Sequence

import netCDF4 as nc4
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_matrix_data import _compute_block_windows


LOGGER = logging.getLogger("stage1_sampling_index")
CATEGORIES = ("dry", "boundary", "wet", "deep")
PHASES = ("quiet", "rising", "peak", "recession")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a label-aware Stage-1 sampling candidate index"
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--static-rasters-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--event-ids", nargs="+", default=None)
    parser.add_argument("--candidates-per-event", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-max-fraction", type=float, default=0.10)
    parser.add_argument("--deep-threshold", type=float, default=1.0)
    parser.add_argument("--deep-min-wet-fraction", type=float, default=0.10)
    parser.add_argument(
        "--deep-depth-statistic",
        choices=["mean", "p90", "max"],
        default="p90",
    )
    parser.add_argument("--flow-weight-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--netcdf-chunk-cache-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_path(raw: str, anchors: Sequence[Path]) -> Path:
    candidate = Path(raw)
    for path in [candidate, *(anchor / candidate for anchor in anchors)]:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve path: {raw}")


def values(array) -> np.ndarray:
    if np.ma.isMaskedArray(array):
        array = array.filled(0.0)
    result = np.asarray(array, dtype=np.float32)
    result[~np.isfinite(result)] = 0.0
    return result


def forcing_phases(forcing: np.ndarray) -> np.ndarray:
    intensity = np.maximum(forcing, 0.0).sum(axis=1, dtype=np.float64)
    kernel = np.ones(49, dtype=np.float64) / 49.0
    intensity = np.convolve(intensity, kernel, mode="same")
    phases = np.full(len(intensity), "quiet", dtype="U9")
    peak_index = int(np.argmax(intensity))
    peak_value = float(intensity[peak_index])
    if peak_value <= 0:
        return phases
    active = intensity >= 0.05 * peak_value
    peak = intensity >= 0.80 * peak_value
    indices = np.arange(len(intensity))
    phases[active & (indices <= peak_index)] = "rising"
    phases[active & (indices > peak_index)] = "recession"
    phases[peak] = "peak"
    return phases


def block_probabilities(static_dir: Path, n_blocks: int, flow_fraction: float):
    names = json.loads(
        (static_dir / "block_static_feature_names.json").read_text()
    )["feature_names"]
    if "flow_acc" not in names or flow_fraction <= 0:
        return np.full(n_blocks, 1.0 / n_blocks, dtype=np.float64)
    static = np.load(static_dir / "block_static_features.npy", mmap_mode="r")
    activity = np.log1p(
        np.maximum(static[:, names.index("flow_acc")], 0.0)
        .max(axis=(1, 2))
        .astype(np.float64)
    )
    if activity.sum() <= 0:
        return np.full(n_blocks, 1.0 / n_blocks, dtype=np.float64)
    return (1.0 - flow_fraction) / n_blocks + flow_fraction * activity / activity.sum()


def choose_candidates(
    rng: np.random.Generator,
    phases: np.ndarray,
    block_prob: np.ndarray,
    count: int,
):
    # Deliberately balance event phases. Missing phases are skipped and the
    # remaining phase probabilities are renormalized.
    phase_weights = {"quiet": 0.15, "rising": 0.30, "peak": 0.30, "recession": 0.25}
    available = [phase for phase in PHASES if np.any(phases == phase)]
    probabilities = np.asarray([phase_weights[phase] for phase in available], dtype=np.float64)
    probabilities /= probabilities.sum()
    selected_phases = rng.choice(available, size=count, p=probabilities)
    times = np.empty(count, dtype=np.int32)
    for phase in available:
        positions = np.flatnonzero(selected_phases == phase)
        valid_times = np.flatnonzero(phases == phase)
        times[positions] = rng.choice(valid_times, size=len(positions), replace=True)
    blocks = rng.choice(len(block_prob), size=count, replace=True, p=block_prob).astype(np.int32)
    return times, blocks, selected_phases


def classify(
    wet_fraction: float,
    depth_statistic: float,
    boundary_max: float,
    deep_threshold: float,
    deep_min_wet_fraction: float,
):
    if wet_fraction <= 0:
        return "dry"
    if wet_fraction < boundary_max:
        return "boundary"
    if wet_fraction >= deep_min_wet_fraction and depth_statistic >= deep_threshold:
        return "deep"
    return "wet"


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not 0 <= args.flow_weight_fraction <= 1:
        raise ValueError("--flow-weight-fraction must be between 0 and 1")
    if not 0 <= args.boundary_max_fraction <= 1:
        raise ValueError("--boundary-max-fraction must be between 0 and 1")
    if not 0 <= args.deep_min_wet_fraction <= 1:
        raise ValueError("--deep-min-wet-fraction must be between 0 and 1")
    if args.deep_min_wet_fraction < args.boundary_max_fraction:
        raise ValueError("--deep-min-wet-fraction cannot be below the boundary threshold")
    if args.candidates_per_event <= 0 or args.batch_size <= 0:
        raise ValueError("candidate and batch counts must be positive")

    manifest_dir = args.manifest_dir.resolve()
    labels_dir = args.labels_10m_dir.resolve()
    static_dir = args.static_rasters_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_path = output_dir / "sampling_candidates.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_dir / "dynamic_manifest.parquet").sort_values("event_id")
    metadata = json.loads((manifest_dir / "dynamic_metadata.json").read_text())
    if args.event_ids:
        requested = set(str(value) for value in args.event_ids)
        available = set(manifest["event_id"].astype(str))
        missing = requested - available
        if missing:
            raise ValueError(f"Requested events are unavailable: {sorted(missing)}")
        manifest = manifest.loc[manifest["event_id"].isin(requested)]
    if args.max_events is not None:
        manifest = manifest.head(args.max_events)
    lookup = pd.read_parquet(labels_dir / "block_index_lookup.parquet").sort_values("block_index")
    block_grid = np.load(labels_dir / "block_index_10m.npy", mmap_mode="r")
    static_shape = np.load(static_dir / "block_static_features.npy", mmap_mode="r").shape
    windows, _ = _compute_block_windows(
        block_grid,
        lookup,
        target_rows=int(static_shape[2]),
        target_cols=int(static_shape[3]),
    )
    n_blocks = len(lookup)
    block_prob = block_probabilities(static_dir, n_blocks, args.flow_weight_fraction)
    rows = []
    anchors = [args.base_dir.resolve(), manifest_dir]
    depth_name = metadata["variable_names"]["depth"]
    cache_bytes = int(args.netcdf_chunk_cache_mb * 1024 * 1024)

    for event in manifest.itertuples(index=False):
        event_seed = args.seed + sum(
            (position + 1) * ord(character)
            for position, character in enumerate(str(event.event_id))
        )
        rng = np.random.default_rng(event_seed)
        forcing_path = resolve_path(str(event.path_to_X_event), anchors)
        netcdf_path = resolve_path(str(event.path_to_netcdf), anchors)
        forcing = np.load(forcing_path, mmap_mode="r")
        phases = forcing_phases(np.asarray(forcing))
        times, blocks, selected_phases = choose_candidates(
            rng, phases, block_prob, args.candidates_per_event
        )
        candidates = pd.DataFrame(
            {"time_index": times, "anchor_block": blocks, "phase": selected_phases}
        ).drop_duplicates(["time_index", "anchor_block"])
        with nc4.Dataset(netcdf_path, "r") as ds:
            variable = ds.variables[depth_name]
            nelems = max(1, cache_bytes // variable.dtype.itemsize)
            variable.set_var_chunk_cache(cache_bytes, nelems, 0.75)
            chunks = variable.chunking()
            if isinstance(chunks, (list, tuple)) and len(chunks) == 3:
                candidates["_time_chunk"] = candidates["time_index"] // int(chunks[0])
                candidates["_row_chunk"] = candidates["anchor_block"].map(
                    lambda value: windows[int(value)].row_start // int(chunks[1])
                )
                candidates["_col_chunk"] = candidates["anchor_block"].map(
                    lambda value: windows[int(value)].col_start // int(chunks[2])
                )
                candidates = candidates.sort_values(
                    ["_time_chunk", "_row_chunk", "_col_chunk", "time_index"]
                )
            else:
                candidates = candidates.sort_values(["time_index", "anchor_block"])
            candidates = candidates.reset_index(drop=True)
            LOGGER.info(
                "%s: evaluating %d unique anchor patches", event.event_id, len(candidates)
            )
            for candidate in candidates.itertuples(index=False):
                block_index = int(candidate.anchor_block)
                window = windows[block_index]
                patch = values(
                    variable[
                        int(candidate.time_index),
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
                valid_depth = patch[mask]
                wet = valid_depth >= args.wet_threshold
                wet_count = int(wet.sum())
                wet_fraction = wet_count / max(len(valid_depth), 1)
                max_depth = float(valid_depth.max(initial=0.0))
                mean_wet_depth = float(valid_depth[wet].mean()) if wet_count else 0.0
                wet_depth_p90 = (
                    float(np.quantile(valid_depth[wet], 0.90)) if wet_count else 0.0
                )
                depth_statistics = {
                    "mean": mean_wet_depth,
                    "p90": wet_depth_p90,
                    "max": max_depth,
                }
                start = min(
                    max(0, block_index - args.batch_size // 2),
                    max(0, n_blocks - args.batch_size),
                )
                rows.append(
                    {
                        "event_id": str(event.event_id),
                        "time_index": int(candidate.time_index),
                        "anchor_block": block_index,
                        "block_start": int(start),
                        "phase": str(candidate.phase),
                        "category": classify(
                            wet_fraction,
                            depth_statistics[args.deep_depth_statistic],
                            args.boundary_max_fraction,
                            args.deep_threshold,
                            args.deep_min_wet_fraction,
                        ),
                        "wet_fraction": np.float32(wet_fraction),
                        "max_depth": np.float32(max_depth),
                        "mean_wet_depth": np.float32(mean_wet_depth),
                        "wet_depth_p90": np.float32(wet_depth_p90),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)
    summary = (
        frame.groupby(["event_id", "phase", "category"], observed=True)
        .agg(
            candidates=("category", "size"),
            mean_wet_fraction=("wet_fraction", "mean"),
            mean_max_depth=("max_depth", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "sampling_summary.csv", index=False)
    payload: Dict[str, object] = {
        "format_version": 1,
        "sampling_strategy": "label_aware_anchor_candidate_pool",
        "n_candidates": int(len(frame)),
        "n_events": int(frame["event_id"].nunique()),
        "event_ids": sorted(frame["event_id"].unique().tolist()),
        "n_blocks": int(n_blocks),
        "batch_size": int(args.batch_size),
        "wet_threshold": float(args.wet_threshold),
        "boundary_max_fraction": float(args.boundary_max_fraction),
        "deep_threshold": float(args.deep_threshold),
        "deep_min_wet_fraction": float(args.deep_min_wet_fraction),
        "deep_depth_statistic": str(args.deep_depth_statistic),
        "flow_weight_fraction": float(args.flow_weight_fraction),
        "candidates_per_event_requested": int(args.candidates_per_event),
        "category_counts": {
            str(key): int(value) for key, value in frame["category"].value_counts().items()
        },
        "phase_counts": {
            str(key): int(value) for key, value in frame["phase"].value_counts().items()
        },
    }
    (output_dir / "sampling_metadata.json").write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote %d candidates to %s", len(frame), output_dir)


if __name__ == "__main__":
    main()
