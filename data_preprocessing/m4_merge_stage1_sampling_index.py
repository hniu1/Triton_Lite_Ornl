#!/usr/bin/env python3
"""Merge per-event Stage-1 sampling-index shards."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shard_paths = sorted(args.shards_dir.resolve().glob("*/sampling_candidates.parquet"))
    if len(shard_paths) != args.expected_events:
        raise ValueError(
            f"Expected {args.expected_events} sampling shards, found {len(shard_paths)}"
        )
    metadata = [
        json.loads((path.parent / "sampling_metadata.json").read_text())
        for path in shard_paths
    ]
    invariant_keys = (
        "format_version",
        "n_blocks",
        "batch_size",
        "wet_threshold",
        "boundary_max_fraction",
        "deep_threshold",
        "flow_weight_fraction",
    )
    optional_invariant_keys = (
        "deep_min_wet_fraction",
        "deep_depth_statistic",
    )
    invariant_keys = invariant_keys + tuple(
        key for key in optional_invariant_keys if all(key in item for item in metadata)
    )
    for key in invariant_keys:
        values = {json.dumps(item[key], sort_keys=True) for item in metadata}
        if len(values) != 1:
            raise ValueError(f"Shard metadata disagree on {key}: {sorted(values)}")

    frames = [pd.read_parquet(path) for path in shard_paths]
    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["event_id", "time_index", "anchor_block"]
    )
    if frame.duplicated(["event_id", "time_index", "anchor_block"]).any():
        raise ValueError("Sampling shards contain duplicate event/time/block candidates")
    if frame["event_id"].nunique() != args.expected_events:
        raise ValueError("Merged candidate table does not contain the expected event count")

    output_dir = args.output_dir.resolve()
    output_path = output_dir / "sampling_candidates.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
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
    payload = {
        key: metadata[0][key] for key in invariant_keys
    }
    payload.update(
        {
            "sampling_strategy": "label_aware_anchor_candidate_pool",
            "n_candidates": int(len(frame)),
            "n_events": int(frame["event_id"].nunique()),
            "event_ids": sorted(frame["event_id"].unique().tolist()),
            "category_counts": {
                str(key): int(value)
                for key, value in frame["category"].value_counts().items()
            },
            "phase_counts": {
                str(key): int(value)
                for key, value in frame["phase"].value_counts().items()
            },
            "source_shards": [str(path) for path in shard_paths],
        }
    )
    (output_dir / "sampling_metadata.json").write_text(json.dumps(payload, indent=2))
    print(f"Merged {len(shard_paths)} shards and {len(frame)} candidates into {output_dir}")


if __name__ == "__main__":
    main()
