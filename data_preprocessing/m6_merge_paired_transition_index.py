#!/usr/bin/env python3
"""Merge and validate per-event exact paired-transition index shards."""

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
            f"Expected {args.expected_events} paired-transition shards, "
            f"found {len(shard_paths)}"
        )
    metadata = [
        json.loads((path.parent / "sampling_metadata.json").read_text())
        for path in shard_paths
    ]
    invariant_keys = (
        "format_version",
        "sampling_strategy",
        "wet_threshold",
        "stable_storage_threshold",
        "stable_extent_threshold",
        "rapid_storage_threshold",
        "rapid_extent_threshold",
    )
    for key in invariant_keys:
        values = {json.dumps(item[key], sort_keys=True) for item in metadata}
        if len(values) != 1:
            raise ValueError(f"Shard metadata disagree on {key}: {sorted(values)}")

    frames = [pd.read_parquet(path) for path in shard_paths]
    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["event_id", "time_index", "anchor_block"]
    )
    keys = ["event_id", "time_index", "anchor_block"]
    if frame.duplicated(keys).any():
        raise ValueError("Paired-transition shards contain duplicate candidates")
    if frame["event_id"].nunique() != args.expected_events:
        raise ValueError("Merged table does not contain the expected event count")
    if frame["local_transition_regime"].isna().any():
        raise ValueError("Merged table contains missing local transition regimes")

    output_dir = args.output_dir.resolve()
    output_path = output_dir / "sampling_candidates.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary = (
        frame.groupby(["event_id", "phase", "local_transition_regime"], observed=True)
        .agg(
            candidates=("local_transition_regime", "size"),
            mean_activity=("local_transition_activity", "mean"),
            mean_extent_delta=("local_extent_delta", "mean"),
            mean_storage_delta=("local_storage_delta", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "transition_sampling_summary.csv", index=False)
    payload = {key: metadata[0][key] for key in invariant_keys}
    payload.update(
        {
            "n_candidates": int(len(frame)),
            "n_events": int(frame["event_id"].nunique()),
            "event_ids": sorted(frame["event_id"].unique().tolist()),
            "regime_counts": {
                str(key): int(value)
                for key, value in frame["local_transition_regime"].value_counts().items()
            },
            "category_counts": {
                str(key): int(value) for key, value in frame["category"].value_counts().items()
            },
            "phase_counts": {
                str(key): int(value) for key, value in frame["phase"].value_counts().items()
            },
            "source_shards": [str(path) for path in shard_paths],
        }
    )
    (output_dir / "sampling_metadata.json").write_text(json.dumps(payload, indent=2))
    print(
        f"Merged {len(shard_paths)} exact paired-transition shards and "
        f"{len(frame)} candidates into {output_dir}"
    )


if __name__ == "__main__":
    main()
