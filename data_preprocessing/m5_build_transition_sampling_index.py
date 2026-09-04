#!/usr/bin/env python3
"""Add consecutive-state transition regimes to the Stage-1 candidate index.

The dense M4 index samples many blocks independently at every event/time. This
script matches the subset of blocks present at both ``t-1`` and ``t`` and uses
their exact label-statistic changes to classify the whole event/time group.
When no block matches, it falls back to changes in group aggregate statistics.
No raw netCDF scan is required.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "event_id",
    "time_index",
    "anchor_block",
    "wet_fraction",
    "max_depth",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stable-wet-delta", type=float, default=0.01)
    parser.add_argument("--stable-depth-delta", type=float, default=0.05)
    parser.add_argument("--rapid-wet-delta", type=float, default=0.05)
    parser.add_argument("--rapid-depth-delta", type=float, default=0.10)
    parser.add_argument("--depth-direction-scale", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def classify_transition_groups(
    wet_delta,
    depth_delta,
    stable_wet_delta=0.01,
    stable_depth_delta=0.05,
    rapid_wet_delta=0.05,
    rapid_depth_delta=0.10,
    depth_direction_scale=2.0,
):
    wet_delta = np.asarray(wet_delta, dtype=np.float64)
    depth_delta = np.asarray(depth_delta, dtype=np.float64)
    stable = (
        (np.abs(wet_delta) <= stable_wet_delta)
        & (np.abs(depth_delta) <= stable_depth_delta)
    )
    rapid = (
        (np.abs(wet_delta) >= rapid_wet_delta)
        | (np.abs(depth_delta) >= rapid_depth_delta)
    ) & ~stable
    direction = wet_delta + depth_delta / float(depth_direction_scale)
    result = np.full(wet_delta.shape, "stable", dtype=object)
    result[(~stable) & (~rapid) & (direction >= 0)] = "filling"
    result[(~stable) & (~rapid) & (direction < 0)] = "draining"
    result[rapid & (direction >= 0)] = "rapid_filling"
    result[rapid & (direction < 0)] = "rapid_draining"
    return result, direction, rapid


def build_transition_groups(frame):
    keys = ["event_id", "time_index", "anchor_block"]
    previous = frame[keys + ["wet_fraction", "max_depth"]].rename(
        columns={"wet_fraction": "previous_wet_fraction", "max_depth": "previous_max_depth"}
    )
    previous = previous.copy()
    previous["time_index"] += 1
    matched = frame[keys + ["wet_fraction", "max_depth"]].merge(
        previous, on=keys, how="inner", validate="one_to_one"
    )
    matched["wet_delta"] = (
        matched["wet_fraction"] - matched["previous_wet_fraction"]
    )
    matched["depth_delta"] = matched["max_depth"] - matched["previous_max_depth"]
    exact = (
        matched.groupby(["event_id", "time_index"], observed=True)
        .agg(
            matched_block_count=("anchor_block", "size"),
            exact_wet_delta=("wet_delta", "median"),
            exact_depth_delta=("depth_delta", "median"),
        )
        .reset_index()
    )

    aggregate = (
        frame.groupby(["event_id", "time_index"], observed=True)
        .agg(
            candidate_count=("anchor_block", "size"),
            group_wet_fraction=("wet_fraction", "mean"),
            group_max_depth=("max_depth", "median"),
        )
        .reset_index()
        .sort_values(["event_id", "time_index"])
    )
    aggregate["aggregate_wet_delta"] = aggregate.groupby("event_id")[
        "group_wet_fraction"
    ].diff()
    aggregate["aggregate_depth_delta"] = aggregate.groupby("event_id")[
        "group_max_depth"
    ].diff()
    groups = aggregate.merge(exact, on=["event_id", "time_index"], how="left")
    has_match = groups["matched_block_count"].fillna(0) > 0
    groups["transition_statistic_source"] = np.where(
        has_match, "matched_blocks", "group_aggregate"
    )
    groups["transition_wet_delta"] = groups["exact_wet_delta"].where(
        has_match, groups["aggregate_wet_delta"]
    )
    groups["transition_depth_delta"] = groups["exact_depth_delta"].where(
        has_match, groups["aggregate_depth_delta"]
    )
    groups[["transition_wet_delta", "transition_depth_delta"]] = groups[
        ["transition_wet_delta", "transition_depth_delta"]
    ].fillna(0.0)
    groups["matched_block_count"] = groups["matched_block_count"].fillna(0).astype(np.int32)
    return groups


def main():
    args = parse_args()
    if args.depth_direction_scale <= 0:
        raise ValueError("--depth-direction-scale must be positive")
    for name in (
        "stable_wet_delta",
        "stable_depth_delta",
        "rapid_wet_delta",
        "rapid_depth_delta",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_path = output_dir / "sampling_candidates.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    frame = pd.read_parquet(input_dir / "sampling_candidates.parquet")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input index is missing columns: {sorted(missing)}")
    if frame.duplicated(["event_id", "time_index", "anchor_block"]).any():
        raise ValueError("Input candidate keys must be unique")

    groups = build_transition_groups(frame)
    regime, direction, rapid = classify_transition_groups(
        groups["transition_wet_delta"],
        groups["transition_depth_delta"],
        args.stable_wet_delta,
        args.stable_depth_delta,
        args.rapid_wet_delta,
        args.rapid_depth_delta,
        args.depth_direction_scale,
    )
    groups["transition_regime"] = regime
    groups["transition_direction_score"] = direction.astype(np.float32)
    groups["rapid_transition"] = rapid
    groups["transition_activity"] = (
        groups["transition_wet_delta"].abs()
        + groups["transition_depth_delta"].abs() / args.depth_direction_scale
    ).astype(np.float32)

    keep = [
        "event_id",
        "time_index",
        "transition_regime",
        "rapid_transition",
        "transition_activity",
        "transition_direction_score",
        "transition_wet_delta",
        "transition_depth_delta",
        "transition_statistic_source",
        "matched_block_count",
        "candidate_count",
    ]
    enriched = frame.merge(
        groups[keep], on=["event_id", "time_index"], how="left", validate="many_to_one"
    )
    if enriched["transition_regime"].isna().any():
        raise RuntimeError("Some candidate rows did not receive a transition regime")

    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    groups.to_parquet(output_dir / "transition_groups.parquet", index=False)
    summary = (
        groups.groupby(["transition_regime", "transition_statistic_source"], observed=True)
        .agg(
            event_times=("time_index", "size"),
            mean_activity=("transition_activity", "mean"),
            median_matched_blocks=("matched_block_count", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "transition_sampling_summary.csv", index=False)
    source_metadata = json.loads((input_dir / "sampling_metadata.json").read_text())
    metadata = dict(source_metadata)
    metadata.update(
        {
            "format_version": 2,
            "sampling_strategy": "label_and_transition_aware_candidate_pool",
            "source_sampling_index": str(input_dir),
            "transition_thresholds": {
                "stable_wet_delta": args.stable_wet_delta,
                "stable_depth_delta": args.stable_depth_delta,
                "rapid_wet_delta": args.rapid_wet_delta,
                "rapid_depth_delta": args.rapid_depth_delta,
                "depth_direction_scale": args.depth_direction_scale,
            },
            "transition_regime_counts": {
                str(key): int(value)
                for key, value in groups["transition_regime"].value_counts().items()
            },
            "matched_event_time_fraction": float(
                (groups["matched_block_count"] > 0).mean()
            ),
        }
    )
    (output_dir / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(enriched):,} candidates and {len(groups):,} event/time groups")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
