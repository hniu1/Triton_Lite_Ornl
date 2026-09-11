#!/usr/bin/env python3
"""Measure the actual label distribution produced by a Stage-1 sampler."""

import argparse
import json
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from stage1_data import BalancedLabelBatchSampler, prepare_stage1_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--static-rasters-dir", type=Path, required=True)
    parser.add_argument("--sampling-index-dir", type=Path, default=None)
    parser.add_argument(
        "--sampling-mode", choices=["anchor", "balanced_batch"], default="anchor"
    )
    parser.add_argument("--sampling-target-wet-cell-fraction", type=float, default=0.0)
    parser.add_argument("--sampling-strict-category-quotas", action="store_true")
    parser.add_argument("--sample-dry-fraction", type=float, default=0.15)
    parser.add_argument("--sample-boundary-fraction", type=float, default=0.25)
    parser.add_argument("--sample-wet-fraction", type=float, default=0.40)
    parser.add_argument("--sample-deep-fraction", type=float, default=0.20)
    parser.add_argument("--sample-quiet-fraction", type=float, default=0.15)
    parser.add_argument("--sample-rising-fraction", type=float, default=0.30)
    parser.add_argument("--sample-peak-fraction", type=float, default=0.30)
    parser.add_argument("--sample-recession-fraction", type=float, default=0.25)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--test-events", nargs="+", default=["D030"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--deep-threshold", type=float, default=1.0)
    parser.add_argument("--boundary-max-fraction", type=float, default=0.10)
    parser.add_argument("--deep-min-wet-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--netcdf-chunk-cache-mb", type=int, default=256)
    parser.add_argument("--max-open-netcdf-handles", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = prepare_stage1_data(
        manifest_dir=args.manifest_dir,
        events_csv=args.events_csv,
        blocks_parquet=args.blocks_parquet,
        labels_10m_dir=args.labels_10m_dir,
        static_rasters_dir=args.static_rasters_dir,
        base_dir=Path("."),
        test_events=args.test_events,
        val_fraction=0.2,
        seed=args.seed,
        batch_size=args.batch_size,
        train_batches_per_epoch=args.batches,
        eval_batches=1,
        train_time_stride=1,
        eval_time_stride=12,
        wet_threshold=args.wet_threshold,
        netcdf_chunk_cache_mb=args.netcdf_chunk_cache_mb,
        max_open_netcdf_handles=args.max_open_netcdf_handles,
        sampling_index_dir=args.sampling_index_dir,
        sampling_mode=args.sampling_mode,
        sampling_target_wet_cell_fraction=args.sampling_target_wet_cell_fraction,
        sampling_strict_category_quotas=args.sampling_strict_category_quotas,
        sampling_category_fractions={
            "dry": args.sample_dry_fraction,
            "boundary": args.sample_boundary_fraction,
            "wet": args.sample_wet_fraction,
            "deep": args.sample_deep_fraction,
        },
        sampling_phase_fractions={
            "quiet": args.sample_quiet_fraction,
            "rising": args.sample_rising_fraction,
            "peak": args.sample_peak_fraction,
            "recession": args.sample_recession_fraction,
        },
    )
    loader = DataLoader(bundle.train_dataset, batch_sampler=bundle.train_sampler, num_workers=0)
    counts = Counter()
    event_counts = Counter()
    phase_counts = Counter()
    wet_cells = 0
    valid_cells = 0
    for batch in loader:
        mask = batch["mask"] > 0.5
        wet = (batch["depth"] >= args.wet_threshold) & mask
        per_patch_valid = mask.sum(dim=(1, 2)).clamp_min(1)
        per_patch_wet = wet.sum(dim=(1, 2))
        fractions = per_patch_wet / per_patch_valid
        p90_depths = []
        for patch_depth, patch_wet in zip(batch["depth"], wet):
            wet_depth = patch_depth[patch_wet]
            p90_depths.append(float(wet_depth.quantile(0.90)) if wet_depth.numel() else 0.0)
        for fraction, p90_depth in zip(fractions.tolist(), p90_depths):
            if fraction == 0:
                counts["dry"] += 1
            elif fraction < args.boundary_max_fraction:
                counts["boundary"] += 1
            elif fraction >= args.deep_min_wet_fraction and p90_depth >= args.deep_threshold:
                counts["deep"] += 1
            else:
                counts["wet"] += 1
        event_counts.update(str(value) for value in batch["event_id"])
        if isinstance(bundle.train_sampler, BalancedLabelBatchSampler):
            event_id = str(batch["event_id"][0])
            event_position = bundle.train_sampler.event_ids.index(event_id)
            time_index = int(batch["time_index"][0])
            phase_counts[bundle.train_sampler.groups[(event_position, time_index)]["phase"]] += 1
        wet_cells += int(wet.sum())
        valid_cells += int(mask.sum())

    n_patches = sum(counts.values())
    payload = {
        "sampler": (
            args.sampling_mode if args.sampling_index_dir else "forcing_flow_proxy"
        ),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "n_patches": int(n_patches),
        "wet_threshold": float(args.wet_threshold),
        "deep_threshold": float(args.deep_threshold),
        "boundary_max_fraction": float(args.boundary_max_fraction),
        "deep_min_wet_fraction": float(args.deep_min_wet_fraction),
        "deep_depth_statistic": "wet_p90",
        "wet_cell_fraction": wet_cells / max(valid_cells, 1),
        "patch_counts": dict(sorted(counts.items())),
        "patch_fractions": {
            key: value / max(n_patches, 1) for key, value in sorted(counts.items())
        },
        "event_patch_counts": dict(sorted(event_counts.items())),
        "phase_batch_counts": dict(sorted(phase_counts.items())),
        "phase_batch_fractions": {
            key: value / max(args.batches, 1) for key, value in sorted(phase_counts.items())
        },
        "strict_category_quotas": bool(args.sampling_strict_category_quotas),
        "split_events": bundle.split_events,
    }
    args.output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_path.resolve().write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
