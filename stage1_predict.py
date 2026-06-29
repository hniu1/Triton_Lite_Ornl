#!/usr/bin/env python3
"""Predict selected Stage-1 event/timestamp/block combinations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage1_data import prepare_stage1_data
from stage1_model import Stage1TimestampModel
from stage1_train import move_batch, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 timestamp inference")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--time-indices", type=int, nargs="+", required=True)
    parser.add_argument("--block-indices", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    device = resolve_device(args.device)
    bundle = prepare_stage1_data(
        manifest_dir=Path(config["manifest_dir"]),
        events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]),
        labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]),
        base_dir=Path(config["base_dir"]),
        test_events=[args.event_id],
        val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]),
        batch_size=max(1, len(args.block_indices)),
        train_batches_per_epoch=1,
        eval_batches=1,
        train_time_stride=1,
        eval_time_stride=1,
        wet_threshold=float(config["wet_threshold"]),
        feature_columns=config["block_feature_columns"],
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
    )
    dataset = bundle.test_dataset
    event_positions = dataset.events.index[
        dataset.events["event_id"] == args.event_id
    ].tolist()
    if len(event_positions) != 1:
        raise ValueError(f"Could not uniquely select test event {args.event_id}")
    event_position = event_positions[0]
    invalid_blocks = [
        value for value in args.block_indices if value < 0 or value >= len(dataset.block_rows)
    ]
    if invalid_blocks:
        raise ValueError(f"Invalid zero-based block positions: {invalid_blocks}")
    max_time = int(dataset.events.iloc[event_position]["n_times"])
    invalid_times = [value for value in args.time_indices if value < 0 or value >= max_time]
    if invalid_times:
        raise ValueError(f"Invalid time indices: {invalid_times}")

    keys = [
        (event_position, time_index, block_position)
        for time_index in args.time_indices
        for block_position in args.block_indices
    ]
    loader = DataLoader(dataset, batch_size=len(args.block_indices), sampler=keys, num_workers=args.num_workers)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model = Stage1TimestampModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            depth, wet_logits, cx, cy = model(
                batch["event"],
                batch["time_index"],
                batch["time_features"],
                batch["block_features"],
                batch["static"],
                batch["mask"],
                shared_event_time=True,
            )
            count = depth.shape[0]
            for local in range(count):
                key = keys[offset + local]
                stem = f"{args.event_id}_t{key[1]:03d}_b{key[2]:06d}"
                np.savez_compressed(
                    output_dir / f"{stem}.npz",
                    depth=depth[local].cpu().numpy(),
                    wet_probability=torch.sigmoid(wet_logits[local]).cpu().numpy(),
                    component_x=cx[local].cpu().numpy(),
                    component_y=cy[local].cpu().numpy(),
                    mask=batch["mask"][local].cpu().numpy(),
                )
                records.append(
                    {
                        "event_id": args.event_id,
                        "time_index": key[1],
                        "block_position": key[2],
                        "path": str(output_dir / f"{stem}.npz"),
                    }
                )
            offset += count
    (output_dir / "prediction_manifest.json").write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
