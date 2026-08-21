#!/usr/bin/env python3
"""Evaluate Stage-1 depth and velocity errors by true-depth regime."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage1_data import prepare_stage1_data
from stage1_model import Stage1TimestampModel
from stage1_train import make_loader, move_batch, resolve_device


BINS = [
    ("dry", 0.0, 0.05),
    ("shallow", 0.05, 0.25),
    ("moderate", 0.25, 1.0),
    ("deep", 1.0, 2.0),
    ("extreme", 2.0, float("inf")),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batches", type=int, default=1000)
    parser.add_argument("--eval-time-stride", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--direction-speed-threshold", type=float, default=0.05)
    return parser.parse_args()


def empty_accumulator():
    return {
        "cells": 0.0,
        "depth_abs": 0.0,
        "depth_sq": 0.0,
        "depth_signed": 0.0,
        "predicted_wet": 0.0,
        "component_count": 0.0,
        "component_abs": 0.0,
        "component_sq": 0.0,
        "speed_abs": 0.0,
        "speed_sq": 0.0,
        "speed_signed": 0.0,
        "true_speed_sum": 0.0,
        "direction_count": 0.0,
        "direction_abs_degrees": 0.0,
    }


def finalize(accumulator, valid_cells):
    cells = max(accumulator["cells"], 1.0)
    components = max(accumulator["component_count"], 1.0)
    directions = max(accumulator["direction_count"], 1.0)
    return {
        "cell_count": int(accumulator["cells"]),
        "valid_cell_fraction": accumulator["cells"] / max(valid_cells, 1.0),
        "depth_mae": accumulator["depth_abs"] / cells,
        "depth_rmse": float(np.sqrt(accumulator["depth_sq"] / cells)),
        "depth_bias": accumulator["depth_signed"] / cells,
        "predicted_wet_rate": accumulator["predicted_wet"] / cells,
        "component_mae": accumulator["component_abs"] / components,
        "component_rmse": float(np.sqrt(accumulator["component_sq"] / components)),
        "speed_mae": accumulator["speed_abs"] / cells,
        "speed_rmse": float(np.sqrt(accumulator["speed_sq"] / cells)),
        "speed_bias": accumulator["speed_signed"] / cells,
        "mean_true_speed": accumulator["true_speed_sum"] / cells,
        "direction_cell_count": int(accumulator["direction_count"]),
        "direction_mae_degrees": accumulator["direction_abs_degrees"] / directions,
    }


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    device = resolve_device(args.device)
    if bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
    bundle = prepare_stage1_data(
        manifest_dir=Path(config["manifest_dir"]),
        events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]),
        labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]),
        base_dir=Path(config["base_dir"]),
        test_events=config["test_events"],
        val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]),
        batch_size=int(config["batch_size"]),
        train_batches_per_epoch=1,
        eval_batches=args.eval_batches,
        train_time_stride=1,
        eval_time_stride=args.eval_time_stride,
        wet_threshold=float(config["wet_threshold"]),
        feature_columns=config["block_feature_columns"],
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config.get("max_open_netcdf_handles", 8)),
    )
    loader = make_loader(bundle.test_dataset, bundle.test_sampler, args.num_workers, device)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1TimestampModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    accumulators = {name: empty_accumulator() for name, _, _ in BINS}
    valid_cells = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            depth, wet_logits, cx, cy = model(
                batch["event"], batch["time_index"], batch["time_features"],
                batch["block_features"], batch["static"], batch["mask"],
                shared_event_time=True,
            )
            if bool(config.get("couple_depth_with_wet_probability", False)):
                depth = depth * torch.sigmoid(wet_logits)
            valid = batch["mask"] > 0.5
            true_depth = batch["depth"]
            true_cx = batch["component_x"]
            true_cy = batch["component_y"]
            predicted_wet = torch.sigmoid(wet_logits) >= 0.5
            true_speed = torch.hypot(true_cx, true_cy)
            predicted_speed = torch.hypot(cx, cy)
            valid_cells += float(valid.sum())
            for name, lower, upper in BINS:
                selected = valid & (true_depth >= lower)
                if np.isfinite(upper):
                    selected &= true_depth < upper
                count = int(selected.sum())
                if count == 0:
                    continue
                acc = accumulators[name]
                depth_error = (depth - true_depth)[selected]
                cx_error = (cx - true_cx)[selected]
                cy_error = (cy - true_cy)[selected]
                speed_error = (predicted_speed - true_speed)[selected]
                acc["cells"] += count
                acc["depth_abs"] += float(depth_error.abs().sum())
                acc["depth_sq"] += float(depth_error.square().sum())
                acc["depth_signed"] += float(depth_error.sum())
                acc["predicted_wet"] += float(predicted_wet[selected].sum())
                acc["component_count"] += 2 * count
                acc["component_abs"] += float(cx_error.abs().sum() + cy_error.abs().sum())
                acc["component_sq"] += float(cx_error.square().sum() + cy_error.square().sum())
                acc["speed_abs"] += float(speed_error.abs().sum())
                acc["speed_sq"] += float(speed_error.square().sum())
                acc["speed_signed"] += float(speed_error.sum())
                acc["true_speed_sum"] += float(true_speed[selected].sum())
                directional = selected & (true_speed >= args.direction_speed_threshold)
                direction_count = int(directional.sum())
                if direction_count:
                    true_angle = torch.atan2(true_cy[directional], true_cx[directional])
                    predicted_angle = torch.atan2(cy[directional], cx[directional])
                    angle_error = torch.atan2(
                        torch.sin(predicted_angle - true_angle),
                        torch.cos(predicted_angle - true_angle),
                    ).abs() * (180.0 / np.pi)
                    acc["direction_count"] += direction_count
                    acc["direction_abs_degrees"] += float(angle_error.sum())
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "test_events": config["test_events"],
        "eval_batches": args.eval_batches,
        "eval_time_stride": args.eval_time_stride,
        "direction_speed_threshold": args.direction_speed_threshold,
        "valid_cells": int(valid_cells),
        "depth_bins": {
            name: {"lower_m": lower, "upper_m": None if not np.isfinite(upper) else upper, **finalize(accumulators[name], valid_cells)}
            for name, lower, upper in BINS
        },
    }
    args.output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_path.resolve().write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
