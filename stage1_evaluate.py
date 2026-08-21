#!/usr/bin/env python3
"""Evaluate a saved Stage-1 checkpoint independently of the training job."""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from stage1_data import prepare_stage1_data
from stage1_model import Stage1TimestampModel
from stage1_train import make_loader, physical_selection_score, resolve_device, run_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--eval-time-stride", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    compatibility_defaults = {
        "depth_loss_mode": "physical",
        "depth_log_huber_delta": 0.2,
        "depth_physical_huber_delta": config.get("depth_huber_delta", 0.25),
        "depth_log_loss_weight": 0.0,
        "depth_physical_loss_weight": 1.0,
        "depth_weight_shallow": 1.0,
        "depth_weight_moderate": 1.0,
        "depth_weight_deep": 1.0,
        "depth_weight_extreme": 1.0,
        "wet_dice_loss_weight": 0.0,
        "wet_dice_smoothing": 1.0,
        "diagnostic_deep_threshold": 1.0,
        "couple_depth_with_wet_probability": False,
        "component_loss_mode": "component_huber",
        "speed_loss_weight": 0.0,
        "direction_loss_weight": 0.0,
        "direction_min_speed": 0.05,
        "velocity_weight_scale": 0.0,
        "velocity_weight_reference_speed": 0.25,
        "velocity_weight_cap": 3.0,
    }
    for key, value in compatibility_defaults.items():
        config.setdefault(key, value)
    train_args = Namespace(**config)
    device = resolve_device(args.device)
    if bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
    eval_batches = int(args.eval_batches or config["eval_batches"])
    eval_time_stride = int(args.eval_time_stride or config["eval_time_stride"])
    workers = int(config["num_workers"] if args.num_workers is None else args.num_workers)
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
        eval_batches=eval_batches,
        train_time_stride=int(config["train_time_stride"]),
        eval_time_stride=eval_time_stride,
        wet_threshold=float(config["wet_threshold"]),
        feature_columns=config["block_feature_columns"],
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config.get("max_open_netcdf_handles", 8)),
    )
    loader = make_loader(bundle.test_dataset, bundle.test_sampler, workers, device)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1TimestampModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, loader, device, train_args, None)
    metrics["physical_score"] = physical_selection_score(metrics, train_args)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": {
            key: checkpoint[key]
            for key in ("checkpoint_metric", "best_selection_score", "best_physical_score", "best_val_loss")
            if key in checkpoint
        },
        "test_events": config["test_events"],
        "eval_batches": eval_batches,
        "eval_time_stride": eval_time_stride,
        "test": metrics,
    }
    output_path = (args.output_path or run_dir / "evaluation_metrics.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
