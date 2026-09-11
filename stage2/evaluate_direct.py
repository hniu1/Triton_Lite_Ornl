#!/usr/bin/env python3
"""Evaluate a direct V5 checkpoint from 0.5 through 24 forecast hours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage2.data import DirectHorizonDataset
from stage2.direct import evaluate_direct_horizons
from stage2.model import Stage2DirectLeadModel
from triton_core.data import prepare_hydraulic_data
from triton_core.runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(range(1, 49)))
    parser.add_argument("--evaluation-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start-time-stride", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    device = resolve_device(args.device)
    if args.disable_cudnn or bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
    max_horizon = int(config["max_horizon_steps"])
    horizons = sorted(set(int(value) for value in args.horizons))
    if not horizons or horizons[0] < 1 or horizons[-1] > max_horizon:
        raise ValueError(f"Horizons must lie inside [1, {max_horizon}]")

    bundle = prepare_hydraulic_data(
        manifest_dir=Path(config["manifest_dir"]),
        events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]),
        labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]),
        base_dir=Path("."),
        test_events=config["test_events"],
        val_fraction=0.2,
        seed=int(config["data_split_seed"]),
        batch_size=args.batch_size,
        train_batches_per_epoch=1,
        eval_batches=1,
        train_time_stride=1,
        eval_time_stride=1,
        wet_threshold=float(config["wet_threshold"]),
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config["max_open_netcdf_handles"]),
    )
    dataset = DirectHorizonDataset(
        getattr(bundle, f"{args.split}_dataset"),
        max_horizon_steps=max_horizon,
        step_hours=float(config["step_hours"]),
        history_states=int(config["model_config"]["history_states"]),
    )
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage2DirectLeadModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, cohort_batches = evaluate_direct_horizons(
        model=model,
        dataset=dataset,
        device=device,
        horizons=horizons,
        batch_size=args.batch_size,
        batch_limit=args.evaluation_batches,
        start_time_stride=args.start_time_stride,
        wet_threshold=float(config["wet_threshold"]),
        component_semantics=bundle.component_semantics,
    )
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "split_events": config["split_events"][args.split],
        "forecast_mode": config["forecast_mode"],
        "forcing_availability": config["forcing_availability"],
        "horizons_steps": horizons,
        "step_hours": float(config["step_hours"]),
        "cohort_batches": cohort_batches,
        "batch_size": args.batch_size,
        "start_time_stride": args.start_time_stride,
        "metrics": metrics,
    }
    output_path = (
        args.output_path
        or run_dir / f"{args.split}_dense_24h_direct_metrics.json"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    for horizon in horizons:
        values = metrics[str(horizon)]
        direct = values["direct"]
        persistence = values["persistence"]
        print(
            f"lead={values['lead_hours']:g}h "
            f"direct_depth={direct['depth_wet_rmse']:.4f} "
            f"persistence_depth={persistence['depth_wet_rmse']:.4f} "
            f"direct_f1={direct['wet_f1']:.4f} "
            f"persistence_f1={persistence['wet_f1']:.4f}",
            flush=True,
        )
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
