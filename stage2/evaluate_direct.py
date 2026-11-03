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
from stage2.graph_data import GraphNeighborhoodDataset, TileTopology
from stage2.target_integrity import require_corrected_targets
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
    parser.add_argument("--wet-probability-threshold", type=float, default=0.5)
    parser.add_argument("--extent-calibration", type=Path, default=None)
    parser.add_argument(
        "--wet-classification-mode", choices=["head", "depth", "or", "and"],
        default="head",
    )
    parser.add_argument(
        "--component-policy",
        choices=["learned", "component_persistence", "velocity_persistence"],
        default="learned",
    )
    parser.add_argument("--per-event", action="store_true")
    parser.add_argument("--depth-diagnostics",action="store_true")
    parser.add_argument("--cohort", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    require_corrected_targets(config["manifest_dir"])
    if config.get('forcing_weights_sha256'):
        from stage2.prepare_dynamic_stream import digest_file
        if digest_file(config['forcing_weights_path']) != config['forcing_weights_sha256']:
            raise ValueError('Forcing weights changed since training')
    device = resolve_device(args.device)
    if args.disable_cudnn or bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
    max_horizon = int(config["max_horizon_steps"])
    horizons = sorted(set(int(value) for value in args.horizons))
    if not horizons or horizons[0] < 1 or horizons[-1] > max_horizon:
        raise ValueError(f"Horizons must lie inside [1, {max_horizon}]")
    if not 0.0 <= args.wet_probability_threshold <= 1.0:
        raise ValueError("Wet probability threshold must lie in [0, 1]")
    extent_mode = args.wet_classification_mode
    extent_threshold = args.wet_probability_threshold
    calibration_source = None
    if args.extent_calibration is not None:
        calibration_source = args.extent_calibration.resolve()
        calibration = json.loads(calibration_source.read_text())
        extent_mode = calibration["selected_mode"]
        extent_threshold = float(calibration["selected_probability_threshold"])
        if not 0.0 <= extent_threshold <= 1.0:
            raise ValueError("Calibrated wet probability threshold must lie in [0, 1]")

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
        hydraulic_halo_cells=int(config.get("hydraulic_halo_cells", 0)),
        forcing_weights_path=(
            Path(config["forcing_weights_path"])
            if config.get("forcing_weights_path")
            else None
        ),
    )
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage2DirectLeadModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if config.get("graph_hops", 0):
        topology = TileTopology.from_raster(dataset.block_index_grid, dataset.block_rows["block_index"])
        if topology.digest != config.get("graph_topology_sha256"):
            raise ValueError("Evaluation topology differs from training")
        if int(config["graph_hops"]) < checkpoint["model_config"].get("graph_layers", 0):
            raise ValueError("Graph context does not cover message passing layers")
        dataset = GraphNeighborhoodDataset(dataset, topology, int(config["graph_hops"]))
    target_batches, cohort = None, None
    if args.cohort is not None:
        from stage2.dynamic_stream_cohort import load_batches, digest_file
        if args.output_path is None or args.split != "test":
            raise ValueError("Conditional evaluation requires a separate output path and test split")
        target_batches, cohort = load_batches(args.cohort, dataset, horizons, args.batch_size)
    evaluation = evaluate_direct_horizons(
        model=model,
        dataset=dataset,
        device=device,
        horizons=horizons,
        batch_size=args.batch_size,
        batch_limit=args.evaluation_batches,
        start_time_stride=args.start_time_stride,
        wet_threshold=float(config["wet_threshold"]),
        component_semantics=bundle.component_semantics,
        wet_probability_threshold=extent_threshold,
        wet_classification_mode=extent_mode,
        component_policy=args.component_policy,
        return_per_event=args.per_event,
        depth_extent_only=bool(config.get("depth_extent_only",False)),
        depth_diagnostics=args.depth_diagnostics,
        target_batches=target_batches,
        allow_partial_target_history=bool(cohort and cohort["rule"].get("lead_eligibility") == "available_origin_history"),
    )
    if args.per_event:
        metrics, cohort_batches, per_event_metrics = evaluation
    else:
        metrics, cohort_batches = evaluation
        per_event_metrics = None
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "trained_outputs": config.get("trained_outputs",["depth","wet_extent","unit_discharge"]),
        "depth_extent_only": bool(config.get("depth_extent_only",False)),
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
        "wet_probability_threshold": extent_threshold,
        "wet_classification_mode": extent_mode,
        "extent_calibration": str(calibration_source) if calibration_source else None,
        "component_policy": args.component_policy,
        "metrics": metrics,
    }
    if cohort is not None:
        payload.update(cohort_type="dynamic_stream_flood_eval", cohort_sha256=digest_file(args.cohort),
                       cohort_path=str(args.cohort.resolve()), selection_rule=cohort["rule"],
                       coverage=cohort["coverage"], target_first=True)
    if per_event_metrics is not None:
        payload["per_event_metrics"] = per_event_metrics
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
