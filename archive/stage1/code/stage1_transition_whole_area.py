#!/usr/bin/env python3
"""Reconstruct teacher-forced transition predictions over the complete domain."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage1_data import prepare_stage1_data
from stage1_transition_data import Stage1TransitionDataset
from stage1_transition_model import Stage1StateTransitionModel
from stage1_transition_rollout import reconstruct_components
from stage1_transition_train import move_batch, resolve_device
from stage1_whole_area_inference import (
    VARIABLES,
    calculate_metrics,
    fill_block,
    plot_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-id", default="D030")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--time-indices", type=int, nargs="+", default=[60, 140, 240, 360, 440])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wet-probability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--component-update-mode",
        choices=("learned", "component_persistence", "velocity_persistence"),
        default="learned",
    )
    parser.add_argument("--max-display-size", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def empty_maps(shape):
    return {
        key: np.full(shape, np.nan, dtype=np.float32) for key in VARIABLES
    }


def prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main():
    args = parse_args()
    if not 0.0 <= args.wet_probability_threshold <= 1.0:
        raise ValueError("Wet probability threshold must be between zero and one")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    maps_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text())
    wet_threshold = float(config["wet_threshold"])
    lag = int(config.get("lag", 1))
    history_states = int(config.get("model_config", {}).get("history_states", 1))
    device = resolve_device(args.device)
    if bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False

    bundle = prepare_stage1_data(
        manifest_dir=Path(config["manifest_dir"]),
        events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]),
        labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]),
        base_dir=Path("."),
        test_events=[args.event_id],
        val_fraction=0.2,
        seed=int(config["seed"]),
        batch_size=args.batch_size,
        train_batches_per_epoch=1,
        eval_batches=1,
        train_time_stride=1,
        eval_time_stride=1,
        wet_threshold=wet_threshold,
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config.get("max_open_netcdf_handles", 8)),
        minimum_time_index=lag * history_states,
    )
    base_dataset = bundle.test_dataset
    dataset = Stage1TransitionDataset(
        base_dataset, lag=lag, history_states=history_states
    )
    matches = base_dataset.events.index[
        base_dataset.events["event_id"] == args.event_id
    ].tolist()
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely locate test event {args.event_id}")
    event_position = int(matches[0])
    event_row = base_dataset.events.iloc[event_position]
    n_times = int(event_row["n_times"])
    times = sorted(set(int(value) for value in args.time_indices))
    invalid = [
        value
        for value in times
        if value < lag * history_states or value >= n_times
    ]
    if invalid:
        raise ValueError(f"Invalid transition target times: {invalid}")

    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1StateTransitionModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    block_index_grid = np.asarray(base_dataset.block_index_grid)
    valid_domain = block_index_grid >= 0
    n_blocks = len(base_dataset.block_rows)
    records = []
    with torch.no_grad():
        for sequence, time_index in enumerate(times, start=1):
            print(f"[Time {sequence}/{len(times)}] timestep={time_index}", flush=True)
            true_maps = empty_maps(block_index_grid.shape)
            raw_maps = empty_maps(block_index_grid.shape)
            persistence_maps = empty_maps(block_index_grid.shape)
            predicted_wet = np.zeros(block_index_grid.shape, dtype=bool)
            persistence_wet = np.zeros(block_index_grid.shape, dtype=bool)
            keys = [
                (event_position, time_index, block_position)
                for block_position in range(n_blocks)
            ]
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=keys,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            offset = 0
            for batch in loader:
                batch = move_batch(batch, device)
                output = model(
                    batch["event"], batch["time_index"], batch["time_features"],
                    batch["block_features"], batch["static"], batch["mask"],
                    batch["previous_depth"], batch["previous_component_x"],
                    batch["previous_component_y"], shared_event_time=True,
                    older_depth=batch.get("older_depth"),
                    older_component_x=batch.get("older_component_x"),
                    older_component_y=batch.get("older_component_y"),
                )
                reconstructed_x, reconstructed_y = reconstruct_components(
                    args.component_update_mode,
                    batch["previous_depth"],
                    batch["previous_component_x"],
                    batch["previous_component_y"],
                    output[0],
                    output[2],
                    output[3],
                    batch["mask"],
                    wet_threshold,
                )
                count = output[0].shape[0]
                predicted_batch = {
                    "depth": output[0].cpu().numpy(),
                    "component_x": reconstructed_x.cpu().numpy(),
                    "component_y": reconstructed_y.cpu().numpy(),
                }
                true_batch = {
                    key: batch[key].cpu().numpy() for key in VARIABLES
                }
                previous_batch = {
                    "depth": batch["previous_depth"].cpu().numpy(),
                    "component_x": batch["previous_component_x"].cpu().numpy(),
                    "component_y": batch["previous_component_y"].cpu().numpy(),
                }
                wet_probability = torch.sigmoid(output[1]).cpu().numpy()
                for local in range(count):
                    block_position = offset + local
                    block_index = int(
                        base_dataset.block_rows.iloc[block_position]["block_index"]
                    )
                    window = base_dataset.block_windows[block_index]
                    for variable in VARIABLES:
                        fill_block(
                            true_maps[variable], true_batch[variable][local],
                            block_index, window, block_index_grid,
                        )
                        fill_block(
                            raw_maps[variable], predicted_batch[variable][local],
                            block_index, window, block_index_grid,
                        )
                        fill_block(
                            persistence_maps[variable], previous_batch[variable][local],
                            block_index, window, block_index_grid,
                        )
                    fill_block(
                        predicted_wet,
                        wet_probability[local] >= args.wet_probability_threshold,
                        block_index, window, block_index_grid,
                    )
                    fill_block(
                        persistence_wet,
                        previous_batch["depth"][local] >= wet_threshold,
                        block_index, window, block_index_grid,
                    )
                offset += count
            if offset != n_blocks:
                raise RuntimeError(f"Expected {n_blocks} blocks but reconstructed {offset}")
            for maps in (true_maps, raw_maps, persistence_maps):
                for variable in VARIABLES:
                    maps[variable][~valid_domain] = np.nan
            gated_maps = {key: value.copy() for key, value in raw_maps.items()}
            for variable in VARIABLES:
                gated_maps[variable][~predicted_wet] = 0.0

            raw_metrics = calculate_metrics(
                true_maps, raw_maps, wet_threshold, predicted_wet,
                bundle.component_semantics,
            )
            gated_metrics = calculate_metrics(
                true_maps, gated_maps, wet_threshold, predicted_wet,
                bundle.component_semantics,
            )
            persistence_metrics = calculate_metrics(
                true_maps, persistence_maps, wet_threshold, persistence_wet,
                bundle.component_semantics,
            )
            elapsed_hours = float(event_row["time_start"]) + time_index * float(
                event_row["time_step"]
            )
            output_path = maps_dir / f"D030_t{time_index:03d}_transition_gated.png"
            display_stride = plot_timestamp(
                true_maps, gated_maps, time_index, elapsed_hours, output_path,
                args.max_display_size, args.dpi,
                component_label="Unit discharge", component_units="m²/s",
            )
            record = {
                "event_id": args.event_id,
                "time_index": time_index,
                "elapsed_hours": elapsed_hours,
                "path": str(output_path),
                "display_stride": display_stride,
                **prefix_metrics("raw", raw_metrics),
                **prefix_metrics("gated", gated_metrics),
                **prefix_metrics("persistence", persistence_metrics),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    with (output_dir / "whole_area_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "component_update_mode": args.component_update_mode,
        "event_id": args.event_id,
        "domain_shape": list(block_index_grid.shape),
        "n_blocks": n_blocks,
        "time_indices": times,
        "transition_lag": lag,
        "wet_threshold": wet_threshold,
        "wet_probability_threshold": args.wet_probability_threshold,
        "figures": records,
    }
    (output_dir / "whole_area_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(records)} whole-area transition figures to {maps_dir}")


if __name__ == "__main__":
    main()
