#!/usr/bin/env python3
"""Evaluate teacher-forced and autoregressive Stage-1 transition forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from stage1_data import prepare_stage1_data
from stage1_train import MetricAccumulator
from stage1_transition_model import Stage1StateTransitionModel
from stage1_transition_train import resolve_device, transition_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12, 24])
    parser.add_argument("--rollout-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start-time-stride", type=int, default=12)
    parser.add_argument("--step-hours", type=float, default=0.5)
    parser.add_argument(
        "--component-update-mode",
        choices=("learned", "component_persistence", "velocity_persistence"),
        default="learned",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def select_rollout_batches(
    n_times,
    n_blocks,
    batch_size,
    max_horizon,
    time_stride,
    batch_limit,
    minimum_start_time=0,
):
    """Select deterministic event/start/local-block batches over the test domain."""

    candidates = []
    for event_position, event_times in enumerate(n_times):
        latest_start = int(event_times) - int(max_horizon) - 1
        for start_time in range(
            int(minimum_start_time), latest_start + 1, int(time_stride)
        ):
            for block_start in range(0, int(n_blocks), int(batch_size)):
                candidates.append(
                    (
                        int(event_position),
                        int(start_time),
                        tuple(
                            range(
                                block_start,
                                min(block_start + int(batch_size), int(n_blocks)),
                            )
                        ),
                    )
                )
    if not candidates:
        raise ValueError("No valid rollout batches for the requested maximum horizon")
    if batch_limit <= 0 or batch_limit >= len(candidates):
        return candidates
    selected = np.linspace(0, len(candidates) - 1, int(batch_limit), dtype=np.int64)
    return [candidates[int(index)] for index in selected]


def load_batch(dataset, event_position, time_index, block_positions, device):
    batch = default_collate(
        [
            dataset[(event_position, time_index, block_position)]
            for block_position in block_positions
        ]
    )
    for key in (
        "event",
        "time_index",
        "time_features",
        "block_features",
        "static",
        "mask",
        "depth",
        "component_x",
        "component_y",
    ):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def predict(
    model,
    batch,
    previous_depth,
    previous_x,
    previous_y,
    older_depth=None,
    older_x=None,
    older_y=None,
):
    return model(
        batch["event"],
        batch["time_index"],
        batch["time_features"],
        batch["block_features"],
        batch["static"],
        batch["mask"],
        previous_depth,
        previous_x,
        previous_y,
        shared_event_time=True,
        older_depth=older_depth,
        older_component_x=older_x,
        older_component_y=older_y,
    )


def reconstruct_components(
    mode,
    previous_depth,
    previous_x,
    previous_y,
    predicted_depth,
    learned_x,
    learned_y,
    mask,
    wet_threshold,
):
    """Apply an explicit hydraulic component-update policy."""

    if mode == "learned":
        return learned_x, learned_y
    if mode == "component_persistence":
        return previous_x * mask, previous_y * mask
    if mode != "velocity_persistence":
        raise ValueError(f"Unknown component update mode: {mode}")
    previous_wet = previous_depth >= wet_threshold
    denominator = previous_depth.clamp_min(wet_threshold)
    velocity_x = torch.where(previous_wet, previous_x / denominator, 0.0)
    velocity_y = torch.where(previous_wet, previous_y / denominator, 0.0)
    return velocity_x * predicted_depth * mask, velocity_y * predicted_depth * mask


def finalize(accumulators):
    metrics = {}
    for name, accumulator in accumulators.items():
        values = accumulator.finalize()
        values["physical_score"] = transition_score(values)
        metrics[name] = values
    return metrics


def main():
    args = parse_args()
    horizons = sorted(set(int(value) for value in args.horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("All rollout horizons must be positive")
    if args.batch_size < 1 or args.start_time_stride < 1:
        raise ValueError("Batch size and start-time stride must be positive")

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    device = resolve_device(args.device)
    if args.disable_cudnn or bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False

    bundle = prepare_stage1_data(
        manifest_dir=Path(config["manifest_dir"]),
        events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]),
        labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]),
        base_dir=Path("."),
        test_events=config["test_events"],
        val_fraction=0.2,
        seed=int(config["seed"]),
        batch_size=args.batch_size,
        train_batches_per_epoch=1,
        eval_batches=1,
        train_time_stride=1,
        eval_time_stride=1,
        wet_threshold=float(config["wet_threshold"]),
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config.get("max_open_netcdf_handles", 8)),
    )
    dataset = bundle.test_dataset
    n_times = dataset.events["n_times"].astype(int).tolist()
    history_states = int(config.get("model_config", {}).get("history_states", 1))
    selected = select_rollout_batches(
        n_times=n_times,
        n_blocks=len(dataset.block_rows),
        batch_size=args.batch_size,
        max_horizon=max(horizons),
        time_stride=args.start_time_stride,
        batch_limit=args.rollout_batches,
        minimum_start_time=history_states - 1,
    )

    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1StateTransitionModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulators = {
        horizon: {
            "autoregressive": MetricAccumulator(),
            "teacher_forced": MetricAccumulator(),
            "persistence": MetricAccumulator(),
        }
        for horizon in horizons
    }
    wet_threshold = float(config["wet_threshold"])
    with torch.no_grad():
        for batch_number, (event_position, start_time, block_positions) in enumerate(
            selected, start=1
        ):
            initial = load_batch(
                dataset, event_position, start_time, block_positions, device
            )
            predicted_depth = initial["depth"]
            predicted_x = initial["component_x"]
            predicted_y = initial["component_y"]
            predicted_older = None
            if history_states == 2:
                predicted_older = load_batch(
                    dataset,
                    event_position,
                    start_time - 1,
                    block_positions,
                    device,
                )
            true_previous = initial
            true_older = predicted_older
            persistence_wet = (initial["depth"] >= wet_threshold).float()

            for step in range(1, max(horizons) + 1):
                target = load_batch(
                    dataset, event_position, start_time + step, block_positions, device
                )
                input_depth, input_x, input_y = predicted_depth, predicted_x, predicted_y
                output = predict(
                    model,
                    target,
                    input_depth,
                    input_x,
                    input_y,
                    None if predicted_older is None else predicted_older["depth"],
                    None if predicted_older is None else predicted_older["component_x"],
                    None if predicted_older is None else predicted_older["component_y"],
                )
                predicted_depth, wet_logits, predicted_x, predicted_y = output[:4]
                predicted_x, predicted_y = reconstruct_components(
                    args.component_update_mode,
                    input_depth,
                    input_x,
                    input_y,
                    predicted_depth,
                    predicted_x,
                    predicted_y,
                    target["mask"],
                    wet_threshold,
                )

                if step in accumulators:
                    teacher = predict(
                        model,
                        target,
                        true_previous["depth"],
                        true_previous["component_x"],
                        true_previous["component_y"],
                        None if true_older is None else true_older["depth"],
                        None if true_older is None else true_older["component_x"],
                        None if true_older is None else true_older["component_y"],
                    )
                    teacher_x, teacher_y = reconstruct_components(
                        args.component_update_mode,
                        true_previous["depth"],
                        true_previous["component_x"],
                        true_previous["component_y"],
                        teacher[0],
                        teacher[2],
                        teacher[3],
                        target["mask"],
                        wet_threshold,
                    )
                    accumulators[step]["autoregressive"].update(
                        predicted_depth,
                        torch.sigmoid(wet_logits),
                        predicted_x,
                        predicted_y,
                        target,
                        wet_threshold,
                        1.0,
                        bundle.component_semantics,
                    )
                    accumulators[step]["teacher_forced"].update(
                        teacher[0],
                        torch.sigmoid(teacher[1]),
                        teacher_x,
                        teacher_y,
                        target,
                        wet_threshold,
                        1.0,
                        bundle.component_semantics,
                    )
                    accumulators[step]["persistence"].update(
                        initial["depth"],
                        persistence_wet,
                        initial["component_x"],
                        initial["component_y"],
                        target,
                        wet_threshold,
                        1.0,
                        bundle.component_semantics,
                    )
                if history_states == 2:
                    predicted_older = {
                        "depth": input_depth,
                        "component_x": input_x,
                        "component_y": input_y,
                    }
                    true_older = true_previous
                true_previous = target
            if batch_number % 10 == 0 or batch_number == len(selected):
                print(f"rollout_batches={batch_number}/{len(selected)}", flush=True)

    horizon_metrics = {}
    for horizon in horizons:
        values = finalize(accumulators[horizon])
        learned = values["autoregressive"]
        persistence = values["persistence"]
        values["autoregressive_vs_persistence"] = {
            "depth_wet_rmse_change_percent": 100.0
            * (learned["depth_wet_rmse"] - persistence["depth_wet_rmse"])
            / max(persistence["depth_wet_rmse"], 1e-12),
            "component_rmse_change_percent": 100.0
            * (learned["component_rmse"] - persistence["component_rmse"])
            / max(persistence["component_rmse"], 1e-12),
            "wet_f1_absolute_change": learned["wet_f1"] - persistence["wet_f1"],
            "physical_score_change_percent": 100.0
            * (learned["physical_score"] - persistence["physical_score"])
            / max(persistence["physical_score"], 1e-12),
        }
        if "derived_velocity_rmse" in learned and "derived_velocity_rmse" in persistence:
            values["autoregressive_vs_persistence"][
                "derived_velocity_rmse_change_percent"
            ] = 100.0 * (
                learned["derived_velocity_rmse"]
                - persistence["derived_velocity_rmse"]
            ) / max(persistence["derived_velocity_rmse"], 1e-12)
        horizon_metrics[str(horizon)] = values
        print(
            f"horizon={horizon} steps ({horizon * args.step_hours:g} h) "
            f"autoregressive_depth={learned['depth_wet_rmse']:.4f} "
            f"persistence_depth={persistence['depth_wet_rmse']:.4f} "
            f"autoregressive_f1={learned['wet_f1']:.4f} "
            f"persistence_f1={persistence['wet_f1']:.4f}",
            flush=True,
        )

    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_events": config["test_events"],
        "horizons_steps": horizons,
        "step_hours": args.step_hours,
        "rollout_batches": len(selected),
        "batch_size": args.batch_size,
        "start_time_stride": args.start_time_stride,
        "component_update_mode": args.component_update_mode,
        "metrics": horizon_metrics,
    }
    output_path = (
        args.output_path or run_dir / "rollout_metrics.json"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
