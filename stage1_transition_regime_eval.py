#!/usr/bin/env python3
"""Evaluate one-step transition skill separately by hydraulic-change regime."""

import argparse
import json
from pathlib import Path

import torch

from stage1_data import prepare_stage1_data
from stage1_train import MetricAccumulator
from stage1_transition_model import Stage1StateTransitionModel
from stage1_transition_rollout import load_batch, predict, select_rollout_batches
from stage1_transition_train import resolve_device, transition_score


REGIMES = ("all", "stable", "filling", "draining", "rapid")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-stride", type=int, default=3)
    parser.add_argument("--stable-depth-threshold", type=float, default=0.01)
    parser.add_argument("--stable-extent-threshold", type=float, default=0.01)
    parser.add_argument("--rapid-depth-threshold", type=float, default=0.10)
    parser.add_argument("--rapid-extent-threshold", type=float, default=0.05)
    parser.add_argument("--extent-direction-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def classify_transition_regimes(
    previous_depth,
    target_depth,
    mask,
    wet_threshold,
    stable_depth_threshold=0.01,
    stable_extent_threshold=0.01,
    rapid_depth_threshold=0.10,
    rapid_extent_threshold=0.05,
    extent_direction_scale=1.0,
):
    """Return per-patch regime masks from exact consecutive hydraulic states."""

    previous_wet = (previous_depth >= wet_threshold).float() * mask
    target_wet = (target_depth >= wet_threshold).float() * mask
    active = torch.maximum(previous_wet, target_wet)
    delta = target_depth - previous_depth
    active_count = active.sum(dim=(-2, -1)).clamp_min(1.0)
    valid_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    signed_depth_change = (delta * active).sum(dim=(-2, -1)) / active_count
    mean_abs_depth_change = (delta.abs() * active).sum(dim=(-2, -1)) / active_count
    extent_change = (target_wet - previous_wet).sum(dim=(-2, -1)) / valid_count

    stable = (
        (mean_abs_depth_change < stable_depth_threshold)
        & (extent_change.abs() < stable_extent_threshold)
    )
    direction = signed_depth_change + extent_direction_scale * extent_change
    filling = (~stable) & (direction >= 0)
    draining = (~stable) & (direction < 0)
    rapid = (
        (mean_abs_depth_change >= rapid_depth_threshold)
        | (extent_change.abs() >= rapid_extent_threshold)
    )
    return {
        "all": torch.ones_like(stable, dtype=torch.bool),
        "stable": stable,
        "filling": filling,
        "draining": draining,
        "rapid": rapid,
    }, {
        "signed_depth_change": signed_depth_change,
        "mean_abs_depth_change": mean_abs_depth_change,
        "wet_extent_change": extent_change,
    }


def subset_target(target, selection):
    return {
        key: value[selection] if torch.is_tensor(value) and value.ndim > 0 else value
        for key, value in target.items()
    }


def finalized_metrics(accumulators, counts):
    result = {}
    for regime, methods in accumulators.items():
        if counts[regime] == 0:
            continue
        result[regime] = {"patches": counts[regime]}
        for method, accumulator in methods.items():
            values = accumulator.finalize()
            values["physical_score"] = transition_score(values)
            result[regime][method] = values
        learned = result[regime]["model"]
        persistence = result[regime]["persistence"]
        result[regime]["model_vs_persistence"] = {
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
        if "derived_velocity_rmse" in learned:
            result[regime]["model_vs_persistence"][
                "derived_velocity_rmse_change_percent"
            ] = 100.0 * (
                learned["derived_velocity_rmse"]
                - persistence["derived_velocity_rmse"]
            ) / max(persistence["derived_velocity_rmse"], 1e-12)
    return result


def main():
    args = parse_args()
    for value in (
        args.stable_depth_threshold,
        args.stable_extent_threshold,
        args.rapid_depth_threshold,
        args.rapid_extent_threshold,
    ):
        if value < 0:
            raise ValueError("Regime thresholds must be non-negative")

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
    history_states = int(config.get("model_config", {}).get("history_states", 1))
    selected = select_rollout_batches(
        n_times=dataset.events["n_times"].astype(int).tolist(),
        n_blocks=len(dataset.block_rows),
        batch_size=args.batch_size,
        max_horizon=1,
        time_stride=args.time_stride,
        batch_limit=args.batches,
        minimum_start_time=history_states - 1,
    )

    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1StateTransitionModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulators = {
        regime: {"model": MetricAccumulator(), "persistence": MetricAccumulator()}
        for regime in REGIMES
    }
    counts = {regime: 0 for regime in REGIMES}
    diagnostic_sums = {
        name: {regime: 0.0 for regime in REGIMES}
        for name in (
            "signed_depth_change",
            "mean_abs_depth_change",
            "wet_extent_change",
        )
    }
    wet_threshold = float(config["wet_threshold"])
    with torch.no_grad():
        for number, (event_position, start_time, blocks) in enumerate(selected, 1):
            previous = load_batch(dataset, event_position, start_time, blocks, device)
            older = None
            if history_states == 2:
                older = load_batch(
                    dataset, event_position, start_time - 1, blocks, device
                )
            target = load_batch(dataset, event_position, start_time + 1, blocks, device)
            output = predict(
                model,
                target,
                previous["depth"],
                previous["component_x"],
                previous["component_y"],
                None if older is None else older["depth"],
                None if older is None else older["component_x"],
                None if older is None else older["component_y"],
            )
            regimes, diagnostics = classify_transition_regimes(
                previous["depth"],
                target["depth"],
                target["mask"],
                wet_threshold,
                args.stable_depth_threshold,
                args.stable_extent_threshold,
                args.rapid_depth_threshold,
                args.rapid_extent_threshold,
                args.extent_direction_scale,
            )
            for regime, selection in regimes.items():
                count = int(selection.sum())
                if count == 0:
                    continue
                counts[regime] += count
                selected_target = subset_target(target, selection)
                accumulators[regime]["model"].update(
                    output[0][selection],
                    torch.sigmoid(output[1][selection]),
                    output[2][selection],
                    output[3][selection],
                    selected_target,
                    wet_threshold,
                    1.0,
                    bundle.component_semantics,
                )
                persistence_wet = (
                    previous["depth"][selection] >= wet_threshold
                ).float()
                accumulators[regime]["persistence"].update(
                    previous["depth"][selection],
                    persistence_wet,
                    previous["component_x"][selection],
                    previous["component_y"][selection],
                    selected_target,
                    wet_threshold,
                    1.0,
                    bundle.component_semantics,
                )
                for name, values in diagnostics.items():
                    diagnostic_sums[name][regime] += float(values[selection].sum())
            if number % 20 == 0 or number == len(selected):
                print(f"regime_batches={number}/{len(selected)}", flush=True)

    metrics = finalized_metrics(accumulators, counts)
    for regime, values in metrics.items():
        values["mean_transition_diagnostics"] = {
            name: diagnostic_sums[name][regime] / counts[regime]
            for name in diagnostic_sums
        }
        comparison = values["model_vs_persistence"]
        print(
            f"regime={regime} patches={counts[regime]} "
            f"depth_change={comparison['depth_wet_rmse_change_percent']:.2f}% "
            f"f1_change={comparison['wet_f1_absolute_change']:+.4f}",
            flush=True,
        )

    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_events": config["test_events"],
        "batches": len(selected),
        "batch_size": args.batch_size,
        "time_stride": args.time_stride,
        "thresholds": {
            "wet": wet_threshold,
            "stable_depth": args.stable_depth_threshold,
            "stable_extent": args.stable_extent_threshold,
            "rapid_depth": args.rapid_depth_threshold,
            "rapid_extent": args.rapid_extent_threshold,
            "extent_direction_scale": args.extent_direction_scale,
        },
        "metrics": metrics,
    }
    output_path = (args.output_path or run_dir / "regime_metrics.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
