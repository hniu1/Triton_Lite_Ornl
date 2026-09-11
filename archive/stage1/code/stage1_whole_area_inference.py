#!/usr/bin/env python3
"""Predict and plot a complete Stage-1 event domain at selected timestamps."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage1-whole-area")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from stage1_data import prepare_stage1_data
from stage1_model import Stage1TimestampModel
from stage1_train import move_batch, resolve_device


VARIABLES = ("depth", "component_x", "component_y")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-id", default="D030")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--time-interval", type=int, default=20)
    parser.add_argument("--time-indices", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wet-threshold", type=float, default=None)
    parser.add_argument("--wet-probability-threshold", type=float, default=None)
    parser.add_argument("--wet-calibration", type=Path, default=None)
    parser.add_argument("--no-wet-gating", action="store_true")
    parser.add_argument("--max-display-size", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def unpad(array, height, width):
    row0 = (array.shape[-2] - height) // 2
    col0 = (array.shape[-1] - width) // 2
    return array[row0 : row0 + height, col0 : col0 + width]


def fill_block(target, patch, block_index, window, block_index_grid):
    patch = unpad(patch, window.height, window.width)
    local_indices = block_index_grid[
        window.row_start : window.row_stop,
        window.col_start : window.col_stop,
    ]
    selected = local_indices == block_index
    view = target[window.row_start : window.row_stop, window.col_start : window.col_stop]
    view[selected] = patch[selected]


def robust_positive_limit(true, predicted, percentile=99.5):
    values = np.concatenate(
        [true[np.isfinite(true)].reshape(-1), predicted[np.isfinite(predicted)].reshape(-1)]
    )
    return max(float(np.percentile(values, percentile)), 1e-6)


def robust_symmetric_limit(array, percentile=99.5):
    values = np.abs(array[np.isfinite(array)])
    return max(float(np.percentile(values, percentile)), 1e-6)


def draw(ax, array, title, cmap, vmin, vmax):
    cmap_object = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap_object.set_bad("white")
    image = ax.imshow(
        np.ma.masked_invalid(array), origin="lower", cmap=cmap_object, vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_timestamp(
    true_maps, predicted_maps, time_index, elapsed_hours, output_path,
    max_display_size, dpi, component_label="Velocity", component_units="m/s",
):
    rows, cols = true_maps["depth"].shape
    stride = max(1, math.ceil(max(rows, cols) / max_display_size))
    true = {key: value[::stride, ::stride] for key, value in true_maps.items()}
    predicted = {key: value[::stride, ::stride] for key, value in predicted_maps.items()}
    true["speed"] = np.hypot(true["component_x"], true["component_y"])
    predicted["speed"] = np.hypot(predicted["component_x"], predicted["component_y"])
    rows_to_plot = [
        ("depth", "Water depth", "viridis", "positive"),
        ("component_x", f"{component_label} X", "coolwarm", "signed"),
        ("component_y", f"{component_label} Y", "coolwarm", "signed"),
        ("speed", f"{component_label} magnitude", "magma", "positive"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
    for row, (key, label, cmap, scale) in enumerate(rows_to_plot):
        error = predicted[key] - true[key]
        if scale == "positive":
            value_limit = robust_positive_limit(true[key], predicted[key])
            value_min = 0.0
        else:
            value_limit = max(
                robust_symmetric_limit(true[key]), robust_symmetric_limit(predicted[key])
            )
            value_min = -value_limit
        error_limit = robust_symmetric_limit(error)
        image_true = draw(axes[row, 0], true[key], f"True {label}", cmap, value_min, value_limit)
        draw(axes[row, 1], predicted[key], f"Predicted {label}", cmap, value_min, value_limit)
        image_error = draw(
            axes[row, 2], error, f"{label} error (pred − true)", "coolwarm", -error_limit, error_limit
        )
        units = "m" if key == "depth" else component_units
        fig.colorbar(image_true, ax=axes[row, :2], shrink=0.75, label=units)
        fig.colorbar(image_error, ax=axes[row, 2], shrink=0.75, label=units)
    fig.suptitle(
        f"D030 whole-domain prediction — timestep {time_index}, elapsed {elapsed_hours:.2f} h",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return stride


def calculate_metrics(
    true_maps, predicted_maps, wet_threshold, predicted_wet=None,
    component_semantics="unknown",
):
    valid = np.isfinite(true_maps["depth"])
    wet = valid & (true_maps["depth"] >= wet_threshold)
    depth_threshold_wet = valid & (predicted_maps["depth"] >= wet_threshold)
    if predicted_wet is None:
        predicted_wet = depth_threshold_wet
    else:
        predicted_wet = valid & predicted_wet
    depth_error = predicted_maps["depth"] - true_maps["depth"]
    cx_error = predicted_maps["component_x"] - true_maps["component_x"]
    cy_error = predicted_maps["component_y"] - true_maps["component_y"]
    tp = int((wet & predicted_wet).sum())
    fp = int((~wet & predicted_wet & valid).sum())
    fn = int((wet & ~predicted_wet).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "valid_cells": int(valid.sum()),
        "wet_cells": int(wet.sum()),
        "wet_cell_fraction": float(wet.sum() / max(valid.sum(), 1)),
        "depth_all_mae": float(np.abs(depth_error[valid]).mean()),
        "depth_all_rmse": float(np.sqrt(np.mean(depth_error[valid] ** 2))),
        "depth_wet_mae": float(np.abs(depth_error[wet]).mean()) if wet.any() else None,
        "depth_wet_rmse": float(np.sqrt(np.mean(depth_error[wet] ** 2))) if wet.any() else None,
        "component_mae": float((np.abs(cx_error[wet]).sum() + np.abs(cy_error[wet]).sum()) / max(2 * wet.sum(), 1)),
        "component_rmse": float(np.sqrt((np.square(cx_error[wet]).sum() + np.square(cy_error[wet]).sum()) / max(2 * wet.sum(), 1))),
        "wet_precision": precision,
        "wet_recall": recall,
        "wet_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "wet_csi": tp / max(tp + fp + fn, 1),
        # Backward-compatible aliases for existing whole-area result readers.
        "wet_precision_depth_threshold": precision,
        "wet_recall_depth_threshold": recall,
        "wet_f1_depth_threshold": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    if component_semantics == "unit_discharge" and wet.any():
        true_depth = np.maximum(true_maps["depth"][wet], wet_threshold)
        predicted_depth = np.maximum(predicted_maps["depth"][wet], wet_threshold)
        ux_error = (
            predicted_maps["component_x"][wet] / predicted_depth
            - true_maps["component_x"][wet] / true_depth
        )
        uy_error = (
            predicted_maps["component_y"][wet] / predicted_depth
            - true_maps["component_y"][wet] / true_depth
        )
        metrics["derived_velocity_mae"] = float(
            (np.abs(ux_error).sum() + np.abs(uy_error).sum())
            / max(2 * wet.sum(), 1)
        )
        metrics["derived_velocity_rmse"] = float(
            np.sqrt(
                (np.square(ux_error).sum() + np.square(uy_error).sum())
                / max(2 * wet.sum(), 1)
            )
        )
    return metrics


def resolve_probability_threshold(args, run_dir):
    if args.no_wet_gating:
        return None, None
    if args.wet_probability_threshold is not None:
        threshold = float(args.wet_probability_threshold)
        source = "command_line"
    else:
        path = (args.wet_calibration or run_dir / "wet_threshold_calibration.json").resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Wet calibration not found: {path}. Run stage1_calibrate_wet_threshold.py "
                "or pass --wet-probability-threshold/--no-wet-gating."
            )
        calibration = json.loads(path.read_text())
        threshold = float(calibration["selected_probability_threshold"])
        source = str(path)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Wet probability threshold must be between 0 and 1")
    return threshold, source


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    maps_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text())
    wet_threshold = float(args.wet_threshold or config["wet_threshold"])
    probability_threshold, threshold_source = resolve_probability_threshold(args, run_dir)
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
        test_events=[args.event_id],
        val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]),
        batch_size=args.batch_size,
        train_batches_per_epoch=1,
        eval_batches=1,
        train_time_stride=1,
        eval_time_stride=1,
        wet_threshold=wet_threshold,
        feature_columns=config["block_feature_columns"],
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config.get("max_open_netcdf_handles", 8)),
    )
    dataset = bundle.test_dataset
    matches = dataset.events.index[dataset.events["event_id"] == args.event_id].tolist()
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely locate test event {args.event_id}")
    event_position = int(matches[0])
    event_row = dataset.events.iloc[event_position]
    n_times = int(event_row["n_times"])
    if args.time_indices:
        time_indices = sorted(set(args.time_indices))
    else:
        time_indices = list(range(0, n_times, args.time_interval))
    invalid_times = [value for value in time_indices if value < 0 or value >= n_times]
    if invalid_times:
        raise ValueError(f"Invalid time indices: {invalid_times}")
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1TimestampModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    block_index_grid = np.asarray(dataset.block_index_grid)
    n_blocks = len(dataset.block_rows)
    valid_domain = block_index_grid >= 0
    records = []
    with torch.no_grad():
        for sequence, time_index in enumerate(time_indices, start=1):
            print(f"[Time {sequence}/{len(time_indices)}] timestep={time_index}", flush=True)
            true_maps = {
                key: np.full(block_index_grid.shape, np.nan, dtype=np.float32) for key in VARIABLES
            }
            predicted_maps = {
                key: np.full(block_index_grid.shape, np.nan, dtype=np.float32) for key in VARIABLES
            }
            predicted_wet_map = np.zeros(block_index_grid.shape, dtype=bool)
            keys = [(event_position, time_index, block_position) for block_position in range(n_blocks)]
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
                depth, wet_logits, cx, cy = model(
                    batch["event"], batch["time_index"], batch["time_features"],
                    batch["block_features"], batch["static"], batch["mask"],
                    shared_event_time=True,
                )
                if bool(config.get("couple_depth_with_wet_probability", False)):
                    depth = depth * torch.sigmoid(wet_logits)
                count = depth.shape[0]
                predicted_batch = {
                    "depth": depth.cpu().numpy(),
                    "component_x": cx.cpu().numpy(),
                    "component_y": cy.cpu().numpy(),
                }
                wet_probability = (
                    torch.sigmoid(wet_logits).cpu().numpy()
                    if probability_threshold is not None else None
                )
                true_batch = {
                    "depth": batch["depth"].cpu().numpy(),
                    "component_x": batch["component_x"].cpu().numpy(),
                    "component_y": batch["component_y"].cpu().numpy(),
                }
                for local in range(count):
                    block_position = offset + local
                    block_index = int(dataset.block_rows.iloc[block_position]["block_index"])
                    window = dataset.block_windows[block_index]
                    for variable in VARIABLES:
                        fill_block(true_maps[variable], true_batch[variable][local], block_index, window, block_index_grid)
                        fill_block(predicted_maps[variable], predicted_batch[variable][local], block_index, window, block_index_grid)
                    if probability_threshold is not None:
                        fill_block(
                            predicted_wet_map, wet_probability[local] >= probability_threshold,
                            block_index, window, block_index_grid,
                        )
                offset += count
            if offset != n_blocks:
                raise RuntimeError(f"Expected {n_blocks} blocks but reconstructed {offset}")
            for maps in (true_maps, predicted_maps):
                for variable in VARIABLES:
                    maps[variable][~valid_domain] = np.nan
            if probability_threshold is not None:
                for variable in VARIABLES:
                    predicted_maps[variable][~predicted_wet_map] = 0.0
            elapsed_hours = float(event_row["time_start"]) + time_index * float(event_row["time_step"])
            output_path = maps_dir / f"D030_t{time_index:03d}_whole_area.png"
            display_stride = plot_timestamp(
                true_maps, predicted_maps, time_index, elapsed_hours, output_path,
                args.max_display_size, args.dpi,
            )
            metrics = calculate_metrics(
                true_maps, predicted_maps, wet_threshold,
                predicted_wet_map if probability_threshold is not None else None,
                config.get("component_semantics", "unknown"),
            )
            record = {
                "event_id": args.event_id,
                "time_index": time_index,
                "elapsed_hours": elapsed_hours,
                "path": str(output_path),
                "display_stride": display_stride,
                **metrics,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del true_maps, predicted_maps
    with (output_dir / "whole_area_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "event_id": args.event_id,
        "domain_shape": list(block_index_grid.shape),
        "n_blocks": n_blocks,
        "time_interval": args.time_interval,
        "time_indices": time_indices,
        "wet_threshold": wet_threshold,
        "wet_gating": probability_threshold is not None,
        "wet_probability_threshold": probability_threshold,
        "wet_threshold_source": threshold_source,
        "figures": records,
    }
    (output_dir / "whole_area_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(records)} whole-area figures to {maps_dir}")


if __name__ == "__main__":
    main()
