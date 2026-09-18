#!/usr/bin/env python3
"""Select the wet-head probability threshold on validation events only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from stage2.data import DirectHorizonDataset
from stage2.direct import move_direct_batch, predict_direct, select_forecast_batches
from stage2.model import Stage2DirectLeadModel
from triton_core.data import prepare_hydraulic_data
from triton_core.runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12, 24, 48])
    parser.add_argument("--evaluation-batches", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start-time-stride", type=int, default=12)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def score_counts(tp, fp, fn, tn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    csi = tp / max(tp + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "precision": precision, "recall": recall, "f1": f1, "csi": csi,
        "dry_specificity": specificity,
        "dry_false_positive_rate": fp / max(tn + fp, 1),
        "balanced_extent_accuracy": 0.5 * (recall + specificity),
    }


def threshold_grid(start, stop, step):
    if not 0.0 <= start <= stop <= 1.0 or step <= 0:
        raise ValueError("Threshold range must satisfy 0 <= start <= stop <= 1")
    values = np.arange(start, stop + 1e-12, step, dtype=np.float64)
    values = values[values <= stop + 1e-12]
    if len(values) == 0 or not np.isclose(values[-1], stop):
        values = np.append(values, float(stop))
    return values


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    device = resolve_device(args.device)
    if args.disable_cudnn or config.get("disable_cudnn", False):
        torch.backends.cudnn.enabled = False
    bundle = prepare_hydraulic_data(
        manifest_dir=Path(config["manifest_dir"]),
        events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]),
        labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]),
        base_dir=Path("."), test_events=config["test_events"], val_fraction=0.2,
        seed=int(config["data_split_seed"]), batch_size=args.batch_size,
        train_batches_per_epoch=1, eval_batches=1, train_time_stride=1,
        eval_time_stride=1, wet_threshold=float(config["wet_threshold"]),
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config["max_open_netcdf_handles"]),
    )
    dataset = DirectHorizonDataset(
        bundle.val_dataset, max_horizon_steps=int(config["max_horizon_steps"]),
        step_hours=float(config["step_hours"]),
        history_states=int(config["model_config"]["history_states"]),
        hydraulic_halo_cells=int(config.get("hydraulic_halo_cells", 0)),
        forcing_weights_path=(Path(config["forcing_weights_path"])
                              if config.get("forcing_weights_path") else None),
    )
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model = Stage2DirectLeadModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    horizons = sorted(set(args.horizons))
    selected = select_forecast_batches(
        dataset.events["n_times"].astype(int).tolist(), len(dataset.block_rows),
        args.batch_size, max(horizons), args.start_time_stride,
        args.evaluation_batches, dataset.history_states - 1,
    )
    thresholds = threshold_grid(
        args.threshold_min, args.threshold_max, args.threshold_step
    )
    modes = ("head", "depth", "or", "and")
    counts = {mode: np.zeros((len(thresholds), 4), dtype=np.int64) for mode in modes}
    with torch.no_grad():
        for event_position, origin_time, block_positions in selected:
            for horizon in horizons:
                batch = default_collate([
                    dataset[(event_position, origin_time + horizon, block, horizon)]
                    for block in block_positions
                ])
                batch = move_direct_batch(batch, device)
                output = predict_direct(model, batch)
                probability = torch.sigmoid(output[1])
                valid = batch["mask"] > 0.5
                truth = (batch["depth"] >= float(config["wet_threshold"])) & valid
                depth_wet = (output[0] >= float(config["wet_threshold"])) & valid
                for index, threshold in enumerate(thresholds):
                    head = (probability >= float(threshold)) & valid
                    predictions = {
                        "head": head, "depth": depth_wet,
                        "or": head | depth_wet, "and": head & depth_wet,
                    }
                    for mode, prediction in predictions.items():
                        counts[mode][index, 0] += (prediction & truth).sum().item()
                        counts[mode][index, 1] += (prediction & ~truth & valid).sum().item()
                        counts[mode][index, 2] += (~prediction & truth).sum().item()
                        counts[mode][index, 3] += (~prediction & ~truth & valid).sum().item()
    rows = []
    for mode in modes:
        mode_thresholds = thresholds if mode != "depth" else np.asarray([0.5])
        mode_counts = counts[mode] if mode != "depth" else counts[mode][:1]
        for threshold, (tp, fp, fn, tn) in zip(mode_thresholds, mode_counts):
            rows.append({
                "mode": mode, "threshold": float(threshold), "tp": int(tp),
                "fp": int(fp), "fn": int(fn), "tn": int(tn),
                **score_counts(int(tp), int(fp), int(fn), int(tn)),
            })
    best = max(
        rows,
        key=lambda row: (
            row["f1"], row["balanced_extent_accuracy"], row["precision"],
            row["mode"] == "depth", row["threshold"],
        ),
    )
    payload = {
        "split": "validation", "split_events": config["split_events"]["val"],
        "horizons_steps": horizons, "cohort_batches": len(selected),
        "selected_mode": best["mode"],
        "selected_probability_threshold": best["threshold"],
        "selected_metrics": best, "candidate_metrics": rows,
    }
    output = (args.output_path or run_dir / "wet_threshold_calibration.json").resolve()
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["selected_metrics"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
