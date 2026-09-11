#!/usr/bin/env python3
"""Calibrate the Stage-1 wet-head probability threshold on validation data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage1_data import prepare_stage1_data
from stage1_model import Stage1TimestampModel
from stage1_train import make_loader, move_batch, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--eval-time-stride", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--metric", choices=["f1", "csi"], default="csi")
    parser.add_argument("--min-precision", type=float, default=0.0)
    return parser.parse_args()


def threshold_grid(start, stop, step):
    if not 0.0 <= start <= stop <= 1.0 or step <= 0.0:
        raise ValueError("Threshold range must satisfy 0 <= min <= max <= 1 and step > 0")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = start + np.arange(count, dtype=np.float64) * step
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return np.minimum(values, stop)


def scores_from_counts(thresholds, tp, fp, fn):
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    csi = tp / np.maximum(tp + fp + fn, 1)
    return [
        {
            "threshold": float(threshold),
            "true_positive": int(a),
            "false_positive": int(b),
            "false_negative": int(c),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "csi": float(j),
        }
        for threshold, a, b, c, p, r, f, j in zip(
            thresholds, tp, fp, fn, precision, recall, f1, csi
        )
    ]


def select_best(rows, metric, min_precision):
    eligible = [row for row in rows if row["precision"] >= min_precision]
    if not eligible:
        raise ValueError(f"No threshold reached minimum precision {min_precision:.3f}")
    return max(eligible, key=lambda row: (row[metric], row["precision"], row["threshold"]))


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    device = resolve_device(args.device)
    if bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
    eval_batches = int(args.eval_batches or config["eval_batches"])
    eval_stride = int(args.eval_time_stride or config["eval_time_stride"])
    workers = int(config["num_workers"] if args.num_workers is None else args.num_workers)
    bundle = prepare_stage1_data(
        manifest_dir=Path(config["manifest_dir"]), events_csv=Path(config["events_csv"]),
        blocks_parquet=Path(config["blocks_parquet"]), labels_10m_dir=Path(config["labels_10m_dir"]),
        static_rasters_dir=Path(config["static_rasters_dir"]), base_dir=Path(config["base_dir"]),
        test_events=config["test_events"], val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]), batch_size=int(config["batch_size"]),
        train_batches_per_epoch=1, eval_batches=eval_batches,
        train_time_stride=int(config["train_time_stride"]), eval_time_stride=eval_stride,
        wet_threshold=float(config["wet_threshold"]), feature_columns=config["block_feature_columns"],
        netcdf_chunk_cache_mb=int(config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(config.get("max_open_netcdf_handles", 8)),
    )
    loader = make_loader(bundle.val_dataset, bundle.val_sampler, workers, device)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1TimestampModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    tp = np.zeros(len(thresholds), dtype=np.int64)
    fp = np.zeros(len(thresholds), dtype=np.int64)
    fn = np.zeros(len(thresholds), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            _, logits, _, _ = model(
                batch["event"], batch["time_index"], batch["time_features"],
                batch["block_features"], batch["static"], batch["mask"],
                shared_event_time=True,
            )
            valid = batch["mask"].bool()
            truth = (batch["depth"] >= float(config["wet_threshold"])) & valid
            probability = torch.sigmoid(logits)
            for index, threshold in enumerate(thresholds):
                predicted = (probability >= float(threshold)) & valid
                tp[index] += int((predicted & truth).sum().item())
                fp[index] += int((predicted & ~truth & valid).sum().item())
                fn[index] += int((~predicted & truth).sum().item())
    rows = scores_from_counts(thresholds, tp, fp, fn)
    if int(tp[0] + fn[0]) == 0:
        raise RuntimeError(
            "Calibration samples contain no wet cells; increase --eval-batches or use a denser "
            "validation time stride."
        )
    best = select_best(rows, args.metric, args.min_precision)
    payload = {
        "checkpoint": str(checkpoint_path), "split": "validation",
        "validation_events": bundle.split_events["val"], "eval_batches": eval_batches,
        "eval_time_stride": eval_stride, "truth_depth_threshold": float(config["wet_threshold"]),
        "selection_metric": args.metric, "minimum_precision": args.min_precision,
        "selected_probability_threshold": best["threshold"], "selected_metrics": best,
        "threshold_metrics": rows,
    }
    output = (args.output_path or run_dir / "wet_threshold_calibration.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
