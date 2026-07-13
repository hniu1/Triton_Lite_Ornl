#!/usr/bin/env python3
"""Train and evaluate the timestamp-conditioned Stage-1 surrogate."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage1_data import Stage1DataBundle, prepare_stage1_data
from stage1_model import Stage1TimestampModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage-1 timestamp surrogate")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--static-rasters-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--test-events", nargs="+", default=None)
    parser.add_argument("--block-feature-columns", nargs="+", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batches-per-epoch", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--train-time-stride", type=int, default=1)
    parser.add_argument("--eval-time-stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--netcdf-chunk-cache-mb", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN kernels; useful on older GPUs that raise CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH.",
    )
    parser.add_argument("--temporal-channels", type=int, default=96)
    parser.add_argument(
        "--temporal-layers",
        type=int,
        default=8,
        help="Eight kernel-3 dilation layers provide a 511-step causal receptive field",
    )
    parser.add_argument("--event-embedding-dim", type=int, default=128)
    parser.add_argument("--conditioning-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--depth-huber-delta", type=float, default=0.25)
    parser.add_argument("--dry-depth-loss-weight", type=float, default=0.05)
    parser.add_argument("--component-huber-delta", type=float, default=0.25)
    parser.add_argument("--wet-loss-weight", type=float, default=0.2)
    parser.add_argument("--component-loss-weight", type=float, default=0.5)
    parser.add_argument("--dry-component-loss-weight", type=float, default=0.05)
    parser.add_argument("--wet-pos-weight", type=float, default=3.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_huber(pred, target, weights, delta: float):
    loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


class MetricAccumulator:
    def __init__(self):
        self.values = {
            "depth_all_count": 0.0,
            "depth_all_abs": 0.0,
            "depth_all_sq": 0.0,
            "depth_wet_count": 0.0,
            "depth_wet_abs": 0.0,
            "depth_wet_sq": 0.0,
            "component_count": 0.0,
            "component_abs": 0.0,
            "component_sq": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "wet_cells": 0.0,
        }

    def update(self, depth, wet_prob, cx, cy, batch, wet_threshold: float):
        mask = batch["mask"] > 0.5
        true_wet = (batch["depth"] >= wet_threshold) & mask
        pred_wet = (wet_prob >= 0.5) & mask
        depth_error = depth - batch["depth"]
        all_error = depth_error[mask]
        wet_error = depth_error[true_wet]
        component_error = torch.cat(
            [(cx - batch["component_x"])[true_wet], (cy - batch["component_y"])[true_wet]]
        )
        v = self.values
        v["depth_all_count"] += all_error.numel()
        v["depth_all_abs"] += all_error.abs().sum().item()
        v["depth_all_sq"] += all_error.square().sum().item()
        v["depth_wet_count"] += wet_error.numel()
        v["depth_wet_abs"] += wet_error.abs().sum().item()
        v["depth_wet_sq"] += wet_error.square().sum().item()
        v["component_count"] += component_error.numel()
        v["component_abs"] += component_error.abs().sum().item()
        v["component_sq"] += component_error.square().sum().item()
        v["tp"] += (pred_wet & true_wet).sum().item()
        v["fp"] += (pred_wet & ~true_wet & mask).sum().item()
        v["fn"] += (~pred_wet & true_wet).sum().item()
        v["wet_cells"] += true_wet.sum().item()

    def finalize(self) -> Dict[str, float]:
        v = self.values
        result = {}
        for prefix in ("depth_all", "depth_wet", "component"):
            count = max(v[f"{prefix}_count"], 1.0)
            result[f"{prefix}_mae"] = v[f"{prefix}_abs"] / count
            result[f"{prefix}_rmse"] = float(np.sqrt(v[f"{prefix}_sq"] / count))
        precision = v["tp"] / max(v["tp"] + v["fp"], 1.0)
        recall = v["tp"] / max(v["tp"] + v["fn"], 1.0)
        result["wet_precision"] = precision
        result["wet_recall"] = recall
        result["wet_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
        result["wet_csi"] = v["tp"] / max(v["tp"] + v["fp"] + v["fn"], 1.0)
        result["wet_cells"] = v["wet_cells"]
        return result


def move_batch(batch, device):
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


def run_epoch(
    model,
    loader,
    device,
    args,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    metrics = MetricAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            depth, wet_logits, cx, cy = model(
                batch["event"],
                batch["time_index"],
                batch["time_features"],
                batch["block_features"],
                batch["static"],
                batch["mask"],
                shared_event_time=True,
            )
            mask = batch["mask"]
            wet = (batch["depth"] >= args.wet_threshold).float() * mask
            depth_loss = masked_huber(
                depth, batch["depth"], wet, args.depth_huber_delta
            )
            dry = mask * (1.0 - wet)
            dry_depth_loss = (depth.square() * dry).sum() / dry.sum().clamp_min(1.0)
            wet_bce = F.binary_cross_entropy_with_logits(
                wet_logits,
                wet,
                reduction="none",
                pos_weight=torch.tensor(args.wet_pos_weight, device=device),
            )
            wet_loss = (wet_bce * mask).sum() / mask.sum().clamp_min(1.0)
            component_weights = wet
            component_loss = 0.5 * (
                masked_huber(
                    cx,
                    batch["component_x"],
                    component_weights,
                    args.component_huber_delta,
                )
                + masked_huber(
                    cy,
                    batch["component_y"],
                    component_weights,
                    args.component_huber_delta,
                )
            )
            dry_component_loss = (
                ((cx.square() + cy.square()) * dry).sum() / dry.sum().clamp_min(1.0)
            )
            loss = (
                depth_loss
                + args.dry_depth_loss_weight * dry_depth_loss
                + args.wet_loss_weight * wet_loss
                + args.component_loss_weight * component_loss
                + args.dry_component_loss_weight * dry_component_loss
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = batch["event"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        metrics.update(
            depth.detach(),
            torch.sigmoid(wet_logits.detach()),
            cx.detach(),
            cy.detach(),
            batch,
            args.wet_threshold,
        )
    result = metrics.finalize()
    result["loss"] = total_loss / max(total_samples, 1)
    return result


def make_loader(dataset, sampler, workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def save_bundle_metadata(bundle: Stage1DataBundle, args, output_dir: Path) -> None:
    np.savez(
        output_dir / "normalization_stats.npz",
        event_mean=bundle.normalization.event_mean,
        event_std=bundle.normalization.event_std,
        block_mean=bundle.normalization.block_mean,
        block_std=bundle.normalization.block_std,
        static_mean=bundle.normalization.static_mean,
        static_std=bundle.normalization.static_std,
        block_feature_columns=np.asarray(bundle.feature_columns),
    )
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "event_shape": list(bundle.event_shape),
            "target_shape": list(bundle.target_shape),
            "static_channels": bundle.static_channels,
            "component_semantics": bundle.component_semantics,
            "variable_names": bundle.variable_names,
            "split_events": bundle.split_events,
            "block_feature_columns": bundle.feature_columns,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    bundle = prepare_stage1_data(
        manifest_dir=args.manifest_dir,
        events_csv=args.events_csv,
        blocks_parquet=args.blocks_parquet,
        labels_10m_dir=args.labels_10m_dir,
        static_rasters_dir=args.static_rasters_dir,
        base_dir=args.base_dir,
        test_events=args.test_events,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        train_batches_per_epoch=args.train_batches_per_epoch,
        eval_batches=args.eval_batches,
        train_time_stride=args.train_time_stride,
        eval_time_stride=args.eval_time_stride,
        wet_threshold=args.wet_threshold,
        feature_columns=args.block_feature_columns,
        netcdf_chunk_cache_mb=args.netcdf_chunk_cache_mb,
    )
    save_bundle_metadata(bundle, args, output_dir)
    train_loader = make_loader(bundle.train_dataset, bundle.train_sampler, args.num_workers, device)
    val_loader = make_loader(bundle.val_dataset, bundle.val_sampler, args.num_workers, device)
    test_loader = make_loader(bundle.test_dataset, bundle.test_sampler, args.num_workers, device)

    model_config = {
        "event_features": bundle.event_shape[1],
        "block_features": len(bundle.feature_columns),
        "static_channels": bundle.static_channels,
        "temporal_channels": args.temporal_channels,
        "temporal_layers": args.temporal_layers,
        "event_embedding_dim": args.event_embedding_dim,
        "conditioning_dim": args.conditioning_dim,
        "base_channels": args.base_channels,
        "dropout": args.dropout,
    }
    model = Stage1TimestampModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_val_loss = float("inf")
    patience = 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"
    print(
        f"[Data] events={bundle.split_events} event_shape={bundle.event_shape} "
        f"target={bundle.target_shape} component_semantics={bundle.component_semantics}"
    )
    print(f"[Device] {device}")
    print(f"[cuDNN] enabled={torch.backends.cudnn.enabled}")

    for epoch in range(1, args.epochs + 1):
        bundle.train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args, None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_depth_wet_rmse={val_metrics['depth_wet_rmse']:.4f} "
            f"val_component_rmse={val_metrics['component_rmse']:.4f} val_f1={val_metrics['wet_f1']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_val_loss": best_val_loss,
                    "component_semantics": bundle.component_semantics,
                    "variable_names": bundle.variable_names,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, args, None)
    payload = {
        "best_val_loss": best_val_loss,
        "test": test_metrics,
        "history": history,
        "component_semantics": bundle.component_semantics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[Test] {json.dumps(test_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
