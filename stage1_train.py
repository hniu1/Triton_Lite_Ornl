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
    parser.add_argument(
        "--sampling-index-dir",
        type=Path,
        default=None,
        help="Optional label-aware M4 candidate index used for stratified training",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["anchor", "balanced_batch"],
        default="anchor",
    )
    parser.add_argument("--sampling-target-wet-cell-fraction", type=float, default=0.0)
    parser.add_argument("--sampling-strict-category-quotas", action="store_true")
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
    parser.add_argument("--max-open-netcdf-handles", type=int, default=8)
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
    parser.add_argument(
        "--depth-loss-mode",
        choices=["huber", "hybrid_log_weighted"],
        default="huber",
    )
    parser.add_argument("--depth-log-huber-delta", type=float, default=0.20)
    parser.add_argument("--depth-physical-huber-delta", type=float, default=1.0)
    parser.add_argument("--depth-log-loss-weight", type=float, default=1.0)
    parser.add_argument("--depth-physical-loss-weight", type=float, default=0.5)
    parser.add_argument("--depth-weight-shallow", type=float, default=1.0)
    parser.add_argument("--depth-weight-moderate", type=float, default=2.0)
    parser.add_argument("--depth-weight-deep", type=float, default=3.0)
    parser.add_argument("--depth-weight-extreme", type=float, default=4.0)
    parser.add_argument("--depth-moderate-threshold", type=float, default=0.25)
    parser.add_argument("--depth-deep-threshold", type=float, default=1.0)
    parser.add_argument("--depth-extreme-threshold", type=float, default=2.0)
    parser.add_argument("--dry-depth-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--couple-depth-with-wet-probability",
        action="store_true",
        help="Multiply positive depth by wet probability before depth loss and metrics",
    )
    parser.add_argument("--component-huber-delta", type=float, default=0.25)
    parser.add_argument("--wet-loss-weight", type=float, default=0.2)
    parser.add_argument("--wet-dice-loss-weight", type=float, default=0.0)
    parser.add_argument("--wet-dice-smoothing", type=float, default=1.0)
    parser.add_argument("--component-loss-weight", type=float, default=0.5)
    parser.add_argument("--dry-component-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--component-loss-mode",
        choices=["component_huber", "speed_aware"],
        default="component_huber",
    )
    parser.add_argument("--speed-loss-weight", type=float, default=0.5)
    parser.add_argument("--direction-loss-weight", type=float, default=0.1)
    parser.add_argument("--direction-min-speed", type=float, default=0.05)
    parser.add_argument("--velocity-weight-scale", type=float, default=2.0)
    parser.add_argument("--velocity-weight-reference-speed", type=float, default=0.25)
    parser.add_argument("--velocity-weight-cap", type=float, default=3.0)
    parser.add_argument("--wet-pos-weight", type=float, default=3.0)
    parser.add_argument("--sample-dry-fraction", type=float, default=0.15)
    parser.add_argument("--sample-boundary-fraction", type=float, default=0.25)
    parser.add_argument("--sample-wet-fraction", type=float, default=0.40)
    parser.add_argument("--sample-deep-fraction", type=float, default=0.20)
    parser.add_argument("--sample-quiet-fraction", type=float, default=0.15)
    parser.add_argument("--sample-rising-fraction", type=float, default=0.30)
    parser.add_argument("--sample-peak-fraction", type=float, default=0.30)
    parser.add_argument("--sample-recession-fraction", type=float, default=0.25)
    parser.add_argument("--diagnostic-deep-threshold", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["loss", "physical_score"],
        default="physical_score",
        help="Metric minimized for checkpoint selection and early stopping",
    )
    parser.add_argument("--selection-depth-weight", type=float, default=1.0)
    parser.add_argument("--selection-component-weight", type=float, default=2.0)
    parser.add_argument("--selection-inundation-weight", type=float, default=0.5)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible checkpoint used to initialize a new training run",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args) -> None:
    if not (
        0 <= args.depth_moderate_threshold
        <= args.depth_deep_threshold
        <= args.depth_extreme_threshold
    ):
        raise ValueError("Depth thresholds must be nonnegative and increasing")
    weight_names = (
        "depth_log_loss_weight",
        "depth_physical_loss_weight",
        "depth_weight_shallow",
        "depth_weight_moderate",
        "depth_weight_deep",
        "depth_weight_extreme",
        "wet_dice_loss_weight",
        "speed_loss_weight",
        "direction_loss_weight",
        "velocity_weight_scale",
        "velocity_weight_cap",
    )
    if any(getattr(args, name) < 0 for name in weight_names):
        raise ValueError("Loss and depth-bin weights cannot be negative")
    if not 0 <= args.sampling_target_wet_cell_fraction <= 1:
        raise ValueError("Sampling target wet-cell fraction must be between 0 and 1")
    if args.sampling_mode == "balanced_batch" and args.sampling_index_dir is None:
        raise ValueError("balanced_batch sampling requires --sampling-index-dir")
    if args.netcdf_chunk_cache_mb < 1 or args.max_open_netcdf_handles < 1:
        raise ValueError("NetCDF cache size and maximum open handles must be positive")
    if args.direction_min_speed < 0 or args.velocity_weight_reference_speed <= 0:
        raise ValueError("Velocity thresholds must be nonnegative and reference speed positive")


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


def depth_bin_weights(target, args):
    weights = torch.full_like(target, args.depth_weight_shallow)
    weights = torch.where(
        target >= args.depth_moderate_threshold,
        torch.as_tensor(args.depth_weight_moderate, device=target.device),
        weights,
    )
    weights = torch.where(
        target >= args.depth_deep_threshold,
        torch.as_tensor(args.depth_weight_deep, device=target.device),
        weights,
    )
    return torch.where(
        target >= args.depth_extreme_threshold,
        torch.as_tensor(args.depth_weight_extreme, device=target.device),
        weights,
    )


def soft_dice_loss(logits, target, mask, smoothing: float):
    probability = torch.sigmoid(logits) * mask
    target = target * mask
    intersection = (probability * target).sum()
    denominator = probability.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smoothing) / (denominator + smoothing)


def speed_aware_component_losses(cx, cy, target_x, target_y, wet, args):
    true_speed = torch.hypot(target_x, target_y)
    # sqrt(x^2 + y^2) has an undefined gradient at exactly (0, 0). The small
    # epsilon keeps zero/near-zero velocity cells from injecting NaN gradients.
    predicted_speed = torch.sqrt(cx.square() + cy.square() + 1e-12)
    speed_ratio = (true_speed / args.velocity_weight_reference_speed).clamp(
        min=0.0, max=args.velocity_weight_cap
    )
    weights = wet * (1.0 + args.velocity_weight_scale * speed_ratio)
    vector_loss = 0.5 * (
        masked_huber(cx, target_x, weights, args.component_huber_delta)
        + masked_huber(cy, target_y, weights, args.component_huber_delta)
    )
    speed_loss = masked_huber(
        predicted_speed, true_speed, weights, args.component_huber_delta
    )
    direction_mask = (wet > 0) & (true_speed >= args.direction_min_speed)
    if direction_mask.any():
        dot = cx[direction_mask] * target_x[direction_mask] + cy[direction_mask] * target_y[direction_mask]
        cosine = dot / (predicted_speed[direction_mask] * true_speed[direction_mask]).clamp_min(1e-6)
        direction_loss = (1.0 - cosine.clamp(-1.0, 1.0)).mean()
    else:
        direction_loss = (cx.sum() + cy.sum()) * 0.0
    total = (
        vector_loss
        + args.speed_loss_weight * speed_loss
        + args.direction_loss_weight * direction_loss
    )
    return total, vector_loss, speed_loss, direction_loss


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
            "velocity_count": 0.0,
            "velocity_abs": 0.0,
            "velocity_sq": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "wet_cells": 0.0,
            "valid_cells": 0.0,
            "patches": 0.0,
            "dry_patches": 0.0,
            "boundary_patches": 0.0,
            "partial_wet_patches": 0.0,
            "mostly_wet_patches": 0.0,
            "deep_patches": 0.0,
        }

    def update(
        self,
        depth,
        wet_prob,
        cx,
        cy,
        batch,
        wet_threshold: float,
        deep_threshold: float,
        component_semantics: str = "unknown",
    ):
        mask = batch["mask"] > 0.5
        true_wet = (batch["depth"] >= wet_threshold) & mask
        pred_wet = (wet_prob >= 0.5) & mask
        depth_error = depth - batch["depth"]
        all_error = depth_error[mask]
        wet_error = depth_error[true_wet]
        component_error = torch.cat(
            [(cx - batch["component_x"])[true_wet], (cy - batch["component_y"])[true_wet]]
        )
        velocity_error = None
        if component_semantics == "unit_discharge" and true_wet.any():
            true_depth = batch["depth"][true_wet].clamp_min(wet_threshold)
            predicted_depth = depth[true_wet].clamp_min(wet_threshold)
            velocity_error = torch.cat(
                [
                    cx[true_wet] / predicted_depth
                    - batch["component_x"][true_wet] / true_depth,
                    cy[true_wet] / predicted_depth
                    - batch["component_y"][true_wet] / true_depth,
                ]
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
        if velocity_error is not None:
            v["velocity_count"] += velocity_error.numel()
            v["velocity_abs"] += velocity_error.abs().sum().item()
            v["velocity_sq"] += velocity_error.square().sum().item()
        v["tp"] += (pred_wet & true_wet).sum().item()
        v["fp"] += (pred_wet & ~true_wet & mask).sum().item()
        v["fn"] += (~pred_wet & true_wet).sum().item()
        v["wet_cells"] += true_wet.sum().item()
        v["valid_cells"] += mask.sum().item()
        valid_per_patch = mask.sum(dim=(1, 2)).clamp_min(1)
        wet_per_patch = true_wet.sum(dim=(1, 2))
        wet_fraction = wet_per_patch / valid_per_patch
        patch_max_depth = batch["depth"].masked_fill(~mask, 0.0).amax(dim=(1, 2))
        v["patches"] += depth.shape[0]
        v["dry_patches"] += (wet_per_patch == 0).sum().item()
        v["boundary_patches"] += (
            (wet_fraction > 0) & (wet_fraction < 0.10)
        ).sum().item()
        v["partial_wet_patches"] += (
            (wet_fraction >= 0.10) & (wet_fraction < 0.50)
        ).sum().item()
        v["mostly_wet_patches"] += (wet_fraction >= 0.50).sum().item()
        v["deep_patches"] += (patch_max_depth >= deep_threshold).sum().item()

    def finalize(self) -> Dict[str, float]:
        v = self.values
        result = {}
        for prefix in ("depth_all", "depth_wet", "component"):
            count = max(v[f"{prefix}_count"], 1.0)
            result[f"{prefix}_mae"] = v[f"{prefix}_abs"] / count
            result[f"{prefix}_rmse"] = float(np.sqrt(v[f"{prefix}_sq"] / count))
        if v["velocity_count"] > 0:
            result["derived_velocity_mae"] = (
                v["velocity_abs"] / v["velocity_count"]
            )
            result["derived_velocity_rmse"] = float(
                np.sqrt(v["velocity_sq"] / v["velocity_count"])
            )
        precision = v["tp"] / max(v["tp"] + v["fp"], 1.0)
        recall = v["tp"] / max(v["tp"] + v["fn"], 1.0)
        result["wet_precision"] = precision
        result["wet_recall"] = recall
        result["wet_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
        result["wet_csi"] = v["tp"] / max(v["tp"] + v["fp"] + v["fn"], 1.0)
        result["wet_cells"] = v["wet_cells"]
        result["wet_cell_fraction"] = v["wet_cells"] / max(v["valid_cells"], 1.0)
        for key in (
            "dry_patches",
            "boundary_patches",
            "partial_wet_patches",
            "mostly_wet_patches",
            "deep_patches",
        ):
            result[key.replace("_patches", "_patch_fraction")] = v[key] / max(
                v["patches"], 1.0
            )
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
    loss_totals = {
        "depth": 0.0,
        "depth_log": 0.0,
        "depth_physical": 0.0,
        "dry_depth": 0.0,
        "wet_bce": 0.0,
        "wet_dice": 0.0,
        "component": 0.0,
        "component_vector": 0.0,
        "speed": 0.0,
        "direction": 0.0,
        "dry_component": 0.0,
    }
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
            if args.couple_depth_with_wet_probability:
                depth = depth * torch.sigmoid(wet_logits)
            if args.depth_loss_mode == "hybrid_log_weighted":
                weighted_wet = wet * depth_bin_weights(batch["depth"], args)
                depth_log_loss = masked_huber(
                    torch.log1p(depth),
                    torch.log1p(batch["depth"]),
                    weighted_wet,
                    args.depth_log_huber_delta,
                )
                depth_physical_loss = masked_huber(
                    depth,
                    batch["depth"],
                    weighted_wet,
                    args.depth_physical_huber_delta,
                )
                depth_loss = (
                    args.depth_log_loss_weight * depth_log_loss
                    + args.depth_physical_loss_weight * depth_physical_loss
                )
            else:
                depth_physical_loss = masked_huber(
                    depth, batch["depth"], wet, args.depth_huber_delta
                )
                depth_log_loss = torch.zeros((), device=device)
                depth_loss = depth_physical_loss
            dry = mask * (1.0 - wet)
            dry_depth_loss = (depth.square() * dry).sum() / dry.sum().clamp_min(1.0)
            wet_bce = F.binary_cross_entropy_with_logits(
                wet_logits,
                wet,
                reduction="none",
                pos_weight=torch.tensor(args.wet_pos_weight, device=device),
            )
            wet_loss = (wet_bce * mask).sum() / mask.sum().clamp_min(1.0)
            wet_dice_loss = soft_dice_loss(
                wet_logits, wet, mask, args.wet_dice_smoothing
            )
            if args.component_loss_mode == "speed_aware":
                component_loss, component_vector_loss, speed_loss, direction_loss = (
                    speed_aware_component_losses(
                        cx, cy, batch["component_x"], batch["component_y"], wet, args
                    )
                )
            else:
                component_vector_loss = 0.5 * (
                    masked_huber(cx, batch["component_x"], wet, args.component_huber_delta)
                    + masked_huber(cy, batch["component_y"], wet, args.component_huber_delta)
                )
                component_loss = component_vector_loss
                speed_loss = torch.zeros((), device=device)
                direction_loss = torch.zeros((), device=device)
            dry_component_loss = (
                ((cx.square() + cy.square()) * dry).sum() / dry.sum().clamp_min(1.0)
            )
            loss = (
                depth_loss
                + args.dry_depth_loss_weight * dry_depth_loss
                + args.wet_loss_weight * wet_loss
                + args.wet_dice_loss_weight * wet_dice_loss
                + args.component_loss_weight * component_loss
                + args.dry_component_loss_weight * dry_component_loss
            )
            named_losses = {
                "total": loss,
                "depth": depth_loss,
                "dry_depth": dry_depth_loss,
                "wet_bce": wet_loss,
                "wet_dice": wet_dice_loss,
                "component": component_loss,
                "component_vector": component_vector_loss,
                "speed": speed_loss,
                "direction": direction_loss,
                "dry_component": dry_component_loss,
            }
            nonfinite = [name for name, value in named_losses.items() if not torch.isfinite(value)]
            if nonfinite:
                raise FloatingPointError(f"Non-finite losses before backward: {nonfinite}")
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite gradient norm before optimizer step")
                optimizer.step()
        batch_size = batch["event"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for key, value in {
            "depth": depth_loss,
            "depth_log": depth_log_loss,
            "depth_physical": depth_physical_loss,
            "dry_depth": dry_depth_loss,
            "wet_bce": wet_loss,
            "wet_dice": wet_dice_loss,
            "component": component_loss,
            "component_vector": component_vector_loss,
            "speed": speed_loss,
            "direction": direction_loss,
            "dry_component": dry_component_loss,
        }.items():
            loss_totals[key] += float(value.detach()) * batch_size
        metrics.update(
            depth.detach(),
            torch.sigmoid(wet_logits.detach()),
            cx.detach(),
            cy.detach(),
            batch,
            args.wet_threshold,
            args.diagnostic_deep_threshold,
            getattr(args, "component_semantics", "unknown"),
        )
    result = metrics.finalize()
    result["loss"] = total_loss / max(total_samples, 1)
    for key, value in loss_totals.items():
        result[f"loss_{key}"] = value / max(total_samples, 1)
    return result


def physical_selection_score(metrics: Dict[str, float], args) -> float:
    return (
        args.selection_depth_weight * metrics["depth_wet_rmse"]
        + args.selection_component_weight * metrics["component_rmse"]
        + args.selection_inundation_weight * (1.0 - metrics["wet_f1"])
    )


def checkpoint_score(metrics: Dict[str, float], args) -> float:
    if args.checkpoint_metric == "loss":
        return float(metrics["loss"])
    return physical_selection_score(metrics, args)


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
    validate_args(args)
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
        max_open_netcdf_handles=args.max_open_netcdf_handles,
        sampling_index_dir=args.sampling_index_dir,
        sampling_category_fractions={
            "dry": args.sample_dry_fraction,
            "boundary": args.sample_boundary_fraction,
            "wet": args.sample_wet_fraction,
            "deep": args.sample_deep_fraction,
        },
        sampling_phase_fractions={
            "quiet": args.sample_quiet_fraction,
            "rising": args.sample_rising_fraction,
            "peak": args.sample_peak_fraction,
            "recession": args.sample_recession_fraction,
        },
        sampling_mode=args.sampling_mode,
        sampling_target_wet_cell_fraction=args.sampling_target_wet_cell_fraction,
        sampling_strict_category_quotas=args.sampling_strict_category_quotas,
    )
    args.component_semantics = bundle.component_semantics
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
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint.resolve(), map_location=device, weights_only=False)
        if initial.get("model_config") != model_config:
            raise ValueError("Initial checkpoint model configuration does not match this run")
        model.load_state_dict(initial["model_state_dict"])
        print(f"[Initialization] loaded {args.initial_checkpoint.resolve()}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_val_loss = float("inf")
    best_physical_score = float("inf")
    best_selection_score = float("inf")
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
        val_metrics["physical_score"] = physical_selection_score(val_metrics, args)
        selection_score = checkpoint_score(val_metrics, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_depth_wet_rmse={val_metrics['depth_wet_rmse']:.4f} "
            f"val_component_rmse={val_metrics['component_rmse']:.4f} val_f1={val_metrics['wet_f1']:.4f} "
            f"physical_score={val_metrics['physical_score']:.4f} "
            f"train_dry_patch_fraction={train_metrics['dry_patch_fraction']:.3f} "
            f"train_wet_cell_fraction={train_metrics['wet_cell_fraction']:.3f}"
        )
        print(
            f"[Loss] epoch={epoch:03d} train_depth={train_metrics['loss_depth']:.6f} "
            f"train_depth_log={train_metrics['loss_depth_log']:.6f} "
            f"train_depth_physical={train_metrics['loss_depth_physical']:.6f} "
            f"train_wet_bce={train_metrics['loss_wet_bce']:.6f} "
            f"train_wet_dice={train_metrics['loss_wet_dice']:.6f} "
            f"train_component={train_metrics['loss_component']:.6f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_val_loss": best_val_loss,
                    "component_semantics": bundle.component_semantics,
                    "variable_names": bundle.variable_names,
                },
                output_dir / "best_val_loss_model.pt",
            )
        if val_metrics["physical_score"] < best_physical_score:
            best_physical_score = val_metrics["physical_score"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_physical_score": best_physical_score,
                    "component_semantics": bundle.component_semantics,
                    "variable_names": bundle.variable_names,
                },
                output_dir / "best_physical_model.pt",
            )
        if selection_score < best_selection_score:
            best_selection_score = selection_score
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "checkpoint_metric": args.checkpoint_metric,
                    "best_selection_score": best_selection_score,
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

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Training produced no finite selectable checkpoint; inspect the first non-finite loss error"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, args, None)
    test_metrics["physical_score"] = physical_selection_score(test_metrics, args)
    payload = {
        "best_val_loss": best_val_loss,
        "best_physical_score": best_physical_score,
        "checkpoint_metric": args.checkpoint_metric,
        "best_selection_score": best_selection_score,
        "test": test_metrics,
        "history": history,
        "component_semantics": bundle.component_semantics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[Test] {json.dumps(test_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
