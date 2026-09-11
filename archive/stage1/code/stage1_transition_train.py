#!/usr/bin/env python3
"""Train a one-step residual hydraulic state-transition surrogate."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage1_data import prepare_stage1_data
from stage1_train import MetricAccumulator, depth_bin_weights, masked_huber, soft_dice_loss
from stage1_transition_data import Stage1TransitionDataset, TransitionBatchSampler
from stage1_transition_model import Stage1StateTransitionModel, load_timestamp_backbone


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--static-rasters-dir", type=Path, required=True)
    parser.add_argument("--sampling-index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--test-events", nargs="+", default=["D030"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batches-per-epoch", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=1000)
    parser.add_argument("--eval-time-stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--netcdf-chunk-cache-mb", type=int, default=32)
    parser.add_argument("--max-open-netcdf-handles", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--depth-log-loss-weight", type=float, default=1.0)
    parser.add_argument("--depth-physical-loss-weight", type=float, default=1.0)
    parser.add_argument("--dry-depth-loss-weight", type=float, default=0.10)
    parser.add_argument("--transition-depth-loss-weight", type=float, default=0.50)
    parser.add_argument("--wet-loss-weight", type=float, default=0.20)
    parser.add_argument("--wet-dice-loss-weight", type=float, default=0.15)
    parser.add_argument("--wet-pos-weight", type=float, default=1.25)
    parser.add_argument("--component-loss-weight", type=float, default=0.50)
    parser.add_argument("--dry-component-loss-weight", type=float, default=0.05)
    parser.add_argument("--depth-weight-shallow", type=float, default=1.0)
    parser.add_argument("--depth-weight-moderate", type=float, default=2.0)
    parser.add_argument("--depth-weight-deep", type=float, default=3.0)
    parser.add_argument("--depth-weight-extreme", type=float, default=4.0)
    parser.add_argument("--depth-moderate-threshold", type=float, default=0.25)
    parser.add_argument("--depth-deep-threshold", type=float, default=1.0)
    parser.add_argument("--depth-extreme-threshold", type=float, default=2.0)
    parser.add_argument("--temporal-channels", type=int, default=96)
    parser.add_argument("--temporal-layers", type=int, default=8)
    parser.add_argument("--event-embedding-dim", type=int, default=128)
    parser.add_argument("--conditioning-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def resolve_device(value):
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    for key in (
        "event", "time_index", "time_features", "block_features", "static", "mask",
        "depth", "component_x", "component_y", "previous_depth",
        "previous_component_x", "previous_component_y", "previous_time_index",
        "older_depth", "older_component_x", "older_component_y", "older_time_index",
    ):
        if key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def transition_score(metrics):
    return (
        metrics["depth_wet_rmse"]
        + 2.0 * metrics["component_rmse"]
        + 0.5 * (1.0 - metrics["wet_f1"])
    )


def transition_loss_terms(output, batch, args, device):
    """Compute the shared one-step objective for teacher-forced or rollout states."""

    depth, wet_logits, cx, cy, depth_delta = output[:5]
    mask = batch["mask"]
    wet = (batch["depth"] >= args.wet_threshold).float() * mask
    previous_wet = (batch["previous_depth"] >= args.wet_threshold).float() * mask
    dry = mask * (1.0 - wet)
    depth_weights = wet * depth_bin_weights(batch["depth"], args)
    depth_log = masked_huber(
        torch.log1p(depth), torch.log1p(batch["depth"]), depth_weights, 0.20
    )
    depth_physical = masked_huber(depth, batch["depth"], depth_weights, 1.0)
    depth_loss = (
        args.depth_log_loss_weight * depth_log
        + args.depth_physical_loss_weight * depth_physical
    )
    true_depth_delta = batch["depth"] - batch["previous_depth"]
    transition_active = torch.maximum(wet, previous_wet)
    transition_active = torch.maximum(
        transition_active, (true_depth_delta.abs() >= 0.01).float() * mask
    )
    transition_depth = masked_huber(
        depth_delta, true_depth_delta, transition_active, 0.10
    )
    dry_depth = (depth.square() * dry).sum() / dry.sum().clamp_min(1.0)
    wet_bce_raw = F.binary_cross_entropy_with_logits(
        wet_logits,
        wet,
        reduction="none",
        pos_weight=torch.tensor(args.wet_pos_weight, device=device),
    )
    wet_bce = (wet_bce_raw * mask).sum() / mask.sum().clamp_min(1.0)
    wet_dice = soft_dice_loss(wet_logits, wet, mask, 1.0)
    component = 0.5 * (
        masked_huber(cx, batch["component_x"], wet, 0.25)
        + masked_huber(cy, batch["component_y"], wet, 0.25)
    )
    dry_component = (
        ((cx.square() + cy.square()) * dry).sum() / dry.sum().clamp_min(1.0)
    )
    loss = (
        depth_loss
        + args.transition_depth_loss_weight * transition_depth
        + args.dry_depth_loss_weight * dry_depth
        + args.wet_loss_weight * wet_bce
        + args.wet_dice_loss_weight * wet_dice
        + args.component_loss_weight * component
        + args.dry_component_loss_weight * dry_component
    )
    values = {
        "loss": loss,
        "depth": depth_loss,
        "depth_log": depth_log,
        "depth_physical": depth_physical,
        "transition_depth": transition_depth,
        "dry_depth": dry_depth,
        "wet_bce": wet_bce,
        "wet_dice": wet_dice,
        "component": component,
        "dry_component": dry_component,
    }
    nonfinite = [name for name, value in values.items() if not torch.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(f"Non-finite transition losses: {nonfinite}")
    return values


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator()
    totals = {name: 0.0 for name in (
        "loss", "depth", "depth_log", "depth_physical", "transition_depth",
        "dry_depth", "wet_bce", "wet_dice", "component", "dry_component",
    )}
    samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            depth, wet_logits, cx, cy, depth_delta, _ = model(
                batch["event"], batch["time_index"], batch["time_features"],
                batch["block_features"], batch["static"], batch["mask"],
                batch["previous_depth"], batch["previous_component_x"],
                batch["previous_component_y"], shared_event_time=True,
            )
            values = transition_loss_terms(
                (depth, wet_logits, cx, cy, depth_delta), batch, args, device
            )
            loss = values["loss"]
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite transition gradient norm")
                optimizer.step()
        count = int(batch["event"].shape[0])
        samples += count
        for name, value in values.items():
            totals[name] += float(value.detach()) * count
        accumulator.update(
            depth.detach(), torch.sigmoid(wet_logits.detach()), cx.detach(), cy.detach(),
            batch, args.wet_threshold, 1.0,
            getattr(args, "component_semantics", "unknown"),
        )
    metrics = accumulator.finalize()
    for name, total in totals.items():
        metrics[f"loss_{name}" if name != "loss" else "loss"] = total / max(samples, 1)
    metrics["physical_score"] = transition_score(metrics)
    return metrics


def evaluate_persistence(loader, device, args):
    accumulator = MetricAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        wet_probability = (batch["previous_depth"] >= args.wet_threshold).float()
        accumulator.update(
            batch["previous_depth"], wet_probability,
            batch["previous_component_x"], batch["previous_component_y"],
            batch, args.wet_threshold, 1.0,
            getattr(args, "component_semantics", "unknown"),
        )
    metrics = accumulator.finalize()
    metrics["physical_score"] = transition_score(metrics)
    return metrics


def main():
    args = parse_args()
    if args.lag < 1:
        raise ValueError("--lag must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_stage1_data(
        manifest_dir=args.manifest_dir, events_csv=args.events_csv,
        blocks_parquet=args.blocks_parquet, labels_10m_dir=args.labels_10m_dir,
        static_rasters_dir=args.static_rasters_dir, base_dir=Path("."),
        test_events=args.test_events, val_fraction=0.2, seed=args.seed,
        batch_size=args.batch_size, train_batches_per_epoch=args.train_batches_per_epoch,
        eval_batches=args.eval_batches, train_time_stride=1,
        eval_time_stride=args.eval_time_stride, wet_threshold=args.wet_threshold,
        netcdf_chunk_cache_mb=args.netcdf_chunk_cache_mb,
        max_open_netcdf_handles=args.max_open_netcdf_handles,
        sampling_index_dir=args.sampling_index_dir, sampling_mode="balanced_batch",
        sampling_target_wet_cell_fraction=0.15, sampling_strict_category_quotas=True,
        sampling_category_fractions={"dry": 0.125, "boundary": 0.25, "wet": 0.3125, "deep": 0.3125},
        sampling_phase_fractions={"quiet": 0.20, "rising": 0.25, "peak": 0.25, "recession": 0.30},
        minimum_time_index=args.lag,
    )
    args.component_semantics = bundle.component_semantics
    datasets = {
        name: Stage1TransitionDataset(getattr(bundle, f"{name}_dataset"), args.lag)
        for name in ("train", "val", "test")
    }
    samplers = {
        name: TransitionBatchSampler(getattr(bundle, f"{name}_sampler"), args.lag)
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(
            datasets[name], batch_sampler=samplers[name], num_workers=args.num_workers,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        for name in datasets
    }

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
    model = Stage1StateTransitionModel(**model_config).to(device)
    initialization = None
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint.resolve(), map_location=device, weights_only=False)
        loaded, skipped = load_timestamp_backbone(model, checkpoint)
        initialization = {
            "checkpoint": str(args.initial_checkpoint.resolve()),
            "loaded_tensors": len(loaded), "skipped_tensors": skipped,
        }
        print(f"[Initialization] loaded={len(loaded)} skipped={len(skipped)}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({
        "model_config": model_config, "split_events": bundle.split_events,
        "component_semantics": bundle.component_semantics,
        "variable_names": bundle.variable_names, "initialization": initialization,
    })
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    best_score = float("inf")
    patience = 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        samplers["train"].set_epoch(epoch)
        train_metrics = run_epoch(model, loaders["train"], device, args, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_depth_wet_rmse={val_metrics['depth_wet_rmse']:.4f} "
            f"val_component_rmse={val_metrics['component_rmse']:.4f} "
            f"val_f1={val_metrics['wet_f1']:.4f} score={val_metrics['physical_score']:.4f}"
        )
        if val_metrics["physical_score"] < best_score:
            best_score = val_metrics["physical_score"]
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(), "model_config": model_config,
                "best_selection_score": best_score, "epoch": epoch,
                "transition_lag": args.lag, "component_semantics": bundle.component_semantics,
            }, checkpoint_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, loaders["test"], device, args)
    persistence_metrics = evaluate_persistence(loaders["test"], device, args)
    payload = {
        "best_selection_score": best_score, "best_epoch": checkpoint["epoch"],
        "test": test_metrics, "persistence_test": persistence_metrics, "history": history,
        "component_semantics": bundle.component_semantics,
        "component_units": "m2 s-1" if bundle.component_semantics == "unit_discharge" else "unknown",
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[Persistence] {json.dumps(persistence_metrics, sort_keys=True)}")
    print(f"[Test] {json.dumps(test_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
