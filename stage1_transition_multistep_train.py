#!/usr/bin/env python3
"""Fine-tune the transition model with scheduled multi-step state exposure."""

import argparse
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage1_data import prepare_stage1_data
from stage1_train import MetricAccumulator
from stage1_transition_data import Stage1TransitionSequenceDataset
from stage1_transition_model import Stage1StateTransitionModel, load_timestamp_backbone
from stage1_transition_train import (
    resolve_device,
    transition_loss_terms,
    transition_score,
)


LOSS_NAMES = (
    "loss",
    "depth",
    "depth_log",
    "depth_physical",
    "transition_depth",
    "dry_depth",
    "wet_bce",
    "wet_dice",
    "component",
    "dry_component",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-run-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", default="best_model.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batches-per-epoch", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--eval-time-stride", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--predicted-state-probability-start", type=float, default=0.25)
    parser.add_argument("--predicted-state-probability-end", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def scheduled_state(predicted, truth, probability, random_values=None):
    """Choose predicted or true previous state independently for each sample."""

    if probability <= 0:
        return truth
    if probability >= 1:
        return predicted
    if random_values is None:
        random_values = torch.rand(
            predicted.shape[0], 1, 1, device=predicted.device
        )
    use_prediction = random_values < float(probability)
    return torch.where(use_prediction, predicted, truth)


def move_sequence_batch(batch, device):
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
        "sequence_depth",
        "sequence_component_x",
        "sequence_component_y",
        "sequence_time_index",
        "sequence_time_features",
    ):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def step_target(batch, step, previous_depth, previous_x, previous_y):
    return {
        "mask": batch["mask"],
        "depth": batch["sequence_depth"][:, step + 1],
        "component_x": batch["sequence_component_x"][:, step + 1],
        "component_y": batch["sequence_component_y"][:, step + 1],
        "previous_depth": previous_depth,
        "previous_component_x": previous_x,
        "previous_component_y": previous_y,
    }


def run_sequence_epoch(
    model, loader, device, loss_args, rollout_steps, predicted_probability,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator()
    totals = {name: 0.0 for name in LOSS_NAMES}
    sequences = 0
    for batch in loader:
        batch = move_sequence_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predicted_depth = batch["sequence_depth"][:, 0]
        predicted_x = batch["sequence_component_x"][:, 0]
        predicted_y = batch["sequence_component_y"][:, 0]
        final_output = None
        final_target = None
        for step in range(rollout_steps):
            true_previous_depth = batch["sequence_depth"][:, step]
            true_previous_x = batch["sequence_component_x"][:, step]
            true_previous_y = batch["sequence_component_y"][:, step]
            if step == 0:
                previous_depth = true_previous_depth
                previous_x = true_previous_x
                previous_y = true_previous_y
            else:
                random_values = None
                if training and 0 < predicted_probability < 1:
                    random_values = torch.rand(
                        predicted_depth.shape[0], 1, 1, device=device
                    )
                previous_depth = scheduled_state(
                    predicted_depth.detach(), true_previous_depth,
                    predicted_probability, random_values,
                )
                previous_x = scheduled_state(
                    predicted_x.detach(), true_previous_x,
                    predicted_probability, random_values,
                )
                previous_y = scheduled_state(
                    predicted_y.detach(), true_previous_y,
                    predicted_probability, random_values,
                )
            target = step_target(
                batch, step, previous_depth, previous_x, previous_y
            )
            with torch.set_grad_enabled(training):
                output = model(
                    batch["event"],
                    batch["sequence_time_index"][:, step],
                    batch["sequence_time_features"][:, step],
                    batch["block_features"],
                    batch["static"],
                    batch["mask"],
                    previous_depth,
                    previous_x,
                    previous_y,
                    shared_event_time=True,
                )
                values = transition_loss_terms(output, target, loss_args, device)
                if training:
                    (values["loss"] / rollout_steps).backward()
            predicted_depth, predicted_x, predicted_y = (
                output[0], output[2], output[3]
            )
            final_output = output
            final_target = target
            count = int(batch["event"].shape[0])
            for name, value in values.items():
                totals[name] += float(value.detach()) * count / rollout_steps
        if training:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite multi-step gradient norm")
            optimizer.step()
        count = int(batch["event"].shape[0])
        sequences += count
        accumulator.update(
            final_output[0].detach(),
            torch.sigmoid(final_output[1].detach()),
            final_output[2].detach(),
            final_output[3].detach(),
            final_target,
            loss_args.wet_threshold,
            1.0,
            loss_args.component_semantics,
        )
    metrics = accumulator.finalize()
    for name, total in totals.items():
        metrics[f"loss_{name}" if name != "loss" else "loss"] = total / max(
            sequences, 1
        )
    metrics["physical_score"] = transition_score(metrics)
    return metrics


def evaluate_sequence_persistence(loader, device, loss_args):
    accumulator = MetricAccumulator()
    for batch in loader:
        batch = move_sequence_batch(batch, device)
        history_states = 1
        if "sequence_history_states" in batch:
            history_states = int(batch["sequence_history_states"][0])
        initial_index = history_states - 1
        initial_depth = batch["sequence_depth"][:, initial_index]
        target = {
            "mask": batch["mask"],
            "depth": batch["sequence_depth"][:, -1],
            "component_x": batch["sequence_component_x"][:, -1],
            "component_y": batch["sequence_component_y"][:, -1],
        }
        accumulator.update(
            initial_depth,
            (initial_depth >= loss_args.wet_threshold).float(),
            batch["sequence_component_x"][:, initial_index],
            batch["sequence_component_y"][:, initial_index],
            target,
            loss_args.wet_threshold,
            1.0,
            loss_args.component_semantics,
        )
    metrics = accumulator.finalize()
    metrics["physical_score"] = transition_score(metrics)
    return metrics


def main():
    args = parse_args()
    if args.rollout_steps < 2:
        raise ValueError("Multi-step fine-tuning requires at least two rollout steps")
    for probability in (
        args.predicted_state_probability_start,
        args.predicted_state_probability_end,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("Scheduled-state probabilities must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    initial_run_dir = args.initial_run_dir.resolve()
    base_config = json.loads((initial_run_dir / "run_config.json").read_text())
    bundle = prepare_stage1_data(
        manifest_dir=Path(base_config["manifest_dir"]),
        events_csv=Path(base_config["events_csv"]),
        blocks_parquet=Path(base_config["blocks_parquet"]),
        labels_10m_dir=Path(base_config["labels_10m_dir"]),
        static_rasters_dir=Path(base_config["static_rasters_dir"]),
        base_dir=Path("."),
        test_events=base_config["test_events"],
        val_fraction=0.2,
        seed=args.seed,
        batch_size=args.batch_size,
        train_batches_per_epoch=args.train_batches_per_epoch,
        eval_batches=args.eval_batches,
        train_time_stride=1,
        eval_time_stride=args.eval_time_stride,
        wet_threshold=float(base_config["wet_threshold"]),
        netcdf_chunk_cache_mb=int(base_config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(base_config.get("max_open_netcdf_handles", 8)),
        sampling_index_dir=Path(base_config["sampling_index_dir"]),
        sampling_mode="balanced_batch",
        sampling_target_wet_cell_fraction=0.15,
        sampling_strict_category_quotas=True,
        sampling_category_fractions={
            "dry": 0.125, "boundary": 0.25, "wet": 0.3125, "deep": 0.3125,
        },
        sampling_phase_fractions={
            "quiet": 0.20, "rising": 0.25, "peak": 0.25, "recession": 0.30,
        },
        minimum_time_index=args.rollout_steps,
    )
    datasets = {
        name: Stage1TransitionSequenceDataset(
            getattr(bundle, f"{name}_dataset"), args.rollout_steps
        )
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(
            datasets[name],
            batch_sampler=getattr(bundle, f"{name}_sampler"),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for name in datasets
    }

    checkpoint_path = initial_run_dir / args.initial_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage1StateTransitionModel(**checkpoint["model_config"]).to(device)
    loaded, skipped = load_timestamp_backbone(model, checkpoint)
    if skipped:
        raise ValueError(f"Initial transition checkpoint has skipped tensors: {skipped}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_args = Namespace(**base_config)
    loss_args.component_semantics = bundle.component_semantics

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(base_config)
    run_config.update(
        {
            "training_mode": "scheduled_multistep",
            "initial_run_dir": str(initial_run_dir),
            "initial_checkpoint": str(checkpoint_path.resolve()),
            "output_dir": str(output_dir),
            "rollout_steps": args.rollout_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_batches_per_epoch": args.train_batches_per_epoch,
            "eval_batches": args.eval_batches,
            "eval_time_stride": args.eval_time_stride,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "predicted_state_probability_start": args.predicted_state_probability_start,
            "predicted_state_probability_end": args.predicted_state_probability_end,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "component_semantics": bundle.component_semantics,
            "component_units": "m2 s-1",
            "model_config": checkpoint["model_config"],
            "split_events": bundle.split_events,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    best_score = float("inf")
    best_epoch = None
    history = []
    output_checkpoint = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        if args.epochs == 1:
            probability = args.predicted_state_probability_end
        else:
            fraction = (epoch - 1) / (args.epochs - 1)
            probability = (
                args.predicted_state_probability_start
                + fraction
                * (
                    args.predicted_state_probability_end
                    - args.predicted_state_probability_start
                )
            )
        bundle.train_sampler.set_epoch(epoch)
        train_metrics = run_sequence_epoch(
            model, loaders["train"], device, loss_args, args.rollout_steps,
            probability, optimizer,
        )
        val_metrics = run_sequence_epoch(
            model, loaders["val"], device, loss_args, args.rollout_steps, 1.0
        )
        history.append(
            {
                "epoch": epoch,
                "predicted_state_probability": probability,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print(
            f"epoch={epoch:03d} predicted_state_probability={probability:.3f} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_depth_wet_rmse={val_metrics['depth_wet_rmse']:.4f} "
            f"val_component_rmse={val_metrics['component_rmse']:.4f} "
            f"val_f1={val_metrics['wet_f1']:.4f} "
            f"score={val_metrics['physical_score']:.4f}",
            flush=True,
        )
        if val_metrics["physical_score"] < best_score:
            best_score = val_metrics["physical_score"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": checkpoint["model_config"],
                    "epoch": epoch,
                    "best_selection_score": best_score,
                    "rollout_steps": args.rollout_steps,
                    "component_semantics": bundle.component_semantics,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_checkpoint,
            )

    selected = torch.load(output_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    test_metrics = run_sequence_epoch(
        model, loaders["test"], device, loss_args, args.rollout_steps, 1.0
    )
    persistence_metrics = evaluate_sequence_persistence(
        loaders["test"], device, loss_args
    )
    payload = {
        "training_mode": "scheduled_multistep",
        "rollout_steps": args.rollout_steps,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "test": test_metrics,
        "persistence_test": persistence_metrics,
        "history": history,
        "component_semantics": bundle.component_semantics,
        "component_units": "m2 s-1",
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[Persistence] {json.dumps(persistence_metrics, sort_keys=True)}")
    print(f"[Test] {json.dumps(test_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
