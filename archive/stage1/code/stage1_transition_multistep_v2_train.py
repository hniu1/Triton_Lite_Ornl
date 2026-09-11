#!/usr/bin/env python3
"""Fine-tune transition rollouts with regime balancing and delta-aware losses."""

import argparse
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage1_data import LabelAwareBatchSampler, prepare_stage1_data
from stage1_train import MetricAccumulator
from stage1_transition_data import (
    Stage1TransitionSequenceDataset,
    minimum_sequence_target_time,
)
from stage1_transition_model import (
    Stage1StateTransitionModel,
    load_transition_checkpoint_compatible,
)
from stage1_transition_multistep_train import (
    evaluate_sequence_persistence,
    move_sequence_batch,
    scheduled_state,
)
from stage1_transition_regime_eval import classify_transition_regimes
from stage1_transition_sampling import (
    LocalTransitionAwareBatchSampler,
    TransitionAwareBatchSampler,
)
from stage1_transition_train import resolve_device, transition_loss_terms, transition_score


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
    "rapid_depth_delta",
    "stable_depth_delta",
    "component_delta",
    "derived_velocity",
    "storage_change",
    "activity_gate",
)


def validate_transition_regime_fractions(fractions):
    """Validate and normalize the requested transition-regime mixture."""
    return LabelAwareBatchSampler._validate_fractions(
        fractions, "transition regime"
    )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-run-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", default="best_model.pt")
    parser.add_argument("--sampling-index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-mode", default="transition_aware_scheduled_multistep_v2"
    )
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batches-per-epoch", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--eval-time-stride", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument(
        "--adaptation-learning-rate",
        type=float,
        default=None,
        help="Optional higher rate for newly initialized history/gate adapters",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--predicted-state-probability-start", type=float, default=0.35)
    parser.add_argument("--predicted-state-probability-end", type=float, default=0.80)
    parser.add_argument("--rapid-depth-delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--stable-depth-delta-loss-weight", type=float, default=0.5)
    parser.add_argument("--component-delta-loss-weight", type=float, default=0.25)
    parser.add_argument("--derived-velocity-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--derived-velocity-loss-type", choices=("huber", "mse"), default="huber"
    )
    parser.add_argument("--derived-velocity-huber-delta", type=float, default=0.25)
    parser.add_argument("--storage-change-loss-weight", type=float, default=0.25)
    parser.add_argument("--activity-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--history-states", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--history-fusion", choices=("concat", "adapter"), default="concat"
    )
    parser.add_argument("--use-activity-gate", action="store_true")
    parser.add_argument("--activity-gate-initial-bias", type=float, default=2.0)
    parser.add_argument(
        "--selection-derived-velocity-weight", type=float, default=0.0
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--stable-depth-threshold", type=float, default=0.01)
    parser.add_argument("--stable-extent-threshold", type=float, default=0.01)
    parser.add_argument("--rapid-depth-threshold", type=float, default=0.10)
    parser.add_argument("--rapid-extent-threshold", type=float, default=0.05)
    parser.add_argument("--sample-stable-fraction", type=float, default=0.25)
    parser.add_argument("--sample-filling-fraction", type=float, default=0.10)
    parser.add_argument("--sample-draining-fraction", type=float, default=0.10)
    parser.add_argument("--sample-rapid-filling-fraction", type=float, default=0.275)
    parser.add_argument("--sample-rapid-draining-fraction", type=float, default=0.275)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def weighted_masked_huber(prediction, target, mask, sample_selection, delta):
    """Huber loss over selected samples and masked cells."""

    if sample_selection.dtype != torch.bool:
        sample_selection = sample_selection.bool()
    selection_shape = [prediction.shape[0]] + [1] * (prediction.ndim - 1)
    weights = mask * sample_selection.reshape(selection_shape).to(mask.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return prediction.sum() * 0.0
    raw = F.huber_loss(prediction, target, reduction="none", delta=delta)
    return (raw * weights).sum() / denominator


def build_optimizer(
    model, learning_rate, adaptation_learning_rate, weight_decay
):
    """Use a faster rate for zero-initialized V4 adaptation parameters."""
    if adaptation_learning_rate is None:
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    adaptation_prefixes = ("history_adapter.", "activity_head.")
    backbone = []
    adaptation = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        destination = (
            adaptation
            if name.startswith(adaptation_prefixes)
            else backbone
        )
        destination.append(parameter)
    groups = [{"params": backbone, "lr": learning_rate}]
    if adaptation:
        groups.append({"params": adaptation, "lr": adaptation_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def weighted_masked_mse(prediction, target, mask, sample_selection):
    """MSE over selected samples and masked cells, aligned with RMSE gates."""

    if sample_selection.dtype != torch.bool:
        sample_selection = sample_selection.bool()
    selection_shape = [prediction.shape[0]] + [1] * (prediction.ndim - 1)
    weights = mask * sample_selection.reshape(selection_shape).to(mask.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return prediction.sum() * 0.0
    return (((prediction - target) ** 2) * weights).sum() / denominator


def delta_aware_loss_terms(output, batch, args, device):
    """Add stable/rapid depth-change and conserved-component-change objectives."""

    values = transition_loss_terms(output, batch, args, device)
    depth, _, component_x, component_y, depth_delta, component_delta = output[:6]
    true_previous_depth = batch["true_previous_depth"]
    regimes, _ = classify_transition_regimes(
        true_previous_depth,
        batch["depth"],
        batch["mask"],
        args.wet_threshold,
        args.stable_depth_threshold,
        args.stable_extent_threshold,
        args.rapid_depth_threshold,
        args.rapid_extent_threshold,
    )
    true_depth_delta = batch["depth"] - batch["previous_depth"]
    target_wet = (batch["depth"] >= args.wet_threshold).float() * batch["mask"]
    previous_wet = (
        true_previous_depth >= args.wet_threshold
    ).float() * batch["mask"]
    active = torch.maximum(target_wet, previous_wet)
    active = torch.maximum(
        active, (true_depth_delta.abs() >= 0.01).float() * batch["mask"]
    )
    rapid_depth_delta = weighted_masked_huber(
        depth_delta,
        true_depth_delta,
        active,
        regimes["rapid"],
        0.10,
    )
    stable_depth_delta = weighted_masked_huber(
        depth_delta,
        true_depth_delta,
        batch["mask"],
        regimes["stable"],
        0.05,
    )

    true_component_delta = torch.stack(
        [
            batch["component_x"] - batch["previous_component_x"],
            batch["component_y"] - batch["previous_component_y"],
        ],
        dim=1,
    )
    component_mask = active.unsqueeze(1).expand_as(component_delta)
    component_delta_loss = weighted_masked_huber(
        component_delta,
        true_component_delta,
        component_mask,
        regimes["all"],
        0.10,
    )
    predicted_denominator = depth.clamp_min(args.wet_threshold)
    target_denominator = batch["depth"].clamp_min(args.wet_threshold)
    predicted_velocity = torch.stack(
        [component_x / predicted_denominator, component_y / predicted_denominator],
        dim=1,
    )
    target_velocity = torch.stack(
        [
            batch["component_x"] / target_denominator,
            batch["component_y"] / target_denominator,
        ],
        dim=1,
    )
    velocity_mask = target_wet.unsqueeze(1).expand_as(predicted_velocity)
    if getattr(args, "derived_velocity_loss_type", "huber") == "mse":
        derived_velocity_loss = weighted_masked_mse(
            predicted_velocity,
            target_velocity,
            velocity_mask,
            regimes["all"],
        )
    else:
        derived_velocity_loss = weighted_masked_huber(
            predicted_velocity,
            target_velocity,
            velocity_mask,
            regimes["all"],
            getattr(args, "derived_velocity_huber_delta", 0.25),
        )
    valid_count = batch["mask"].sum(dim=(-2, -1)).clamp_min(1.0)
    predicted_storage_change = (
        (depth - batch["previous_depth"]) * batch["mask"]
    ).sum(dim=(-2, -1)) / valid_count
    target_storage_change = (
        true_depth_delta * batch["mask"]
    ).sum(dim=(-2, -1)) / valid_count
    storage_weights = torch.where(
        regimes["rapid"],
        torch.full_like(predicted_storage_change, 2.0),
        torch.ones_like(predicted_storage_change),
    )
    storage_change_loss = (
        F.huber_loss(
            predicted_storage_change,
            target_storage_change,
            reduction="none",
            delta=0.05,
        )
        * storage_weights
    ).sum() / storage_weights.sum().clamp_min(1.0)
    activity_gate_loss = depth.sum() * 0.0
    if len(output) > 6:
        activity_logits = output[6]
        patch_dynamic = (~regimes["stable"]).to(batch["mask"].dtype)[:, None, None]
        changed_wet_extent = (target_wet != previous_wet).to(batch["mask"].dtype)
        changed_depth = (true_depth_delta.abs() >= 0.01).to(batch["mask"].dtype)
        activity_target = patch_dynamic * torch.maximum(
            torch.maximum(target_wet, previous_wet),
            torch.maximum(changed_wet_extent, changed_depth),
        )
        activity_target = activity_target * batch["mask"]
        positive = activity_target.sum().clamp_min(1.0)
        valid = batch["mask"].sum().clamp_min(1.0)
        negative = (valid - positive).clamp_min(1.0)
        positive_weight = (negative / positive).clamp(1.0, 20.0)
        raw_activity_loss = F.binary_cross_entropy_with_logits(
            activity_logits,
            activity_target,
            reduction="none",
            pos_weight=positive_weight,
        )
        activity_gate_loss = (
            raw_activity_loss * batch["mask"]
        ).sum() / valid
    values["loss"] = (
        values["loss"]
        + args.rapid_depth_delta_loss_weight * rapid_depth_delta
        + args.stable_depth_delta_loss_weight * stable_depth_delta
        + args.component_delta_loss_weight * component_delta_loss
        + args.derived_velocity_loss_weight * derived_velocity_loss
        + args.storage_change_loss_weight * storage_change_loss
        + getattr(args, "activity_gate_loss_weight", 0.0) * activity_gate_loss
    )
    values.update(
        {
            "rapid_depth_delta": rapid_depth_delta,
            "stable_depth_delta": stable_depth_delta,
            "component_delta": component_delta_loss,
            "derived_velocity": derived_velocity_loss,
            "storage_change": storage_change_loss,
            "activity_gate": activity_gate_loss,
        }
    )
    nonfinite = [name for name, value in values.items() if not torch.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(f"Non-finite delta-aware losses: {nonfinite}")
    return values


def step_target_v2(
    batch, step, state_offset, previous_depth, previous_x, previous_y
):
    return {
        "mask": batch["mask"],
        "depth": batch["sequence_depth"][:, step + state_offset + 1],
        "component_x": batch["sequence_component_x"][:, step + state_offset + 1],
        "component_y": batch["sequence_component_y"][:, step + state_offset + 1],
        "previous_depth": previous_depth,
        "previous_component_x": previous_x,
        "previous_component_y": previous_y,
        "true_previous_depth": batch["sequence_depth"][:, step + state_offset],
        "true_previous_component_x": batch["sequence_component_x"][:, step + state_offset],
        "true_previous_component_y": batch["sequence_component_y"][:, step + state_offset],
    }


def run_sequence_epoch_v2(
    model,
    loader,
    device,
    loss_args,
    rollout_steps,
    predicted_probability,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator()
    totals = {name: 0.0 for name in LOSS_NAMES}
    sequences = 0
    history_states = int(getattr(model, "history_states", 1))
    state_offset = history_states - 1
    for batch in loader:
        batch = move_sequence_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predicted_depth = batch["sequence_depth"][:, state_offset]
        predicted_x = batch["sequence_component_x"][:, state_offset]
        predicted_y = batch["sequence_component_y"][:, state_offset]
        last_input_depth = None
        last_input_x = None
        last_input_y = None
        final_output = None
        final_target = None
        for step in range(rollout_steps):
            true_previous_depth = batch["sequence_depth"][:, step + state_offset]
            true_previous_x = batch["sequence_component_x"][:, step + state_offset]
            true_previous_y = batch["sequence_component_y"][:, step + state_offset]
            true_older_depth = None
            true_older_x = None
            true_older_y = None
            if history_states == 2:
                true_older_depth = batch["sequence_depth"][:, step]
                true_older_x = batch["sequence_component_x"][:, step]
                true_older_y = batch["sequence_component_y"][:, step]
            if step == 0:
                previous_depth = true_previous_depth
                previous_x = true_previous_x
                previous_y = true_previous_y
                older_depth = true_older_depth
                older_x = true_older_x
                older_y = true_older_y
            else:
                random_values = None
                if training and 0 < predicted_probability < 1:
                    random_values = torch.rand(
                        predicted_depth.shape[0], 1, 1, device=device
                    )
                previous_depth = scheduled_state(
                    predicted_depth.detach(),
                    true_previous_depth,
                    predicted_probability,
                    random_values,
                )
                previous_x = scheduled_state(
                    predicted_x.detach(),
                    true_previous_x,
                    predicted_probability,
                    random_values,
                )
                previous_y = scheduled_state(
                    predicted_y.detach(),
                    true_previous_y,
                    predicted_probability,
                    random_values,
                )
                older_depth = None
                older_x = None
                older_y = None
                if history_states == 2:
                    older_depth = scheduled_state(
                        last_input_depth.detach(),
                        true_older_depth,
                        predicted_probability,
                        random_values,
                    )
                    older_x = scheduled_state(
                        last_input_x.detach(),
                        true_older_x,
                        predicted_probability,
                        random_values,
                    )
                    older_y = scheduled_state(
                        last_input_y.detach(),
                        true_older_y,
                        predicted_probability,
                        random_values,
                    )
            target = step_target_v2(
                batch,
                step,
                state_offset,
                previous_depth,
                previous_x,
                previous_y,
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
                    older_depth=older_depth,
                    older_component_x=older_x,
                    older_component_y=older_y,
                )
                values = delta_aware_loss_terms(output, target, loss_args, device)
                if training:
                    (values["loss"] / rollout_steps).backward()
            predicted_depth, predicted_x, predicted_y = output[0], output[2], output[3]
            last_input_depth = previous_depth
            last_input_x = previous_x
            last_input_y = previous_y
            final_output = output
            final_target = target
            count = int(batch["event"].shape[0])
            for name, value in values.items():
                totals[name] += float(value.detach()) * count / rollout_steps
        if training:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite V2 gradient norm")
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


def operational_selection_score(metrics, derived_velocity_weight=0.0):
    """Checkpoint score aligned with depth, extent, discharge, and velocity."""

    return metrics["physical_score"] + float(derived_velocity_weight) * metrics.get(
        "derived_velocity_rmse", 0.0
    )


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
    transition_regime_fractions = {
        "stable": args.sample_stable_fraction,
        "filling": args.sample_filling_fraction,
        "draining": args.sample_draining_fraction,
        "rapid_filling": args.sample_rapid_filling_fraction,
        "rapid_draining": args.sample_rapid_draining_fraction,
    }
    transition_regime_fractions = validate_transition_regime_fractions(
        transition_regime_fractions
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    initial_run_dir = args.initial_run_dir.resolve()
    base_config = json.loads((initial_run_dir / "run_config.json").read_text())
    data_split_seed = int(base_config.get("data_split_seed", base_config["seed"]))
    minimum_target_time = minimum_sequence_target_time(
        args.rollout_steps, args.history_states
    )
    bundle = prepare_stage1_data(
        manifest_dir=Path(base_config["manifest_dir"]),
        events_csv=Path(base_config["events_csv"]),
        blocks_parquet=Path(base_config["blocks_parquet"]),
        labels_10m_dir=Path(base_config["labels_10m_dir"]),
        static_rasters_dir=Path(base_config["static_rasters_dir"]),
        base_dir=Path("."),
        test_events=base_config["test_events"],
        val_fraction=0.2,
        seed=data_split_seed,
        batch_size=args.batch_size,
        train_batches_per_epoch=args.train_batches_per_epoch,
        eval_batches=args.eval_batches,
        train_time_stride=1,
        eval_time_stride=args.eval_time_stride,
        wet_threshold=float(base_config["wet_threshold"]),
        netcdf_chunk_cache_mb=int(base_config["netcdf_chunk_cache_mb"]),
        max_open_netcdf_handles=int(base_config.get("max_open_netcdf_handles", 8)),
        minimum_time_index=minimum_target_time,
    )
    sampling_index_dir = args.sampling_index_dir.resolve()
    candidates = pd.read_parquet(sampling_index_dir / "sampling_candidates.parquet")
    candidates = candidates.loc[candidates["time_index"] >= minimum_target_time]
    category_fractions = {
        "dry": 0.125,
        "boundary": 0.25,
        "wet": 0.3125,
        "deep": 0.3125,
    }
    phase_fractions = {
        "quiet": 0.20,
        "rising": 0.25,
        "peak": 0.25,
        "recession": 0.30,
    }
    sampler_class = (
        LocalTransitionAwareBatchSampler
        if "local_transition_regime" in candidates.columns
        else TransitionAwareBatchSampler
    )
    transition_sampling_mode = (
        "exact_local_transition" if sampler_class is LocalTransitionAwareBatchSampler
        else "event_time_transition"
    )
    train_sampler = sampler_class(
        candidates=candidates,
        event_ids=bundle.train_dataset.events["event_id"].astype(str).tolist(),
        n_times=bundle.train_dataset.events["n_times"].astype(int).tolist(),
        n_blocks=len(bundle.train_dataset.block_rows),
        batch_size=args.batch_size,
        batches_per_epoch=args.train_batches_per_epoch,
        seed=args.seed,
        category_fractions=category_fractions,
        phase_fractions=phase_fractions,
        transition_regime_fractions=transition_regime_fractions,
        target_wet_cell_fraction=0.15,
        strict_category_quotas=True,
    )
    datasets = {
        name: Stage1TransitionSequenceDataset(
            getattr(bundle, f"{name}_dataset"),
            args.rollout_steps,
            history_states=args.history_states,
        )
        for name in ("train", "val", "test")
    }
    samplers = {
        "train": train_sampler,
        "val": bundle.val_sampler,
        "test": bundle.test_sampler,
    }
    loaders = {
        name: DataLoader(
            datasets[name],
            batch_sampler=samplers[name],
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for name in datasets
    }

    checkpoint_path = initial_run_dir / args.initial_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = dict(checkpoint["model_config"])
    model_config.update(
        {
            "history_states": args.history_states,
            "history_fusion": args.history_fusion,
            "use_activity_gate": args.use_activity_gate,
            "activity_gate_initial_bias": args.activity_gate_initial_bias,
        }
    )
    model = Stage1StateTransitionModel(**model_config).to(device)
    loaded, adapted, skipped = load_transition_checkpoint_compatible(
        model, checkpoint
    )
    optimizer = build_optimizer(
        model,
        args.learning_rate,
        args.adaptation_learning_rate,
        args.weight_decay,
    )
    loss_args = Namespace(**base_config)
    loss_args.component_semantics = bundle.component_semantics
    for name in (
        "rapid_depth_delta_loss_weight",
        "stable_depth_delta_loss_weight",
        "component_delta_loss_weight",
        "derived_velocity_loss_weight",
        "derived_velocity_loss_type",
        "derived_velocity_huber_delta",
        "storage_change_loss_weight",
        "activity_gate_loss_weight",
        "stable_depth_threshold",
        "stable_extent_threshold",
        "rapid_depth_threshold",
        "rapid_extent_threshold",
    ):
        setattr(loss_args, name, getattr(args, name))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(base_config)
    run_config.update(
        {
            "training_mode": args.training_mode,
            "initial_run_dir": str(initial_run_dir),
            "initial_checkpoint": str(checkpoint_path.resolve()),
            "sampling_index_dir": str(sampling_index_dir),
            "output_dir": str(output_dir),
            "rollout_steps": args.rollout_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_batches_per_epoch": args.train_batches_per_epoch,
            "eval_batches": args.eval_batches,
            "eval_time_stride": args.eval_time_stride,
            "learning_rate": args.learning_rate,
            "adaptation_learning_rate": args.adaptation_learning_rate,
            "weight_decay": args.weight_decay,
            "predicted_state_probability_start": args.predicted_state_probability_start,
            "predicted_state_probability_end": args.predicted_state_probability_end,
            "rapid_depth_delta_loss_weight": args.rapid_depth_delta_loss_weight,
            "stable_depth_delta_loss_weight": args.stable_depth_delta_loss_weight,
            "component_delta_loss_weight": args.component_delta_loss_weight,
            "derived_velocity_loss_weight": args.derived_velocity_loss_weight,
            "derived_velocity_loss_type": args.derived_velocity_loss_type,
            "derived_velocity_huber_delta": args.derived_velocity_huber_delta,
            "storage_change_loss_weight": args.storage_change_loss_weight,
            "activity_gate_loss_weight": args.activity_gate_loss_weight,
            "history_states": args.history_states,
            "history_fusion": args.history_fusion,
            "use_activity_gate": args.use_activity_gate,
            "activity_gate_initial_bias": args.activity_gate_initial_bias,
            "selection_derived_velocity_weight": args.selection_derived_velocity_weight,
            "save_every_epoch": args.save_every_epoch,
            "stable_depth_threshold": args.stable_depth_threshold,
            "stable_extent_threshold": args.stable_extent_threshold,
            "rapid_depth_threshold": args.rapid_depth_threshold,
            "rapid_extent_threshold": args.rapid_extent_threshold,
            "transition_regime_fractions": transition_regime_fractions,
            "transition_sampling_mode": transition_sampling_mode,
            "sampling_category_fractions": category_fractions,
            "sampling_phase_fractions": phase_fractions,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "data_split_seed": data_split_seed,
            "component_semantics": bundle.component_semantics,
            "component_units": "m2 s-1",
            "model_config": model_config,
            "initialization": {
                "loaded": loaded,
                "adapted": adapted,
                "skipped": skipped,
            },
            "split_events": bundle.split_events,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    best_score = float("inf")
    best_epoch = None
    history = []
    output_checkpoint = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        fraction = 1.0 if args.epochs == 1 else (epoch - 1) / (args.epochs - 1)
        probability = args.predicted_state_probability_start + fraction * (
            args.predicted_state_probability_end
            - args.predicted_state_probability_start
        )
        train_sampler.set_epoch(epoch)
        train_metrics = run_sequence_epoch_v2(
            model,
            loaders["train"],
            device,
            loss_args,
            args.rollout_steps,
            probability,
            optimizer,
        )
        val_metrics = run_sequence_epoch_v2(
            model,
            loaders["val"],
            device,
            loss_args,
            args.rollout_steps,
            1.0,
        )
        history.append(
            {
                "epoch": epoch,
                "predicted_state_probability": probability,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        selection_score = operational_selection_score(
            val_metrics, args.selection_derived_velocity_weight
        )
        history[-1]["selection_score"] = selection_score
        print(
            f"epoch={epoch:03d} predicted_state_probability={probability:.3f} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_depth_wet_rmse={val_metrics['depth_wet_rmse']:.4f} "
            f"val_component_rmse={val_metrics['component_rmse']:.4f} "
            f"val_f1={val_metrics['wet_f1']:.4f} "
            f"physical_score={val_metrics['physical_score']:.4f} "
            f"selection_score={selection_score:.4f}",
            flush=True,
        )
        epoch_checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "selection_score": selection_score,
            "physical_score": val_metrics["physical_score"],
            "rollout_steps": args.rollout_steps,
            "history_states": args.history_states,
            "component_semantics": bundle.component_semantics,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if args.save_every_epoch:
            torch.save(epoch_checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            epoch_checkpoint["best_selection_score"] = best_score
            torch.save(epoch_checkpoint, output_checkpoint)

    selected = torch.load(output_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    test_metrics = run_sequence_epoch_v2(
        model, loaders["test"], device, loss_args, args.rollout_steps, 1.0
    )
    persistence_metrics = evaluate_sequence_persistence(
        loaders["test"], device, loss_args
    )
    payload = {
        "training_mode": args.training_mode,
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
