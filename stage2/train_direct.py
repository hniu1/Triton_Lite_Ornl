#!/usr/bin/env python3
"""Train the Stage-2 V5 direct lead-time-conditioned tile baseline."""

from __future__ import annotations

import argparse
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage2.data import DirectHorizonDataset, RandomLeadTimeBatchSampler
from stage2.direct import (
    evaluate_direct_horizons,
    horizon_selection_score,
    run_direct_train_epoch,
)
from stage2.model import Stage2DirectLeadModel, load_v4b_initialization
from triton_core.data import prepare_hydraulic_data
from triton_core.runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-run-dir",
        type=Path,
        default=Path("results/stage1_transition_multistep_v4b"),
    )
    parser.add_argument("--initial-checkpoint", default="best_model.pt")
    parser.add_argument("--sampling-index-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-horizon-steps", type=int, default=48)
    parser.add_argument("--step-hours", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batches-per-epoch", type=int, default=750)
    parser.add_argument("--selection-batches", type=int, default=40)
    parser.add_argument("--selection-start-time-stride", type=int, default=12)
    parser.add_argument(
        "--selection-horizons", type=int, nargs="+", default=[1, 6, 12, 24, 48]
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--new-conditioning-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--selection-derived-velocity-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--netcdf-chunk-cache-mb", type=int, default=None)
    parser.add_argument("--max-open-netcdf-handles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, backbone_lr, new_lr, weight_decay):
    new_prefixes = ("origin_time_encoder.", "lead_encoder.", "conditioning.")
    backbone = []
    new_conditioning = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(new_prefixes):
            new_conditioning.append(parameter)
        else:
            backbone.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(backbone_lr)},
            {"params": new_conditioning, "lr": float(new_lr)},
        ],
        weight_decay=float(weight_decay),
    )


def build_bundle(base_config, args, sampling_index_dir):
    data_seed = int(base_config.get("data_split_seed", base_config["seed"]))
    cache_mb = (
        int(args.netcdf_chunk_cache_mb)
        if args.netcdf_chunk_cache_mb is not None
        else int(base_config["netcdf_chunk_cache_mb"])
    )
    handle_limit = (
        int(args.max_open_netcdf_handles)
        if args.max_open_netcdf_handles is not None
        else int(base_config.get("max_open_netcdf_handles", 8))
    )
    return prepare_hydraulic_data(
        manifest_dir=Path(base_config["manifest_dir"]),
        events_csv=Path(base_config["events_csv"]),
        blocks_parquet=Path(base_config["blocks_parquet"]),
        labels_10m_dir=Path(base_config["labels_10m_dir"]),
        static_rasters_dir=Path(base_config["static_rasters_dir"]),
        base_dir=Path("."),
        test_events=base_config["test_events"],
        val_fraction=0.2,
        seed=data_seed,
        batch_size=args.batch_size,
        train_batches_per_epoch=args.train_batches_per_epoch,
        eval_batches=args.selection_batches,
        train_time_stride=1,
        eval_time_stride=args.selection_start_time_stride,
        wet_threshold=float(base_config["wet_threshold"]),
        netcdf_chunk_cache_mb=cache_mb,
        max_open_netcdf_handles=handle_limit,
        sampling_index_dir=sampling_index_dir,
        sampling_mode="balanced_batch",
        sampling_target_wet_cell_fraction=0.15,
        sampling_strict_category_quotas=True,
        sampling_category_fractions=base_config.get("sampling_category_fractions"),
        sampling_phase_fractions=base_config.get("sampling_phase_fractions"),
        minimum_time_index=args.max_horizon_steps + 1,
    )


def main():
    args = parse_args()
    if args.max_horizon_steps < 1 or args.step_hours <= 0:
        raise ValueError("Forecast horizon and timestep must be positive")
    if any(
        horizon < 1 or horizon > args.max_horizon_steps
        for horizon in args.selection_horizons
    ):
        raise ValueError("Selection horizons must lie inside the trained range")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    initial_run_dir = args.initial_run_dir.resolve()
    base_config = json.loads((initial_run_dir / "run_config.json").read_text())
    sampling_index_dir = (
        args.sampling_index_dir.resolve()
        if args.sampling_index_dir is not None
        else Path(base_config["sampling_index_dir"]).resolve()
    )
    bundle = build_bundle(base_config, args, sampling_index_dir)
    history_states = 2
    direct_datasets = {
        name: DirectHorizonDataset(
            getattr(bundle, f"{name}_dataset"),
            max_horizon_steps=args.max_horizon_steps,
            step_hours=args.step_hours,
            history_states=history_states,
        )
        for name in ("train", "val", "test")
    }
    horizons = list(range(1, args.max_horizon_steps + 1))
    train_sampler = RandomLeadTimeBatchSampler(
        bundle.train_sampler, horizons=horizons, seed=args.seed
    )
    train_loader = DataLoader(
        direct_datasets["train"],
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    initial_checkpoint_path = initial_run_dir / args.initial_checkpoint
    initial_checkpoint = torch.load(
        initial_checkpoint_path, map_location=device, weights_only=False
    )
    source_model_config = initial_checkpoint["model_config"]
    model_config = {
        key: source_model_config[key]
        for key in (
            "event_features",
            "block_features",
            "static_channels",
            "temporal_channels",
            "temporal_layers",
            "event_embedding_dim",
            "conditioning_dim",
            "base_channels",
            "dropout",
        )
    }
    model_config.update(
        {
            "history_states": history_states,
            "history_fusion": "adapter",
            "use_activity_gate": True,
            "activity_gate_initial_bias": -1.5,
            "lead_feature_dim": 4,
        }
    )
    model = Stage2DirectLeadModel(**model_config).to(device)
    loaded, adapted, skipped = load_v4b_initialization(model, initial_checkpoint)
    print(
        f"[V4b initialization] loaded={len(loaded)} "
        f"adapted={adapted} skipped={skipped}",
        flush=True,
    )
    optimizer = build_optimizer(
        model,
        args.backbone_learning_rate,
        args.new_conditioning_learning_rate,
        args.weight_decay,
    )
    loss_args = Namespace(**base_config)
    loss_args.component_semantics = bundle.component_semantics

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "training_mode": "stage2_v5_direct_lead_tile",
        "forecast_mode": "direct_origin_to_target",
        "forcing_availability": "known_through_target_time",
        "intermediate_hydraulic_states_used": False,
        "initial_run_dir": str(initial_run_dir),
        "initial_checkpoint": str(initial_checkpoint_path.resolve()),
        "sampling_index_dir": str(sampling_index_dir),
        "output_dir": str(output_dir),
        "max_horizon_steps": args.max_horizon_steps,
        "step_hours": args.step_hours,
        "trained_horizons_steps": horizons,
        "selection_horizons_steps": sorted(set(args.selection_horizons)),
        "selection_batches": args.selection_batches,
        "selection_start_time_stride": args.selection_start_time_stride,
        "selection_derived_velocity_weight": args.selection_derived_velocity_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "backbone_learning_rate": args.backbone_learning_rate,
        "new_conditioning_learning_rate": args.new_conditioning_learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "data_split_seed": int(
            base_config.get("data_split_seed", base_config["seed"])
        ),
        "disable_cudnn": args.disable_cudnn,
        "wet_threshold": float(base_config["wet_threshold"]),
        "component_semantics": bundle.component_semantics,
        "component_units": "m2 s-1",
        "model_config": model_config,
        "split_events": bundle.split_events,
        "test_events": base_config["test_events"],
        "manifest_dir": base_config["manifest_dir"],
        "events_csv": base_config["events_csv"],
        "blocks_parquet": base_config["blocks_parquet"],
        "labels_10m_dir": base_config["labels_10m_dir"],
        "static_rasters_dir": base_config["static_rasters_dir"],
        "netcdf_chunk_cache_mb": (
            args.netcdf_chunk_cache_mb
            if args.netcdf_chunk_cache_mb is not None
            else int(base_config["netcdf_chunk_cache_mb"])
        ),
        "max_open_netcdf_handles": (
            args.max_open_netcdf_handles
            if args.max_open_netcdf_handles is not None
            else int(base_config.get("max_open_netcdf_handles", 8))
        ),
        "initialization": {
            "loaded_tensors": loaded,
            "adapted_tensors": adapted,
            "skipped_tensors": skipped,
        },
        "loss_config": {
            key: value
            for key, value in base_config.items()
            if "loss" in key
            or key
            in {
                "wet_threshold",
                "wet_pos_weight",
                "stable_depth_threshold",
                "stable_extent_threshold",
                "rapid_depth_threshold",
                "rapid_extent_threshold",
            }
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    best_score = float("inf")
    best_epoch = None
    history = []
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = run_direct_train_epoch(
            model, train_loader, device, loss_args, optimizer
        )
        validation, validation_batches = evaluate_direct_horizons(
            model=model,
            dataset=direct_datasets["val"],
            device=device,
            horizons=args.selection_horizons,
            batch_size=args.batch_size,
            batch_limit=args.selection_batches,
            start_time_stride=args.selection_start_time_stride,
            wet_threshold=float(base_config["wet_threshold"]),
            component_semantics=bundle.component_semantics,
        )
        selection_score = horizon_selection_score(
            validation, args.selection_derived_velocity_weight
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation_horizons": validation,
            "validation_batches": validation_batches,
            "selection_score": selection_score,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"train_depth={train_metrics['depth_wet_rmse']:.4f} "
            f"train_f1={train_metrics['wet_f1']:.4f} "
            f"validation_selection={selection_score:.6f}",
            flush=True,
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "selection_score": selection_score,
            "max_horizon_steps": args.max_horizon_steps,
            "step_hours": args.step_hours,
            "component_semantics": bundle.component_semantics,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if args.save_every_epoch:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(checkpoint, best_path)

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    test_metrics, test_batches = evaluate_direct_horizons(
        model=model,
        dataset=direct_datasets["test"],
        device=device,
        horizons=args.selection_horizons,
        batch_size=args.batch_size,
        batch_limit=args.selection_batches,
        start_time_stride=args.selection_start_time_stride,
        wet_threshold=float(base_config["wet_threshold"]),
        component_semantics=bundle.component_semantics,
    )
    payload = {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_horizons_steps": sorted(set(args.selection_horizons)),
        "test_batches": test_batches,
        "test_horizons": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[Best] epoch={best_epoch} score={best_score:.6f}", flush=True)
    print(f"Wrote {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
