import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blockwise_matrix_data import BlockwiseMatrixDataBundle, prepare_blockwise_matrix_datasets
from blockwise_model import BlockwiseFloodMatrixModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an 80x80 block-wise Triton Lite flood-depth surrogate")
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional tuned config JSON. CLI hyperparameters override values from this file.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--block-feature-columns", nargs="+", default=None)
    parser.add_argument("--test-events", nargs="+", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal-channels", type=int, default=64)
    parser.add_argument("--event-embedding-dim", type=int, default=64)
    parser.add_argument("--block-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-base-channels", type=int, default=128)
    parser.add_argument("--target-rows", type=int, default=80)
    parser.add_argument("--target-cols", type=int, default=80)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--depth-weight-alpha", type=float, default=0.0)
    parser.add_argument("--depth-weight-cap", type=float, default=3.0)
    parser.add_argument("--aux-wet-loss-weight", type=float, default=0.2)
    parser.add_argument("--aux-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-velocity-mag-loss-weight", type=float, default=0.0)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--velocity-huber-delta", type=float, default=0.25)
    parser.add_argument("--predict-velocity", action="store_true")
    parser.add_argument("--predict-velocity-magnitude", action="store_true")
    parser.add_argument("--velocity-x-manifest-column", type=str, default="path_to_peak_velx_10m")
    parser.add_argument("--velocity-y-manifest-column", type=str, default="path_to_peak_vely_10m")
    parser.add_argument("--velocity-mag-manifest-column", type=str, default="path_to_peak_velmag_10m")
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--static-rasters-dir",
        type=Path,
        default=None,
        help="Directory containing block_static_features.npy and block_static_feature_stats.json (V3)",
    )
    parser.add_argument(
        "--raster-enc-channels",
        type=int,
        default=16,
        help="Number of output channels from the StaticRasterEncoder CNN",
    )
    return parser.parse_args()


def parse_args_defaults() -> Dict[str, object]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--block-feature-columns", nargs="+", default=None)
    parser.add_argument("--test-events", nargs="+", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal-channels", type=int, default=64)
    parser.add_argument("--event-embedding-dim", type=int, default=64)
    parser.add_argument("--block-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-base-channels", type=int, default=128)
    parser.add_argument("--target-rows", type=int, default=80)
    parser.add_argument("--target-cols", type=int, default=80)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--depth-weight-alpha", type=float, default=0.0)
    parser.add_argument("--depth-weight-cap", type=float, default=3.0)
    parser.add_argument("--aux-wet-loss-weight", type=float, default=0.2)
    parser.add_argument("--aux-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-velocity-mag-loss-weight", type=float, default=0.0)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--velocity-huber-delta", type=float, default=0.25)
    parser.add_argument("--predict-velocity", action="store_true")
    parser.add_argument("--predict-velocity-magnitude", action="store_true")
    parser.add_argument("--velocity-x-manifest-column", type=str, default="path_to_peak_velx_10m")
    parser.add_argument("--velocity-y-manifest-column", type=str, default="path_to_peak_vely_10m")
    parser.add_argument("--velocity-mag-manifest-column", type=str, default="path_to_peak_velmag_10m")
    parser.add_argument("--early-stop-patience", type=int, default=12)
    defaults = parser.parse_args([])
    return vars(defaults)


def apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config_json is None:
        return args

    config = json.loads(args.config_json.read_text())
    defaults = parse_args_defaults()
    for key, default_value in defaults.items():
        if key in config and getattr(args, key, None) == default_value:
            setattr(args, key, config[key])
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return

    major, minor = torch.cuda.get_device_capability()
    if major < 5:
        # K80-class GPUs can trigger cuDNN arch mismatch with newer cuDNN builds.
        # Disable cuDNN and use native CUDA kernels instead.
        torch.backends.cudnn.enabled = False
        print(f"[CUDA] Disabled cuDNN for compute capability {major}.{minor}")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def masked_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    depth_weight_alpha: float = 0.0,
    depth_weight_cap: float = 3.0,
) -> torch.Tensor:
    abs_error = torch.abs(predictions - targets)
    quadratic = torch.minimum(abs_error, torch.full_like(abs_error, delta))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    if depth_weight_alpha > 0.0:
        depth_proxy = torch.clamp(targets, min=0.0, max=depth_weight_cap)
        depth_weights = 1.0 + depth_weight_alpha * depth_proxy
        cell_weights = mask * depth_weights
    else:
        cell_weights = mask

    weighted = loss * cell_weights
    return weighted.sum() / cell_weights.sum().clamp_min(1.0)


def masked_wet_bce_loss(
    wet_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    wet_threshold: float,
) -> torch.Tensor:
    wet_targets = (targets >= wet_threshold).float()
    loss = F.binary_cross_entropy_with_logits(wet_logits, wet_targets, reduction="none")
    weighted = loss * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def masked_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, float]:
    valid = masks > 0.5
    prediction_array = predictions[valid]
    target_array = targets[valid]
    if len(prediction_array) == 0:
        raise ValueError("Masked metric computation received zero valid target cells")

    errors = prediction_array - target_array
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    target_mean = float(np.mean(target_array))
    ss_tot = float(np.sum((target_array - target_mean) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def run_epoch(
    model: BlockwiseFloodMatrixModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    huber_delta: float,
    depth_weight_alpha: float = 0.0,
    depth_weight_cap: float = 3.0,
    aux_wet_loss_weight: float = 0.2,
    aux_velocity_loss_weight: float = 0.0,
    aux_velocity_mag_loss_weight: float = 0.0,
    wet_threshold: float = 0.05,
    velocity_huber_delta: float = 0.25,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_depth_loss = 0.0
    total_wet_loss = 0.0
    total_velocity_x_loss = 0.0
    total_velocity_y_loss = 0.0
    total_velocity_mag_loss = 0.0
    predictions = []
    targets = []
    masks = []

    for batch in loader:
        event_tensor, block_features, block_mask, target_map = batch[:4]
        cursor = 4
        target_velx = None
        target_vely = None
        target_velmag = None
        if model.predict_velocity:
            target_velx = batch[cursor]
            target_vely = batch[cursor + 1]
            cursor += 2
        if model.predict_velocity_magnitude:
            target_velmag = batch[cursor]
            cursor += 1
        static_raster = batch[cursor] if len(batch) > cursor else None

        if static_raster is not None:
            static_raster = static_raster.to(device)
        event_tensor = event_tensor.to(device)
        block_features = block_features.to(device)
        block_mask = block_mask.to(device)
        target_map = target_map.to(device)
        if target_velx is not None:
            target_velx = target_velx.to(device)
        if target_vely is not None:
            target_vely = target_vely.to(device)
        if target_velmag is not None:
            target_velmag = target_velmag.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            model_outputs = model(event_tensor, block_features, block_mask, static_raster)
            prediction_map = model_outputs[0]
            wet_logits = model_outputs[1]
            velocity_x_map = model_outputs[2] if model.predict_velocity else None
            velocity_y_map = model_outputs[3] if model.predict_velocity else None
            velocity_mag_map = None
            if model.predict_velocity_magnitude:
                velocity_mag_map = model_outputs[4] if model.predict_velocity else model_outputs[2]

            depth_loss = masked_huber_loss(
                prediction_map,
                target_map,
                block_mask,
                delta=huber_delta,
                depth_weight_alpha=depth_weight_alpha,
                depth_weight_cap=depth_weight_cap,
            )
            wet_loss = masked_wet_bce_loss(
                wet_logits,
                target_map,
                block_mask,
                wet_threshold=wet_threshold,
            )

            velocity_x_loss = torch.tensor(0.0, device=device)
            velocity_y_loss = torch.tensor(0.0, device=device)
            if model.predict_velocity and velocity_x_map is not None and velocity_y_map is not None and target_velx is not None and target_vely is not None:
                velocity_x_loss = masked_huber_loss(
                    velocity_x_map,
                    target_velx,
                    block_mask,
                    delta=velocity_huber_delta,
                )
                velocity_y_loss = masked_huber_loss(
                    velocity_y_map,
                    target_vely,
                    block_mask,
                    delta=velocity_huber_delta,
                )

            velocity_mag_loss = torch.tensor(0.0, device=device)
            if model.predict_velocity_magnitude and velocity_mag_map is not None and target_velmag is not None:
                velocity_mag_loss = masked_huber_loss(
                    velocity_mag_map,
                    target_velmag,
                    block_mask,
                    delta=velocity_huber_delta,
                )

            loss = (
                depth_loss
                + (aux_wet_loss_weight * wet_loss)
                + (aux_velocity_loss_weight * (velocity_x_loss + velocity_y_loss))
                + (aux_velocity_mag_loss_weight * velocity_mag_loss)
            )
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * event_tensor.shape[0]
        total_depth_loss += depth_loss.item() * event_tensor.shape[0]
        total_wet_loss += wet_loss.item() * event_tensor.shape[0]
        total_velocity_x_loss += velocity_x_loss.item() * event_tensor.shape[0]
        total_velocity_y_loss += velocity_y_loss.item() * event_tensor.shape[0]
        total_velocity_mag_loss += velocity_mag_loss.item() * event_tensor.shape[0]
        predictions.append(prediction_map.detach().cpu().numpy())
        targets.append(target_map.detach().cpu().numpy())
        masks.append(block_mask.detach().cpu().numpy())

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)
    metrics = masked_regression_metrics(prediction_array, target_array, mask_array)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["depth_loss"] = total_depth_loss / len(loader.dataset)
    metrics["wet_bce"] = total_wet_loss / len(loader.dataset)
    metrics["velocity_x_loss"] = total_velocity_x_loss / len(loader.dataset)
    metrics["velocity_y_loss"] = total_velocity_y_loss / len(loader.dataset)
    metrics["velocity_mag_loss"] = total_velocity_mag_loss / len(loader.dataset)
    return metrics


def save_split_tables(bundle: BlockwiseMatrixDataBundle, output_dir: Path) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    bundle.splits.train_df.to_csv(split_dir / "train_samples.csv", index=False)
    bundle.splits.val_df.to_csv(split_dir / "val_samples.csv", index=False)
    bundle.splits.test_df.to_csv(split_dir / "test_samples.csv", index=False)


def save_normalization(bundle: BlockwiseMatrixDataBundle, output_dir: Path) -> None:
    np.savez(
        output_dir / "normalization_stats.npz",
        event_mean=bundle.normalization.event_mean,
        event_std=bundle.normalization.event_std,
        block_mean=bundle.normalization.block_mean,
        block_std=bundle.normalization.block_std,
        block_feature_columns=np.asarray(bundle.feature_columns, dtype=object),
    )


def save_run_config(args: argparse.Namespace, bundle: BlockwiseMatrixDataBundle, output_dir: Path) -> None:
    config = {
        "events_csv": str(args.events_csv),
        "blocks_parquet": str(args.blocks_parquet),
        "labels_10m_dir": str(args.labels_10m_dir),
        "base_dir": str(args.base_dir),
        "block_feature_columns": bundle.feature_columns,
        "test_events": args.test_events,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "temporal_channels": args.temporal_channels,
        "event_embedding_dim": args.event_embedding_dim,
        "block_hidden_dim": args.block_hidden_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "decoder_base_channels": args.decoder_base_channels,
        "huber_delta": args.huber_delta,
        "depth_weight_alpha": args.depth_weight_alpha,
        "depth_weight_cap": args.depth_weight_cap,
        "aux_wet_loss_weight": args.aux_wet_loss_weight,
        "aux_velocity_loss_weight": args.aux_velocity_loss_weight,
        "aux_velocity_mag_loss_weight": args.aux_velocity_mag_loss_weight,
        "wet_threshold": args.wet_threshold,
        "velocity_huber_delta": args.velocity_huber_delta,
        "predict_velocity": args.predict_velocity,
        "predict_velocity_magnitude": args.predict_velocity_magnitude,
        "velocity_x_manifest_column": args.velocity_x_manifest_column,
        "velocity_y_manifest_column": args.velocity_y_manifest_column,
        "velocity_mag_manifest_column": args.velocity_mag_manifest_column,
        "target_shape": list(bundle.target_shape),
        "event_shape": list(bundle.event_shape),
        "train_samples": len(bundle.train_dataset),
        "val_samples": len(bundle.val_dataset),
        "test_samples": len(bundle.test_dataset),
        "static_rasters_dir": str(args.static_rasters_dir) if args.static_rasters_dir else None,
        "static_raster_channels": bundle.static_raster_channels,
        "raster_enc_channels": args.raster_enc_channels,
        "has_velocity_targets": bundle.has_velocity_targets,
        "has_velocity_magnitude_target": bundle.has_velocity_magnitude_target,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def main() -> None:
    args = apply_config_overrides(parse_args())
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_blockwise_matrix_datasets(
        events_csv=args.events_csv.resolve(),
        blocks_parquet=args.blocks_parquet.resolve(),
        labels_10m_dir=args.labels_10m_dir.resolve(),
        base_dir=args.base_dir.resolve(),
        feature_columns=args.block_feature_columns,
        test_events=args.test_events,
        val_fraction=args.val_fraction,
        seed=args.seed,
        target_rows=args.target_rows,
        target_cols=args.target_cols,
        static_rasters_dir=args.static_rasters_dir.resolve() if args.static_rasters_dir else None,
        include_velocity_targets=args.predict_velocity,
        include_velocity_magnitude_target=args.predict_velocity_magnitude,
        velocity_x_manifest_column=args.velocity_x_manifest_column,
        velocity_y_manifest_column=args.velocity_y_manifest_column,
        velocity_mag_manifest_column=args.velocity_mag_manifest_column,
    )

    save_split_tables(bundle, output_dir)
    save_normalization(bundle, output_dir)
    save_run_config(args, bundle, output_dir)

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    train_loader = make_loader(bundle.train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(bundle.val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(bundle.test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    _, event_feature_dim = bundle.event_shape
    model = BlockwiseFloodMatrixModel(
        event_features=event_feature_dim,
        block_features=len(bundle.feature_columns),
        target_rows=bundle.target_shape[0],
        target_cols=bundle.target_shape[1],
        temporal_channels=args.temporal_channels,
        event_embedding_dim=args.event_embedding_dim,
        block_hidden_dim=args.block_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        decoder_base_channels=args.decoder_base_channels,
        dropout=args.dropout,
        static_raster_channels=bundle.static_raster_channels,
        raster_enc_channels=args.raster_enc_channels,
        predict_velocity=args.predict_velocity,
        predict_velocity_magnitude=args.predict_velocity_magnitude,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    patience = 0
    checkpoint_path = output_dir / "best_model.pt"
    history = []

    print(
        f"[Data] train={len(bundle.train_dataset)} val={len(bundle.val_dataset)} test={len(bundle.test_dataset)} "
        f"| event_shape={bundle.event_shape} | target_shape={bundle.target_shape} | block_features={len(bundle.feature_columns)}"
    )
    if bundle.static_raster_channels > 0:
        print(f"[Data] static_raster_channels={bundle.static_raster_channels} raster_enc_channels={args.raster_enc_channels} dir={args.static_rasters_dir}")
    print(f"[Device] {device}")
    if args.depth_weight_alpha > 0.0:
        print(
            f"[Loss] depth_weight_alpha={args.depth_weight_alpha} depth_weight_cap={args.depth_weight_cap}"
        )
    print(
        f"[Loss] aux_wet_loss_weight={args.aux_wet_loss_weight} wet_threshold={args.wet_threshold}"
    )
    if args.predict_velocity:
        print(
            f"[Loss] aux_velocity_loss_weight={args.aux_velocity_loss_weight} velocity_huber_delta={args.velocity_huber_delta}"
        )
    if args.predict_velocity_magnitude:
        print(
            f"[Loss] aux_velocity_mag_loss_weight={args.aux_velocity_mag_loss_weight}"
        )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.huber_delta,
            depth_weight_alpha=args.depth_weight_alpha,
            depth_weight_cap=args.depth_weight_cap,
            aux_wet_loss_weight=args.aux_wet_loss_weight,
            aux_velocity_loss_weight=args.aux_velocity_loss_weight,
            aux_velocity_mag_loss_weight=args.aux_velocity_mag_loss_weight,
            wet_threshold=args.wet_threshold,
            velocity_huber_delta=args.velocity_huber_delta,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            args.huber_delta,
            depth_weight_alpha=args.depth_weight_alpha,
            depth_weight_cap=args.depth_weight_cap,
            aux_wet_loss_weight=args.aux_wet_loss_weight,
            aux_velocity_loss_weight=args.aux_velocity_loss_weight,
            aux_velocity_mag_loss_weight=args.aux_velocity_mag_loss_weight,
            wet_threshold=args.wet_threshold,
            velocity_huber_delta=args.velocity_huber_delta,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print(
            f"epoch {epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} train_rmse={train_metrics['rmse']:.6f} "
                f"train_wet_bce={train_metrics['wet_bce']:.6f} "
                f"train_vx={train_metrics['velocity_x_loss']:.6f} train_vy={train_metrics['velocity_y_loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_rmse={val_metrics['rmse']:.6f} "
                f"val_wet_bce={val_metrics['wet_bce']:.6f} "
                f"val_vx={val_metrics['velocity_x_loss']:.6f} val_vy={val_metrics['velocity_y_loss']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss - 1e-8:
            best_val_loss = val_metrics["loss"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "event_features": event_feature_dim,
                        "block_features": len(bundle.feature_columns),
                        "target_rows": bundle.target_shape[0],
                        "target_cols": bundle.target_shape[1],
                        "temporal_channels": args.temporal_channels,
                        "event_embedding_dim": args.event_embedding_dim,
                        "block_hidden_dim": args.block_hidden_dim,
                        "fusion_hidden_dim": args.fusion_hidden_dim,
                        "decoder_base_channels": args.decoder_base_channels,
                        "dropout": args.dropout,
                        "static_raster_channels": bundle.static_raster_channels,
                        "raster_enc_channels": args.raster_enc_channels,
                        "predict_velocity": args.predict_velocity,
                        "predict_velocity_magnitude": args.predict_velocity_magnitude,
                    },
                    "feature_columns": bundle.feature_columns,
                    "event_shape": bundle.event_shape,
                    "target_shape": bundle.target_shape,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = run_epoch(
        model,
        val_loader,
        None,
        device,
        args.huber_delta,
        depth_weight_alpha=args.depth_weight_alpha,
        depth_weight_cap=args.depth_weight_cap,
        aux_wet_loss_weight=args.aux_wet_loss_weight,
        aux_velocity_loss_weight=args.aux_velocity_loss_weight,
        aux_velocity_mag_loss_weight=args.aux_velocity_mag_loss_weight,
        wet_threshold=args.wet_threshold,
        velocity_huber_delta=args.velocity_huber_delta,
    )
    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        device,
        args.huber_delta,
        depth_weight_alpha=args.depth_weight_alpha,
        depth_weight_cap=args.depth_weight_cap,
        aux_wet_loss_weight=args.aux_wet_loss_weight,
        aux_velocity_loss_weight=args.aux_velocity_loss_weight,
        aux_velocity_mag_loss_weight=args.aux_velocity_mag_loss_weight,
        wet_threshold=args.wet_threshold,
        velocity_huber_delta=args.velocity_huber_delta,
    )

    metrics_payload = {
        "best_val_loss": best_val_loss,
        "final_val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    print(
        "[Final] "
        f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f} "
        f"| test_rmse={test_metrics['rmse']:.6f} test_mae={test_metrics['mae']:.6f}"
    )


if __name__ == "__main__":
    main()