"""
Compute uncertainty metrics from an ensemble of models.

This script:
1. Trains 5 seeds with identical config (or loads pretrained checkpoints)
2. Runs inference on test/OOD/spatial-holdout splits
3. Computes depth mean, std, prediction intervals
4. Evaluates wet probability calibration (Brier, ECE, reliability)
5. Reports coverage and sharpness for uncertainty quantification

Usage:
    python compute_uncertainty_ensemble.py \
        --reference-run results_blockwise_matrix_train_v3 \
        --num-seeds 5 \
        --output-dir results_ensemble_uncertainty_v3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from blockwise_matrix_data import prepare_blockwise_matrix_datasets
from blockwise_model import BlockwiseFloodMatrixModel
from predict_blockwise_matrix import load_checkpoint
from train_blockwise_matrix import (
    configure_cuda_runtime,
    make_loader,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty from multi-seed ensemble"
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Path to reference training run (contains run_config.json and splits/)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds to train/load",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for uncertainty results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs per seed (if training from scratch)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Starting seed; seeds will be base_seed, base_seed+1, ..., base_seed+num_seeds-1",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load pre-trained checkpoints instead of training",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to save/load per-seed checkpoints (supports resume after walltime preemption)",
    )
    parser.add_argument(
        "--pi-levels",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95],
        help="Prediction interval levels to evaluate, e.g. --pi-levels 0.8 0.9 0.95",
    )
    parser.add_argument(
        "--pi-method",
        type=str,
        default="empirical",
        choices=["empirical"],
        help="PI computation method; empirical uses ensemble quantiles",
    )
    return parser.parse_args()


def load_reference_config(ref_run: Path) -> Dict:
    """Load run config from reference training run."""
    config_path = ref_run / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Reference config not found: {config_path}")
    return json.loads(config_path.read_text())


def load_data_bundle(ref_run: Path, ref_config: Dict, base_dir: Path = Path(".")):
    """Load data bundle using reference splits."""
    bundle = prepare_blockwise_matrix_datasets(
        events_csv=(base_dir / ref_config["events_csv"]).resolve(),
        blocks_parquet=(base_dir / ref_config["blocks_parquet"]).resolve(),
        labels_10m_dir=(base_dir / ref_config["labels_10m_dir"]).resolve(),
        base_dir=base_dir.resolve(),
        feature_columns=ref_config["block_feature_columns"],
        test_events=ref_config["test_events"],
        val_fraction=ref_config["val_fraction"],
        seed=ref_config["seed"],
        target_rows=ref_config["target_shape"][0],
        target_cols=ref_config["target_shape"][1],
        static_rasters_dir=(
            (base_dir / ref_config["static_rasters_dir"]).resolve()
            if ref_config.get("static_rasters_dir")
            else None
        ),
    )
    return bundle


def build_model_from_config(ref_config: Dict, device: torch.device) -> BlockwiseFloodMatrixModel:
    """Instantiate model from reference config."""
    model = BlockwiseFloodMatrixModel(
        event_features=ref_config["event_shape"][1],
        block_features=len(ref_config["block_feature_columns"]),
        target_rows=ref_config["target_shape"][0],
        target_cols=ref_config["target_shape"][1],
        temporal_channels=ref_config["temporal_channels"],
        event_embedding_dim=ref_config["event_embedding_dim"],
        block_hidden_dim=ref_config["block_hidden_dim"],
        fusion_hidden_dim=ref_config["fusion_hidden_dim"],
        decoder_base_channels=ref_config["decoder_base_channels"],
        dropout=ref_config.get("dropout", 0.1),
        static_raster_channels=ref_config.get("static_raster_channels", 0),
        raster_enc_channels=ref_config.get("raster_enc_channels", 16),
    ).to(device)
    return model


def train_seed_model(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    ref_config: Dict,
    epochs: int,
) -> Dict:
    """Train model for one seed. Returns best checkpoint state dict."""
    from train_blockwise_matrix import run_epoch

    set_seed(seed)
    model_copy = build_model_from_config(ref_config, device)

    optimizer = torch.optim.Adam(
        model_copy.parameters(),
        lr=ref_config.get("learning_rate", 1e-3),
        weight_decay=ref_config.get("weight_decay", 1e-5),
    )

    best_val_loss = float("inf")
    patience = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model_copy,
            train_loader,
            optimizer,
            device,
            ref_config.get("huber_delta", 0.25),
            depth_weight_alpha=ref_config.get("depth_weight_alpha", 0.0),
            depth_weight_cap=ref_config.get("depth_weight_cap", 3.0),
            aux_wet_loss_weight=ref_config.get("aux_wet_loss_weight", 0.2),
            wet_threshold=ref_config.get("wet_threshold", 0.05),
        )
        val_metrics = run_epoch(
            model_copy,
            val_loader,
            None,
            device,
            ref_config.get("huber_delta", 0.25),
            depth_weight_alpha=ref_config.get("depth_weight_alpha", 0.0),
            depth_weight_cap=ref_config.get("depth_weight_cap", 3.0),
            aux_wet_loss_weight=ref_config.get("aux_wet_loss_weight", 0.2),
            wet_threshold=ref_config.get("wet_threshold", 0.05),
        )

        if epoch % 10 == 0:
            print(
                f"[Seed {seed} Epoch {epoch:3d}] "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f}"
            )

        if val_metrics["loss"] < best_val_loss - 1e-8:
            best_val_loss = val_metrics["loss"]
            patience = 0
            best_state = {
                "model_state_dict": model_copy.state_dict().copy(),
                "best_val_loss": best_val_loss,
            }
        else:
            patience += 1
            if patience >= ref_config.get("early_stop_patience", 12):
                print(f"[Seed {seed}] Early stopping at epoch {epoch}")
                break

    return best_state


def seed_checkpoint_path(checkpoint_dir: Path, seed: int) -> Path:
    return checkpoint_dir / f"seed_{seed}" / "best_model.pt"


def save_seed_checkpoint(checkpoint_path: Path, state: Dict, ref_config: Dict, seed: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": state["model_state_dict"],
        "best_val_loss": state["best_val_loss"],
        "seed": seed,
        "model_config": {
            "event_features": ref_config["event_shape"][1],
            "block_features": len(ref_config["block_feature_columns"]),
            "target_rows": ref_config["target_shape"][0],
            "target_cols": ref_config["target_shape"][1],
            "temporal_channels": ref_config["temporal_channels"],
            "event_embedding_dim": ref_config["event_embedding_dim"],
            "block_hidden_dim": ref_config["block_hidden_dim"],
            "fusion_hidden_dim": ref_config["fusion_hidden_dim"],
            "decoder_base_channels": ref_config["decoder_base_channels"],
            "dropout": ref_config.get("dropout", 0.1),
            "static_raster_channels": ref_config.get("static_raster_channels", 0),
            "raster_enc_channels": ref_config.get("raster_enc_channels", 16),
        },
    }
    torch.save(payload, checkpoint_path)


def run_inference_ensemble(
    models: List[BlockwiseFloodMatrixModel],
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ensemble inference.

    Returns:
        depth_preds: (n_samples, 80, 80, n_seeds)
        wet_logits: (n_samples, 80, 80, n_seeds)
        targets: (n_samples, 80, 80)
        masks: (n_samples, 80, 80)
    """
    depth_preds_list = []
    wet_logits_list = []
    targets_list = []
    masks_list = []

    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 5:
                event_tensor, block_features, block_mask, target_map, static_raster = [
                    b.to(device) for b in batch
                ]
            else:
                event_tensor, block_features, block_mask, target_map = [
                    b.to(device) for b in batch
                ]
                static_raster = None

            depth_batch = []
            wet_batch = []

            for model in models:
                depth_pred, wet_logits = model(
                    event_tensor,
                    block_features,
                    block_mask,
                    static_raster=static_raster,
                )
                depth_batch.append(depth_pred.detach().cpu().numpy())
                wet_batch.append(wet_logits.detach().cpu().numpy())

            depth_preds_list.append(np.stack(depth_batch, axis=-1))
            wet_logits_list.append(np.stack(wet_batch, axis=-1))
            targets_list.append(target_map.detach().cpu().numpy())
            masks_list.append(block_mask.detach().cpu().numpy())

    depth_preds = np.concatenate(depth_preds_list, axis=0)  # (n_samples, 80, 80, n_seeds)
    wet_logits = np.concatenate(wet_logits_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)

    return depth_preds, wet_logits, targets, masks


def compute_calibration_metrics(
    wet_probs: np.ndarray,
    wet_targets: np.ndarray,
    masks: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute wet probability calibration metrics.

    Args:
        wet_probs: (n_samples, 80, 80) ensemble mean wet probabilities
        wet_targets: (n_samples, 80, 80) binary targets
        masks: (n_samples, 80, 80)
        n_bins: number of calibration bins

    Returns:
        Brier score, ECE, calibration stats
    """
    valid = masks > 0.5
    probs_valid = wet_probs[valid].flatten()
    targets_valid = wet_targets[valid].flatten()

    # Brier score
    brier = float(np.mean((probs_valid - targets_valid) ** 2))

    # ECE (Expected Calibration Error)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        in_bin = (probs_valid >= bin_edges[i]) & (probs_valid < bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = targets_valid[in_bin].mean()
        bin_conf = probs_valid[in_bin].mean()
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_sizes.append(in_bin.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_sizes = np.array(bin_sizes)
    bin_weights = bin_sizes / bin_sizes.sum()

    ece = float(np.sum(bin_weights * np.abs(bin_accs - bin_confs)))

    return {
        "brier_score": brier,
        "ece": ece,
        "bin_accs": bin_accs.tolist(),
        "bin_confs": bin_confs.tolist(),
        "bin_sizes": bin_sizes.tolist(),
    }


def compute_uncertainty_metrics(
    depth_preds: np.ndarray,
    wet_logits: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    wet_threshold: float = 0.05,
    pi_levels: List[float] = None,
    pi_method: str = "empirical",
) -> Dict:
    """
    Compute comprehensive uncertainty metrics.

    Args:
        depth_preds: (n_samples, 80, 80, n_seeds)
        wet_logits: (n_samples, 80, 80, n_seeds)
        targets: (n_samples, 80, 80)
        masks: (n_samples, 80, 80)
        wet_threshold: depth threshold for binary wet label
        pi_levels: prediction interval levels (e.g., [0.8, 0.9, 0.95])
        pi_method: PI method (currently empirical quantiles)

    Returns:
        Dictionary with depth and wet metrics
    """
    if pi_levels is None:
        pi_levels = [0.90]

    n_seeds = depth_preds.shape[-1]
    depth_mean = depth_preds.mean(axis=-1)  # (n, 80, 80)
    depth_std = depth_preds.std(axis=-1)

    wet_probs_list = [1.0 / (1.0 + np.exp(-wet_logits[..., i])) for i in range(n_seeds)]
    wet_probs_mean = np.stack(wet_probs_list, axis=-1).mean(axis=-1)  # (n, 80, 80)

    # Depth regression metrics
    valid = masks > 0.5
    pred_valid = depth_mean[valid]
    target_valid = targets[valid]
    errors = pred_valid - target_valid

    depth_metrics = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "r2": float(
            1.0 - np.sum(errors ** 2) / np.sum((target_valid - target_valid.mean()) ** 2)
        ),
    }

    # Prediction intervals (empirical ensemble quantiles)
    if pi_method != "empirical":
        raise ValueError(f"Unsupported pi_method: {pi_method}")

    pi_metrics = {}
    for level in pi_levels:
        if level <= 0.0 or level >= 1.0:
            raise ValueError(f"PI level must be in (0,1), got {level}")
        alpha = (1.0 - level) / 2.0
        pi_lower = np.quantile(depth_preds, alpha, axis=-1)
        pi_upper = np.quantile(depth_preds, 1.0 - alpha, axis=-1)
        coverage = (target_valid >= pi_lower[valid]) & (target_valid <= pi_upper[valid])
        pi_metrics[f"pi_{int(level * 100)}"] = {
            "target_coverage": float(level),
            "coverage": float(coverage.mean()),
            "coverage_error": float(coverage.mean() - level),
            "mean_width": float((pi_upper[valid] - pi_lower[valid]).mean()),
        }

    # Wet probability calibration
    wet_targets = (targets >= wet_threshold).astype(np.float32)
    wet_calib = compute_calibration_metrics(wet_probs_mean, wet_targets, masks)

    # Depth-bin analysis
    depth_bins = [(0.0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100.0)]
    bin_rmses = {}
    for lo, hi in depth_bins:
        in_bin = (target_valid >= lo) & (target_valid < hi)
        if in_bin.sum() == 0:
            bin_rmses[f"{lo:.1f}-{hi:.1f}m"] = np.nan
        else:
            bin_rmse = float(np.sqrt(np.mean((pred_valid[in_bin] - target_valid[in_bin]) ** 2)))
            bin_rmses[f"{lo:.1f}-{hi:.1f}m"] = bin_rmse

    return {
        "depth_metrics": depth_metrics,
        "pi_metrics": pi_metrics,
        "pi_method": pi_method,
        "wet_calibration": wet_calib,
        "depth_bin_rmses": bin_rmses,
        "ensemble_uncertainty": {
            "mean_depth_std": float(depth_std[valid].mean()),
            "p90_depth_std": float(np.quantile(depth_std[valid], 0.90)),
            "n_seeds": int(n_seeds),
        },
        "n_valid_pixels": int(valid.sum()),
    }


def save_ensemble_outputs(
    output_dir: Path,
    depth_preds: np.ndarray,
    wet_logits: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    metrics: Dict,
):
    """Save ensemble predictions and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble arrays
    np.save(output_dir / "depth_preds_ensemble.npy", depth_preds)
    np.save(output_dir / "wet_logits_ensemble.npy", wet_logits)
    np.save(output_dir / "targets.npy", targets)
    np.save(output_dir / "masks.npy", masks)

    # Save metrics
    with open(output_dir / "uncertainty_metrics.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            return obj

        json.dump(convert_types(metrics), f, indent=2)

    print(f"[Output] Saved to {output_dir}")
    print(json.dumps(metrics, indent=2, default=str))


def main():
    args = parse_args()
    ref_config = load_reference_config(args.reference_run)
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    configure_cuda_runtime(device)

    print("[Data] Loading reference splits...")
    bundle = load_data_bundle(args.reference_run, ref_config)

    test_loader = make_loader(bundle.test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader = make_loader(
        bundle.train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(bundle.val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"[Data] test={len(bundle.test_dataset)} train={len(bundle.train_dataset)} val={len(bundle.val_dataset)}")

    models = []
    print(f"\n[Ensemble] Preparing {args.num_seeds} seeds...")

    for i in range(args.num_seeds):
        seed = args.base_seed + i
        print(f"\n--- Seed {i+1}/{args.num_seeds} (seed={seed}) ---")
        ckpt_path = seed_checkpoint_path(checkpoint_dir, seed)
        model = build_model_from_config(ref_config, device)

        if ckpt_path.exists():
            print(f"[Seed {seed}] Loading existing checkpoint: {ckpt_path}")
            checkpoint = load_checkpoint(ckpt_path, device)
            model.load_state_dict(checkpoint["model_state_dict"])
            models.append(model)
            continue

        if args.skip_training:
            raise FileNotFoundError(
                f"--skip-training was set but checkpoint not found for seed {seed}: {ckpt_path}"
            )

        print(f"[Seed {seed}] Training from scratch")
        best_checkpoint = train_seed_model(
            seed, train_loader, val_loader, device, ref_config, args.epochs
        )
        save_seed_checkpoint(ckpt_path, best_checkpoint, ref_config, seed)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        models.append(model)

    print(f"\n[Inference] Running ensemble on test set...")
    depth_preds, wet_logits, targets, masks = run_inference_ensemble(models, test_loader, device)

    print(f"[Metrics] Computing uncertainty metrics...")
    metrics = compute_uncertainty_metrics(
        depth_preds,
        wet_logits,
        targets,
        masks,
        wet_threshold=ref_config.get("wet_threshold", 0.05),
        pi_levels=args.pi_levels,
        pi_method=args.pi_method,
    )

    save_ensemble_outputs(args.output_dir, depth_preds, wet_logits, targets, masks, metrics)


if __name__ == "__main__":
    main()
