import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from blockwise_matrix_data import prepare_blockwise_matrix_datasets
from blockwise_model import BlockwiseFloodMatrixModel
from train_blockwise_matrix import configure_cuda_runtime, make_loader, resolve_device, run_epoch, set_seed


DEFAULT_SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0, 1e-6, 1e-5],
    "batch_size": [8, 16, 32],
    "dropout": [0.0, 0.1, 0.2],
    "temporal_channels": [32, 64, 128],
    "event_embedding_dim": [32, 64, 128],
    "block_hidden_dim": [32, 64, 128],
    "fusion_hidden_dim": [64, 128, 256],
    "decoder_base_channels": [64, 128],
    "huber_delta": [0.1, 0.25, 0.5],
    "depth_weight_alpha": [0.0, 0.5, 1.0],
    "depth_weight_cap": [2.0, 3.0, 4.0],
    "aux_wet_loss_weight": [0.0, 0.1, 0.2, 0.4],
    "wet_threshold": [0.03, 0.05, 0.1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the 80x80 block-wise Triton Lite flood-depth surrogate")
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--block-feature-columns", nargs="+", default=None)
    parser.add_argument("--test-events", nargs="+", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument("--target-rows", type=int, default=80)
    parser.add_argument("--target-cols", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def sample_trial_params(rng: random.Random) -> Dict[str, float]:
    return {key: rng.choice(values) for key, values in DEFAULT_SEARCH_SPACE.items()}


def ensure_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

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
    )

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    _, event_feature_dim = bundle.event_shape
    trial_records = []
    best_trial = None
    best_val_loss = float("inf")

    for trial_index in range(1, args.num_trials + 1):
        params = sample_trial_params(rng)
        model = BlockwiseFloodMatrixModel(
            event_features=event_feature_dim,
            block_features=len(bundle.feature_columns),
            target_rows=bundle.target_shape[0],
            target_cols=bundle.target_shape[1],
            temporal_channels=int(params["temporal_channels"]),
            event_embedding_dim=int(params["event_embedding_dim"]),
            block_hidden_dim=int(params["block_hidden_dim"]),
            fusion_hidden_dim=int(params["fusion_hidden_dim"]),
            decoder_base_channels=int(params["decoder_base_channels"]),
            dropout=float(params["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        train_loader = make_loader(bundle.train_dataset, int(params["batch_size"]), shuffle=True, num_workers=args.num_workers)
        val_loader = make_loader(bundle.val_dataset, int(params["batch_size"]), shuffle=False, num_workers=args.num_workers)

        trial_best_val = float("inf")
        patience = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                float(params["huber_delta"]),
                depth_weight_alpha=float(params["depth_weight_alpha"]),
                depth_weight_cap=float(params["depth_weight_cap"]),
                aux_wet_loss_weight=float(params["aux_wet_loss_weight"]),
                wet_threshold=float(params["wet_threshold"]),
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                float(params["huber_delta"]),
                depth_weight_alpha=float(params["depth_weight_alpha"]),
                depth_weight_cap=float(params["depth_weight_cap"]),
                aux_wet_loss_weight=float(params["aux_wet_loss_weight"]),
                wet_threshold=float(params["wet_threshold"]),
            )
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            if val_metrics["loss"] < trial_best_val - 1e-8:
                trial_best_val = val_metrics["loss"]
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    break

        record = {
            "trial": trial_index,
            "params": {key: ensure_python_scalar(value) for key, value in params.items()},
            "best_val_loss": float(trial_best_val),
            "best_val_rmse": float(min(epoch["val"]["rmse"] for epoch in history)),
            "epochs_run": len(history),
        }
        trial_records.append(record)
        print(
            "trial {trial:03d} val_loss={val_loss:.6f} val_rmse={val_rmse:.6f} params={params}".format(
                trial=trial_index,
                val_loss=record["best_val_loss"],
                val_rmse=record["best_val_rmse"],
                params=record["params"],
            )
        )

        if trial_best_val < best_val_loss:
            best_val_loss = trial_best_val
            best_trial = record

    if best_trial is None:
        raise RuntimeError("No tuning trials were completed")

    best_config = {
        "block_feature_columns": bundle.feature_columns,
        "test_events": args.test_events,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "target_rows": bundle.target_shape[0],
        "target_cols": bundle.target_shape[1],
        "batch_size": best_trial["params"]["batch_size"],
        "learning_rate": best_trial["params"]["learning_rate"],
        "weight_decay": best_trial["params"]["weight_decay"],
        "dropout": best_trial["params"]["dropout"],
        "temporal_channels": best_trial["params"]["temporal_channels"],
        "event_embedding_dim": best_trial["params"]["event_embedding_dim"],
        "block_hidden_dim": best_trial["params"]["block_hidden_dim"],
        "fusion_hidden_dim": best_trial["params"]["fusion_hidden_dim"],
        "decoder_base_channels": best_trial["params"]["decoder_base_channels"],
        "huber_delta": best_trial["params"]["huber_delta"],
        "depth_weight_alpha": best_trial["params"]["depth_weight_alpha"],
        "depth_weight_cap": best_trial["params"]["depth_weight_cap"],
        "aux_wet_loss_weight": best_trial["params"]["aux_wet_loss_weight"],
        "wet_threshold": best_trial["params"]["wet_threshold"],
    }

    (output_dir / "trials.json").write_text(json.dumps(trial_records, indent=2))
    (output_dir / "best_config.json").write_text(json.dumps(best_config, indent=2))
    summary = {
        "best_trial": best_trial,
        "num_trials": len(trial_records),
        "event_shape": list(bundle.event_shape),
        "target_shape": list(bundle.target_shape),
        "block_feature_columns": bundle.feature_columns,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Best config written to {}".format(output_dir / "best_config.json"))


if __name__ == "__main__":
    main()