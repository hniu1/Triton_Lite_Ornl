import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from blockwise_data import (
    REQUIRED_BLOCK_COLUMNS,
    REQUIRED_EVENT_COLUMNS,
    REQUIRED_LABEL_COLUMNS,
    _resolve_event_path,
    _validate_columns,
    event_key,
)
from blockwise_model import BlockwiseFloodModel
from train_blockwise import load_checkpoint, regression_metrics, resolve_device


class InferenceDataset(Dataset):
    def __init__(self, frame, event_arrays, block_feature_map):
        self.frame = frame.reset_index(drop=True)
        self.event_arrays = event_arrays
        self.block_feature_map = block_feature_map

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return (
            torch.from_numpy(self.event_arrays[row["event_key"]].copy()),
            torch.from_numpy(self.block_feature_map[(row["watershed_id"], row["block_id"])].copy()),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict block-wise flood depth with a trained block-wise surrogate")
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--labels-parquet", type=Path, default=None, help="Optional labels for evaluation")
    parser.add_argument("--event-ids", nargs="+", default=None, help="Optional subset of event IDs to score")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_events(events_csv, base_dir):
    events_df = pd.read_csv(events_csv)
    _validate_columns(events_df, REQUIRED_EVENT_COLUMNS, "events.csv")
    events_df = events_df.copy()
    events_df["event_key"] = [
        event_key(watershed_id, event_id)
        for watershed_id, event_id in zip(events_df["watershed_id"], events_df["event_id"])
    ]
    events_df["resolved_event_path"] = [
        str(_resolve_event_path(raw_path, base_dir, events_csv, watershed_id, event_id))
        for raw_path, watershed_id, event_id in zip(
            events_df["path_to_X_event"],
            events_df["watershed_id"],
            events_df["event_id"],
        )
    ]
    return events_df


def load_normalization(path):
    stats = np.load(path, allow_pickle=True)
    feature_columns = [str(value) for value in stats["block_feature_columns"].tolist()]
    return {
        "event_mean": stats["event_mean"].astype(np.float32),
        "event_std": stats["event_std"].astype(np.float32),
        "block_mean": stats["block_mean"].astype(np.float32),
        "block_std": stats["block_std"].astype(np.float32),
        "feature_columns": feature_columns,
    }


def load_normalized_events(events_df, event_mean, event_std):
    arrays = {}
    expected_shape = None
    for row in events_df.itertuples(index=False):
        array = np.load(row.resolved_event_path).astype(np.float32)
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("All inference events must share one common shape")
        arrays[row.event_key] = ((array - event_mean) / event_std).astype(np.float32)
    return arrays


def build_block_feature_map(blocks_df, feature_columns, block_mean, block_std):
    mapping = {}
    for row in blocks_df.itertuples(index=False):
        vector = np.asarray([getattr(row, column) for column in feature_columns], dtype=np.float32)
        mapping[(row.watershed_id, row.block_id)] = ((vector - block_mean) / block_std).astype(np.float32)
    return mapping


def build_inference_frame(events_df, blocks_df, labels_df=None):
    if labels_df is not None:
        frame = labels_df[["event_id", "watershed_id", "block_id"]].copy()
    else:
        frame = events_df[["event_id", "watershed_id", "event_key"]].merge(
            blocks_df[["watershed_id", "block_id"]],
            on="watershed_id",
            how="inner",
        )
    frame = frame.merge(
        events_df[["event_id", "watershed_id", "event_key"]],
        on=["event_id", "watershed_id"],
        how="inner",
        validate="many_to_one",
    )
    return frame.sort_values(["watershed_id", "event_id", "block_id"]).reset_index(drop=True)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    events_df = load_events(args.events_csv.resolve(), args.base_dir.resolve())
    if args.event_ids:
        requested = set(args.event_ids)
        events_df = events_df.loc[events_df["event_id"].isin(requested)].reset_index(drop=True)
        if events_df.empty:
            raise ValueError("No events matched --event-ids")

    blocks_df = pd.read_parquet(args.blocks_parquet.resolve())
    _validate_columns(blocks_df, REQUIRED_BLOCK_COLUMNS, "blocks.parquet")

    labels_df = None
    if args.labels_parquet is not None:
        labels_df = pd.read_parquet(args.labels_parquet.resolve())
        _validate_columns(labels_df, REQUIRED_LABEL_COLUMNS, "labels.parquet")
        if args.event_ids:
            labels_df = labels_df.loc[labels_df["event_id"].isin(set(args.event_ids))].reset_index(drop=True)

    norm = load_normalization(args.normalization_stats.resolve())
    missing_features = [column for column in norm["feature_columns"] if column not in blocks_df.columns]
    if missing_features:
        raise ValueError("blocks.parquet is missing feature columns required by normalization stats: {}".format(missing_features))

    event_arrays = load_normalized_events(events_df, norm["event_mean"], norm["event_std"])
    block_feature_map = build_block_feature_map(
        blocks_df[["watershed_id", "block_id"] + norm["feature_columns"]].copy(),
        norm["feature_columns"],
        norm["block_mean"],
        norm["block_std"],
    )
    inference_frame = build_inference_frame(events_df, blocks_df, labels_df)
    dataset = InferenceDataset(inference_frame, event_arrays, block_feature_map)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = load_checkpoint(args.checkpoint.resolve(), device)
    model = BlockwiseFloodModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for event_tensor, block_features in loader:
            pred = model(event_tensor.to(device), block_features.to(device))
            predictions.append(pred.cpu().numpy().astype(np.float32))
    inference_frame["y_pred"] = np.concatenate(predictions)

    if labels_df is not None:
        output_df = inference_frame.merge(
            labels_df[["event_id", "watershed_id", "block_id", "y"]],
            on=["event_id", "watershed_id", "block_id"],
            how="left",
        )
        valid = output_df["y"].notna()
        metrics = regression_metrics(
            output_df.loc[valid, "y_pred"].to_numpy(dtype=np.float32),
            output_df.loc[valid, "y"].to_numpy(dtype=np.float32),
        )
        metrics_path = args.output_parquet.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(
            "[Metrics] rmse={rmse:.6f} mae={mae:.6f} r2={r2:.6f}".format(
                rmse=metrics["rmse"], mae=metrics["mae"], r2=metrics["r2"]
            )
        )
    else:
        output_df = inference_frame

    output_df.to_parquet(args.output_parquet, index=False)
    print("[Output] wrote {} rows to {}".format(len(output_df), args.output_parquet))


if __name__ == "__main__":
    main()
