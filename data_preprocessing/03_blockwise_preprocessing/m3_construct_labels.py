import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio


LOGGER = logging.getLogger("m3_construct_labels")


TIFF_PATTERN = re.compile(
    r"^(?P<watershed>[a-z0-9_]+)_ACC_(?P<event>D\d{3})_(?P=watershed)_block_(?P<block_idx>\d+)\.tif$",
    flags=re.IGNORECASE,
)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_id(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def infer_watershed_from_dir(path: Path) -> str:
    name = path.name.lower()
    for prefix in ["block_tiffs_", "water_depth_rasters_"]:
        if name.startswith(prefix):
            return normalize_id(name[len(prefix) :])
    return normalize_id(name)


def build_block_id(watershed_id: str, block_index: int, block_id_mode: str) -> str:
    if block_id_mode == "index":
        return str(block_index)
    if block_id_mode == "watershed_block":
        return f"{watershed_id}_block_{block_index}"
    if block_id_mode == "watershed_b_padded":
        return f"{watershed_id}_b{block_index:06d}"
    raise ValueError(f"Unsupported block_id_mode: {block_id_mode}")


def peak_depth_from_tiff(path: Path) -> float:
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)

    if np.ma.isMaskedArray(band):
        values = np.asarray(band.compressed(), dtype=np.float32)
    else:
        values = np.asarray(band, dtype=np.float32).ravel()
        values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def collect_tiff_files(input_dirs: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Input directory does not exist: {directory}")
        files.extend(sorted(directory.glob("*.tif")))
    if not files:
        raise FileNotFoundError("No .tif files found in --input-dirs")
    return files


def parse_file_metadata(path: Path, default_watershed_id: Optional[str]) -> Tuple[str, str, int]:
    filename = path.name
    match = TIFF_PATTERN.match(filename)
    if match:
        watershed_id = normalize_id(match.group("watershed"))
        event_id = match.group("event").upper()
        block_index = int(match.group("block_idx"))
        return watershed_id, event_id, block_index

    event_match = re.search(r"(D\d{3})", filename, flags=re.IGNORECASE)
    block_match = re.search(r"block[_-]?(\d+)", filename, flags=re.IGNORECASE)
    if not event_match or not block_match:
        raise ValueError(f"Unable to parse event/block from filename: {filename}")

    event_id = event_match.group(1).upper()
    block_index = int(block_match.group(1))

    if default_watershed_id:
        watershed_id = normalize_id(default_watershed_id)
    else:
        watershed_id = infer_watershed_from_dir(path.parent)
    return watershed_id, event_id, block_index


def build_labels_table(
    tiff_files: Sequence[Path],
    block_id_mode: str,
    default_watershed_id: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for idx, path in enumerate(tiff_files, start=1):
        watershed_id, event_id, block_index = parse_file_metadata(path, default_watershed_id)
        block_id = build_block_id(watershed_id, block_index, block_id_mode)
        y_value = peak_depth_from_tiff(path)

        rows.append(
            {
                "event_id": event_id,
                "watershed_id": watershed_id,
                "block_id": block_id,
                "y": y_value,
                "_source_tiff": str(path),
            }
        )

        if idx % 500 == 0:
            LOGGER.info("Processed %d TIFF files", idx)

    labels = pd.DataFrame(rows)
    labels = labels.sort_values(["watershed_id", "event_id", "block_id"]).reset_index(drop=True)
    return labels


def validate_pair_completeness(labels: pd.DataFrame) -> None:
    issues: List[str] = []
    for watershed_id, group in labels.groupby("watershed_id", sort=False):
        event_ids = sorted(group["event_id"].unique().tolist())
        block_ids = sorted(group["block_id"].unique().tolist())
        expected = len(event_ids) * len(block_ids)
        actual = len(group)
        if actual != expected:
            issues.append(
                f"watershed_id={watershed_id}: expected {expected} pairs ({len(event_ids)} events x {len(block_ids)} blocks), got {actual}"
            )

    if issues:
        raise ValueError("Missing event-block pairs detected:\n" + "\n".join(issues))


def validate_against_events(labels: pd.DataFrame, events_csv: Path) -> None:
    events = pd.read_csv(events_csv)
    required = {"event_id", "watershed_id"}
    missing_cols = required - set(events.columns)
    if missing_cols:
        raise ValueError(f"events_csv missing required columns: {sorted(missing_cols)}")

    events["event_id"] = events["event_id"].astype(str).str.upper()
    events["watershed_id"] = events["watershed_id"].astype(str).map(normalize_id)

    events_key = set(events[["watershed_id", "event_id"]].itertuples(index=False, name=None))
    labels_key = set(labels[["watershed_id", "event_id"]].itertuples(index=False, name=None))

    missing_in_events = labels_key - events_key
    if missing_in_events:
        sample = sorted(list(missing_in_events))[:10]
        raise ValueError(f"labels contain watershed/event not found in events_csv, sample: {sample}")


def validate_against_blocks(labels: pd.DataFrame, blocks_parquet: Path) -> None:
    blocks = pd.read_parquet(blocks_parquet)
    required = {"watershed_id", "block_id"}
    missing_cols = required - set(blocks.columns)
    if missing_cols:
        raise ValueError(f"blocks_parquet missing required columns: {sorted(missing_cols)}")

    blocks["watershed_id"] = blocks["watershed_id"].astype(str).map(normalize_id)
    blocks["block_id"] = blocks["block_id"].astype(str)

    blocks_key = set(blocks[["watershed_id", "block_id"]].itertuples(index=False, name=None))
    labels_key = set(labels[["watershed_id", "block_id"]].itertuples(index=False, name=None))

    missing_in_blocks = labels_key - blocks_key
    if missing_in_blocks:
        sample = sorted(list(missing_in_blocks))[:10]
        raise ValueError(f"labels contain watershed/block not found in blocks_parquet, sample: {sample}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 3: Construct labels.parquet from hydraulic block outputs")
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True, help="Directories containing per-event block TIFF outputs")
    parser.add_argument(
        "--block-id-mode",
        choices=["index", "watershed_block", "watershed_b_padded"],
        default="watershed_block",
        help="How to build block_id from parsed block index",
    )
    parser.add_argument(
        "--default-watershed-id",
        type=str,
        default=None,
        help="Override watershed ID for all parsed files (optional)",
    )
    parser.add_argument("--events-csv", type=Path, default=None, help="Optional events table for validation")
    parser.add_argument("--blocks-parquet", type=Path, default=None, help="Optional blocks table for validation")
    parser.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="Allow incomplete event-block Cartesian coverage within each watershed",
    )
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    tiff_files = collect_tiff_files(args.input_dirs)
    LOGGER.info("Found %d TIFF files", len(tiff_files))

    labels = build_labels_table(
        tiff_files=tiff_files,
        block_id_mode=args.block_id_mode,
        default_watershed_id=args.default_watershed_id,
    )

    duplicates = labels.duplicated(subset=["event_id", "watershed_id", "block_id"])
    if duplicates.any():
        dup_count = int(duplicates.sum())
        raise ValueError(f"Duplicate labels found for event/watershed/block keys: {dup_count}")

    if labels["y"].isna().any():
        missing = int(labels["y"].isna().sum())
        raise ValueError(f"Found {missing} labels with NaN y values")

    if not args.allow_missing_pairs:
        validate_pair_completeness(labels)

    if args.events_csv is not None:
        validate_against_events(labels, args.events_csv)
        LOGGER.info("Validated labels against events table: %s", args.events_csv)

    if args.blocks_parquet is not None:
        validate_against_blocks(labels, args.blocks_parquet)
        LOGGER.info("Validated labels against blocks table: %s", args.blocks_parquet)

    out = labels[["event_id", "watershed_id", "block_id", "y"]].copy()
    out = out.sort_values(["watershed_id", "event_id", "block_id"]).reset_index(drop=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_parquet, index=False)

    LOGGER.info("Wrote labels table with %d rows -> %s", len(out), args.output_parquet)
    LOGGER.info(
        "Summary: watersheds=%d events=%d hydro_blocks=%d",
        out["watershed_id"].nunique(),
        out["event_id"].nunique(),
        out["block_id"].nunique(),
    )


if __name__ == "__main__":
    main()
