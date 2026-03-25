import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds


LOGGER = logging.getLogger("m3_construct_labels_from_netcdf")


@dataclass
class BlockMask:
    watershed_id: str
    block_id: str
    row_off: int
    col_off: int
    height: int
    width: int
    mask: np.ndarray


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_id(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_event_id_from_nc(path: Path) -> str:
    match = re.search(r"(D\d{3})", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse event id from netCDF filename: {path.name}")
    return match.group(1).upper()


def block_id_from_row(row: pd.Series, watershed_id: str, block_id_column: Optional[str], block_id_mode: str, index_1b: int) -> str:
    if block_id_column and block_id_column in row.index:
        return str(row[block_id_column])

    if block_id_mode == "watershed_b_padded":
        return f"{watershed_id}_b{index_1b:06d}"
    if block_id_mode == "watershed_block":
        return f"{watershed_id}_block_{index_1b-1}"
    if block_id_mode == "index":
        return str(index_1b - 1)
    raise ValueError(f"Unsupported block_id_mode: {block_id_mode}")


def infer_transform_from_nc(nc_path: Path, depth_var: str) -> Tuple[Affine, int, int]:
    with nc4.Dataset(nc_path, "r") as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        if depth_var not in ds.variables:
            raise ValueError(f"Variable '{depth_var}' not found in {nc_path}")
        _, rows, cols = ds.variables[depth_var].shape

    if len(x) < 2 or len(y) < 2:
        raise ValueError("x/y coordinates in netCDF are insufficient to infer transform")

    xres = float(x[1] - x[0])
    yres = float(y[1] - y[0])
    xmin = float(x[0] - xres / 2.0)
    ymax = float(y[0] - yres / 2.0)
    transform = Affine.translation(xmin, ymax) * Affine.scale(xres, yres)
    return transform, rows, cols


def clamp_window(window: Window, width: int, height: int) -> Optional[Window]:
    row_off = max(0, int(np.floor(window.row_off)))
    col_off = max(0, int(np.floor(window.col_off)))
    row_max = min(height, int(np.ceil(window.row_off + window.height)))
    col_max = min(width, int(np.ceil(window.col_off + window.width)))
    h = row_max - row_off
    w = col_max - col_off
    if h <= 0 or w <= 0:
        return None
    return Window(col_off=col_off, row_off=row_off, width=w, height=h)


def build_block_masks(
    blocks_file: Path,
    watershed_id: str,
    transform: Affine,
    rows: int,
    cols: int,
    blocks_crs: Optional[str],
    nc_crs: str,
    block_id_column: Optional[str],
    block_id_mode: str,
    max_blocks: Optional[int],
) -> List[BlockMask]:
    gdf = gpd.read_file(blocks_file)
    if gdf.crs is None:
        if not blocks_crs:
            raise ValueError("Blocks file has no CRS; provide --blocks-crs")
        gdf = gdf.set_crs(blocks_crs)
        LOGGER.warning("Set missing blocks CRS with override: %s", blocks_crs)

    gdf = gdf.to_crs(nc_crs)
    gdf = gdf.loc[gdf.geometry.notnull() & ~gdf.geometry.is_empty].reset_index(drop=True)
    if max_blocks is not None:
        gdf = gdf.iloc[:max_blocks].copy()

    masks: List[BlockMask] = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        block_id = block_id_from_row(row, watershed_id, block_id_column, block_id_mode, i + 1)
        window = from_bounds(*geom.bounds, transform=transform)
        window = clamp_window(window, cols, rows)
        if window is None:
            continue

        r0 = int(window.row_off)
        c0 = int(window.col_off)
        h = int(window.height)
        w = int(window.width)
        sub_transform = rasterio.windows.transform(window, transform)
        mask = geometry_mask([geom], out_shape=(h, w), transform=sub_transform, invert=True)
        if not mask.any():
            continue

        masks.append(BlockMask(watershed_id, block_id, r0, c0, h, w, mask))

    if not masks:
        raise ValueError("No usable block masks generated")
    LOGGER.info("Prepared %d block masks", len(masks))
    return masks


def compute_peak_label_for_block(event_cube: np.ndarray, block_mask: np.ndarray) -> float:
    subset = event_cube[:, block_mask]
    if np.ma.isMaskedArray(subset):
        values = np.asarray(subset.compressed(), dtype=np.float32)
    else:
        values = np.asarray(subset, dtype=np.float32).ravel()
        values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def construct_labels(
    nc_files: Sequence[Path],
    masks: Sequence[BlockMask],
    depth_var: str,
    hydro_threshold: float,
    drop_nonhydro_blocks: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for nc_path in nc_files:
        event_id = parse_event_id_from_nc(nc_path)
        with nc4.Dataset(nc_path, "r") as ds:
            if depth_var not in ds.variables:
                raise ValueError(f"Variable '{depth_var}' missing in {nc_path}")
            var = ds.variables[depth_var]

            for mask in masks:
                cube = var[:, mask.row_off : mask.row_off + mask.height, mask.col_off : mask.col_off + mask.width]
                y_value = compute_peak_label_for_block(cube, mask.mask)
                rows.append(
                    {
                        "event_id": event_id,
                        "watershed_id": mask.watershed_id,
                        "block_id": mask.block_id,
                        "y": y_value,
                    }
                )

        LOGGER.info("Computed labels for event %s", event_id)

    labels = pd.DataFrame(rows)
    if labels.empty:
        raise ValueError("No labels computed")

    labels = labels.sort_values(["watershed_id", "event_id", "block_id"]).reset_index(drop=True)
    if labels["y"].isna().any():
        raise ValueError(f"NaN labels found: {int(labels['y'].isna().sum())}")

    if drop_nonhydro_blocks:
        max_by_block = labels.groupby(["watershed_id", "block_id"], as_index=False)["y"].max()
        hydro_blocks = set(
            max_by_block.loc[max_by_block["y"] > hydro_threshold, ["watershed_id", "block_id"]].itertuples(
                index=False, name=None
            )
        )
        labels = labels.loc[
            labels[["watershed_id", "block_id"]]
            .apply(tuple, axis=1)
            .isin(hydro_blocks)
        ].reset_index(drop=True)
        LOGGER.info("Retained hydro blocks only (> %.6f), rows=%d", hydro_threshold, len(labels))

    return labels


def validate_completeness(labels: pd.DataFrame) -> None:
    issues = []
    for watershed_id, group in labels.groupby("watershed_id", sort=False):
        events = sorted(group["event_id"].unique().tolist())
        blocks = sorted(group["block_id"].unique().tolist())
        expected = len(events) * len(blocks)
        actual = len(group)
        if actual != expected:
            issues.append(
                f"watershed={watershed_id}: expected {expected} pairs ({len(events)}x{len(blocks)}), got {actual}"
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

    missing_in_labels = events_key - labels_key
    if missing_in_labels:
        sample = sorted(list(missing_in_labels))[:10]
        raise ValueError(f"events_csv contains watershed/event not found in labels, sample: {sample}")


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

    missing_in_labels = blocks_key - labels_key
    if missing_in_labels:
        sample = sorted(list(missing_in_labels))[:10]
        raise ValueError(f"blocks_parquet contains watershed/block not found in labels, sample: {sample}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 3: Construct labels.parquet from upstream netCDF outputs")
    parser.add_argument("--netcdf-dir", type=Path, required=True)
    parser.add_argument("--netcdf-pattern", type=str, default="D*_ACC_future.nc")
    parser.add_argument("--blocks-file", type=Path, required=True)
    parser.add_argument("--watershed-id", type=str, required=True)
    parser.add_argument("--depth-var", type=str, default="output_depth")
    parser.add_argument("--nc-crs", type=str, default="EPSG:26916")
    parser.add_argument("--blocks-crs", type=str, default=None)
    parser.add_argument("--block-id-column", type=str, default=None)
    parser.add_argument("--block-id-mode", choices=["watershed_b_padded", "watershed_block", "index"], default="watershed_b_padded")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--drop-nonhydro-blocks", action="store_true")
    parser.add_argument("--hydro-threshold", type=float, default=0.0)
    parser.add_argument("--events-csv", type=Path, default=None, help="Optional events table for validation")
    parser.add_argument("--blocks-parquet", type=Path, default=None, help="Optional blocks table for validation")
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    nc_files = sorted(args.netcdf_dir.glob(args.netcdf_pattern))
    if not nc_files:
        raise FileNotFoundError(f"No netCDF files matched {args.netcdf_pattern} in {args.netcdf_dir}")
    if args.max_events is not None:
        nc_files = nc_files[: args.max_events]
    LOGGER.info("Using %d netCDF files", len(nc_files))

    transform, rows, cols = infer_transform_from_nc(nc_files[0], args.depth_var)
    masks = build_block_masks(
        blocks_file=args.blocks_file,
        watershed_id=normalize_id(args.watershed_id),
        transform=transform,
        rows=rows,
        cols=cols,
        blocks_crs=args.blocks_crs,
        nc_crs=args.nc_crs,
        block_id_column=args.block_id_column,
        block_id_mode=args.block_id_mode,
        max_blocks=args.max_blocks,
    )

    labels = construct_labels(
        nc_files=nc_files,
        masks=masks,
        depth_var=args.depth_var,
        hydro_threshold=args.hydro_threshold,
        drop_nonhydro_blocks=args.drop_nonhydro_blocks,
    )

    validate_completeness(labels)

    if args.events_csv is not None:
        validate_against_events(labels, args.events_csv)
        LOGGER.info("Validated labels against events table: %s", args.events_csv)

    if args.blocks_parquet is not None:
        validate_against_blocks(labels, args.blocks_parquet)
        LOGGER.info("Validated labels against blocks table: %s", args.blocks_parquet)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.output_parquet, index=False)
    LOGGER.info("Wrote labels.parquet rows=%d -> %s", len(labels), args.output_parquet)


if __name__ == "__main__":
    main()
