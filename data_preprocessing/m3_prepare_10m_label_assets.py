import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import Affine


LOGGER = logging.getLogger("m3_prepare_10m_label_assets")


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


def compute_event_peak_grid(var: nc4.Variable, time_chunk_size: int = 4) -> np.ndarray:
    peak_grid: Optional[np.ndarray] = None
    for time_start in range(0, var.shape[0], time_chunk_size):
        time_stop = min(var.shape[0], time_start + time_chunk_size)
        frame = var[time_start:time_stop, :, :]
        if np.ma.isMaskedArray(frame):
            frame_values = frame.filled(np.nan).astype(np.float32, copy=False)
        else:
            frame_values = np.asarray(frame, dtype=np.float32)
            frame_values[~np.isfinite(frame_values)] = np.nan

        frame_peak = np.fmax.reduce(frame_values, axis=0)
        if peak_grid is None:
            peak_grid = frame_peak.copy()
            continue
        np.fmax(peak_grid, frame_peak, out=peak_grid)

    if peak_grid is None:
        raise ValueError("Encountered netCDF depth variable with zero timesteps")
    return peak_grid


def build_block_index_raster(
    blocks_file: Path,
    transform: Affine,
    rows: int,
    cols: int,
    nc_crs: str,
    blocks_crs: Optional[str],
    watershed_id: str,
    block_id_column: Optional[str],
    block_id_mode: str,
    max_blocks: Optional[int],
) -> Tuple[np.ndarray, pd.DataFrame]:
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

    block_ids: List[str] = []
    shapes: List[Tuple[object, int]] = []
    for i, row in gdf.iterrows():
        if block_id_column and block_id_column in row.index:
            block_id = str(row[block_id_column])
        elif block_id_mode == "watershed_b_padded":
            block_id = f"{watershed_id}_b{i+1:06d}"
        elif block_id_mode == "watershed_block":
            block_id = f"{watershed_id}_block_{i}"
        elif block_id_mode == "index":
            block_id = str(i)
        else:
            raise ValueError(f"Unsupported block_id_mode: {block_id_mode}")

        block_ids.append(block_id)
        shapes.append((row.geometry, i))

    if not shapes:
        raise ValueError("No usable block geometries found")

    block_index = rasterize(
        shapes=shapes,
        out_shape=(rows, cols),
        transform=transform,
        fill=-1,
        dtype=np.int32,
        all_touched=False,
    )

    cells_per_block = np.bincount(block_index[block_index >= 0], minlength=len(block_ids)).astype(np.int64)
    block_lookup = pd.DataFrame(
        {
            "watershed_id": [watershed_id] * len(block_ids),
            "block_id": block_ids,
            "block_index": np.arange(len(block_ids), dtype=np.int32),
            "n_cells_10m": cells_per_block,
        }
    )
    return block_index, block_lookup


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 3 (10m assets): prepare event peak rasters and block-index raster aligned to netCDF grid"
    )
    parser.add_argument("--netcdf-dir", type=Path, required=True)
    parser.add_argument("--netcdf-pattern", type=str, default="D*_ACC_future.nc")
    parser.add_argument("--depth-var", type=str, default="output_depth")
    parser.add_argument("--max-events", type=int, default=None)

    parser.add_argument("--blocks-file", type=Path, required=True)
    parser.add_argument("--watershed-id", type=str, required=True)
    parser.add_argument("--nc-crs", type=str, default="EPSG:26916")
    parser.add_argument("--blocks-crs", type=str, default=None)
    parser.add_argument("--block-id-column", type=str, default=None)
    parser.add_argument(
        "--block-id-mode",
        choices=["watershed_b_padded", "watershed_block", "index"],
        default="watershed_b_padded",
    )
    parser.add_argument("--max-blocks", type=int, default=None)

    parser.add_argument("--output-dir", type=Path, required=True)
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
    watershed_id = normalize_id(args.watershed_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks_dir = args.output_dir / "events_peak_10m"
    peaks_dir.mkdir(parents=True, exist_ok=True)

    block_index, block_lookup = build_block_index_raster(
        blocks_file=args.blocks_file,
        transform=transform,
        rows=rows,
        cols=cols,
        nc_crs=args.nc_crs,
        blocks_crs=args.blocks_crs,
        watershed_id=watershed_id,
        block_id_column=args.block_id_column,
        block_id_mode=args.block_id_mode,
        max_blocks=args.max_blocks,
    )

    np.save(args.output_dir / "block_index_10m.npy", block_index)
    block_lookup.to_parquet(args.output_dir / "block_index_lookup.parquet", index=False)
    LOGGER.info("Wrote block index raster and lookup (%d blocks)", len(block_lookup))

    manifest_rows: List[Dict[str, object]] = []
    for nc_path in nc_files:
        event_id = parse_event_id_from_nc(nc_path)
        with nc4.Dataset(nc_path, "r") as ds:
            if args.depth_var not in ds.variables:
                raise ValueError(f"Variable '{args.depth_var}' missing in {nc_path}")
            peak_grid = compute_event_peak_grid(ds.variables[args.depth_var])

        peak_path = peaks_dir / f"{event_id}_peak_10m.npy"
        np.save(peak_path, peak_grid)
        manifest_rows.append(
            {
                "event_id": event_id,
                "watershed_id": watershed_id,
                "path_to_peak_10m": str(peak_path),
                "rows": int(peak_grid.shape[0]),
                "cols": int(peak_grid.shape[1]),
            }
        )
        LOGGER.info("Wrote 10m peak raster for event %s -> %s", event_id, peak_path)

    manifest = pd.DataFrame(manifest_rows).sort_values(["watershed_id", "event_id"]).reset_index(drop=True)
    manifest.to_parquet(args.output_dir / "labels_10m_manifest.parquet", index=False)

    metadata = {
        "watershed_id": watershed_id,
        "nc_crs": args.nc_crs,
        "depth_var": args.depth_var,
        "rows": int(rows),
        "cols": int(cols),
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
    }
    (args.output_dir / "labels_10m_metadata.json").write_text(json.dumps(metadata, indent=2))

    LOGGER.info("Wrote 10m label asset manifest rows=%d -> %s", len(manifest), args.output_dir / "labels_10m_manifest.parquet")


if __name__ == "__main__":
    main()