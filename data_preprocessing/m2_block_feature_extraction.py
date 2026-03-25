import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry.base import BaseGeometry


LOGGER = logging.getLogger("m2_block_feature_extraction")


FEATURE_COLUMNS = [
    "centroid_x",
    "centroid_y",
    "area",
    "mean_elevation",
    "elevation_range",
    "mean_slope",
    "distance_to_outlet",
]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_watershed_id(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def infer_watershed_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"^blocks?_", "", stem)
    stem = re.sub(r"_blocks?$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def expand_block_sources(blocks_files: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in blocks_files:
        raw_path = Path(raw)
        if any(char in raw for char in ["*", "?", "["]):
            paths.extend(sorted(raw_path.parent.glob(raw_path.name)))
        else:
            paths.append(raw_path)

    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)

    if not unique_paths:
        raise FileNotFoundError("No block shapefiles found from --blocks-files")

    return unique_paths


def ensure_crs(gdf: gpd.GeoDataFrame, crs_override: Optional[str], what: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        if not crs_override:
            raise ValueError(f"{what} has no CRS metadata; provide --{what}-crs")
        gdf = gdf.set_crs(crs_override)
        LOGGER.warning("Set missing CRS for %s using override: %s", what, crs_override)
    return gdf


def load_blocks(
    block_paths: Sequence[Path],
    watershed_id_column: str,
    block_id_column: str,
    blocks_crs: Optional[str],
) -> gpd.GeoDataFrame:
    all_frames: List[gpd.GeoDataFrame] = []

    for path in block_paths:
        frame = gpd.read_file(path)
        frame = ensure_crs(frame, blocks_crs, "blocks")

        if watershed_id_column in frame.columns:
            frame["watershed_id"] = frame[watershed_id_column].astype(str).map(normalize_watershed_id)
        else:
            inferred = infer_watershed_from_filename(path)
            if not inferred:
                raise ValueError(
                    f"Could not infer watershed_id from filename {path.name}; add column '{watershed_id_column}'"
                )
            frame["watershed_id"] = inferred

        if block_id_column in frame.columns:
            frame["block_id"] = frame[block_id_column].astype(str)
            frame["_needs_generated_block_id"] = False
        else:
            frame["block_id"] = None
            frame["_needs_generated_block_id"] = True

        if "geometry" not in frame.columns:
            raise ValueError(f"No geometry column found in blocks file: {path}")

        frame = frame[["watershed_id", "block_id", "geometry", "_needs_generated_block_id"]].copy()
        frame = frame.loc[frame.geometry.notnull()].copy()
        frame = frame.loc[~frame.geometry.is_empty].copy()

        all_frames.append(frame)
        LOGGER.info("Loaded %s blocks from %s", len(frame), path)

    merged = pd.concat(all_frames, ignore_index=True)

    mixed_watersheds = []
    for watershed_id, group in merged.groupby("watershed_id", sort=False):
        needs_generated = bool(group["_needs_generated_block_id"].any())
        has_existing = bool((~group["_needs_generated_block_id"]).any())
        if needs_generated and has_existing:
            mixed_watersheds.append(watershed_id)
    if mixed_watersheds:
        raise ValueError(
            "Some watersheds mix explicit block IDs with generated block IDs. "
            f"Provide a complete '{block_id_column}' column or omit it consistently. Affected watersheds: {mixed_watersheds}"
        )

    if bool(merged["_needs_generated_block_id"].any()):
        generated_index = merged.groupby("watershed_id", sort=False).cumcount() + 1
        merged.loc[merged["_needs_generated_block_id"], "block_id"] = [
            f"{wid}_b{idx:06d}"
            for wid, idx in zip(
                merged.loc[merged["_needs_generated_block_id"], "watershed_id"],
                generated_index.loc[merged["_needs_generated_block_id"]],
            )
        ]

    duplicates = merged.duplicated(subset=["watershed_id", "block_id"])
    if bool(duplicates.any()):
        dup_rows = merged.loc[duplicates, ["watershed_id", "block_id"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate watershed/block IDs detected in block inputs, sample: {dup_rows}")

    merged = merged[["watershed_id", "block_id", "geometry"]].copy()
    blocks_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=all_frames[0].crs)
    return blocks_gdf


def load_outlets(
    outlets_file: Path,
    watershed_id_column: str,
    outlets_crs: Optional[str],
    outlet_x_column: str,
    outlet_y_column: str,
) -> gpd.GeoDataFrame:
    suffix = outlets_file.suffix.lower()
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(outlets_file)
        required = {watershed_id_column, outlet_x_column, outlet_y_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Outlet table missing required columns: {sorted(missing)}")

        gdf = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(frame[outlet_x_column], frame[outlet_y_column]),
            crs=outlets_crs,
        )
    else:
        gdf = gpd.read_file(outlets_file)
        gdf = ensure_crs(gdf, outlets_crs, "outlets")
        if watershed_id_column not in gdf.columns:
            raise ValueError(f"Outlets file must contain watershed column '{watershed_id_column}'")

    gdf = ensure_crs(gdf, outlets_crs, "outlets")
    gdf["watershed_id"] = gdf[watershed_id_column].astype(str).map(normalize_watershed_id)
    gdf = gdf[["watershed_id", "geometry"]].copy()
    gdf = gdf.loc[gdf.geometry.notnull()].copy()
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("Outlets file produced no valid outlet points")

    LOGGER.info("Loaded %d outlet points from %s", len(gdf), outlets_file)
    return gdf


def build_slope_array(dem_array: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    dx = abs(transform.a)
    dy = abs(transform.e)
    if dx == 0 or dy == 0:
        raise ValueError("Invalid DEM transform with zero pixel size")

    grad_y, grad_x = np.gradient(dem_array, dy, dx)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    slope[~np.isfinite(dem_array)] = np.nan
    return slope.astype(np.float32)


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


def zonal_values(
    array: np.ndarray,
    transform: rasterio.Affine,
    geom: BaseGeometry,
) -> np.ndarray:
    window = from_bounds(*geom.bounds, transform=transform)
    clipped_window = clamp_window(window, width=array.shape[1], height=array.shape[0])
    if clipped_window is None:
        return np.array([], dtype=np.float32)

    r0 = int(clipped_window.row_off)
    r1 = r0 + int(clipped_window.height)
    c0 = int(clipped_window.col_off)
    c1 = c0 + int(clipped_window.width)

    subarray = array[r0:r1, c0:c1]
    sub_transform = rasterio.windows.transform(clipped_window, transform)

    in_geom = geometry_mask(
        [geom],
        out_shape=subarray.shape,
        transform=sub_transform,
        invert=True,
    )

    values = subarray[in_geom]
    values = values[np.isfinite(values)]
    return values


def compute_block_features(
    blocks_gdf: gpd.GeoDataFrame,
    outlets_gdf: gpd.GeoDataFrame,
    dem_array: np.ndarray,
    slope_array: np.ndarray,
    dem_transform: rasterio.Affine,
) -> pd.DataFrame:
    outlet_geom_by_watershed: Dict[str, BaseGeometry] = {
        watershed_id: group.geometry.union_all()
        for watershed_id, group in outlets_gdf.groupby("watershed_id", sort=False)
    }

    rows: List[Dict[str, float]] = []
    for idx, row in blocks_gdf.iterrows():
        watershed_id = row["watershed_id"]
        block_id = row["block_id"]
        geom = row.geometry

        if watershed_id not in outlet_geom_by_watershed:
            raise ValueError(f"No outlet found for watershed_id='{watershed_id}'")

        centroid = geom.centroid
        area = float(geom.area)
        elevation_values = zonal_values(dem_array, dem_transform, geom)
        slope_values = zonal_values(slope_array, dem_transform, geom)

        if elevation_values.size == 0:
            mean_elevation = np.nan
            elevation_range = np.nan
        else:
            mean_elevation = float(np.mean(elevation_values))
            elevation_range = float(np.max(elevation_values) - np.min(elevation_values))

        mean_slope = float(np.mean(slope_values)) if slope_values.size > 0 else np.nan
        distance_to_outlet = float(centroid.distance(outlet_geom_by_watershed[watershed_id]))

        rows.append(
            {
                "watershed_id": watershed_id,
                "block_id": block_id,
                "centroid_x": float(centroid.x),
                "centroid_y": float(centroid.y),
                "area": area,
                "mean_elevation": mean_elevation,
                "elevation_range": elevation_range,
                "mean_slope": mean_slope,
                "distance_to_outlet": distance_to_outlet,
            }
        )

        if (idx + 1) % 200 == 0:
            LOGGER.info("Processed %d blocks", idx + 1)

    result = pd.DataFrame(rows)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 2: Block-level feature extraction")
    parser.add_argument(
        "--blocks-files",
        nargs="+",
        required=True,
        help="One or more block polygon files or glob patterns (e.g., '/path/blocks_*.shp')",
    )
    parser.add_argument("--dem-raster", type=Path, required=True, help="DEM raster path")
    parser.add_argument(
        "--outlets-file",
        type=Path,
        required=True,
        help="Outlet points source (vector file or CSV with x/y columns)",
    )
    parser.add_argument("--output-parquet", type=Path, required=True, help="Output blocks.parquet path")

    parser.add_argument("--watershed-id-column", type=str, default="watershed_id")
    parser.add_argument("--block-id-column", type=str, default="block_id")
    parser.add_argument("--blocks-crs", type=str, default=None, help="CRS override when blocks file has no CRS")

    parser.add_argument("--outlets-crs", type=str, default=None, help="CRS override for outlets CSV or missing CRS")
    parser.add_argument("--outlet-x-column", type=str, default="X")
    parser.add_argument("--outlet-y-column", type=str, default="Y")

    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    block_paths = expand_block_sources(args.blocks_files)
    blocks = load_blocks(
        block_paths=block_paths,
        watershed_id_column=args.watershed_id_column,
        block_id_column=args.block_id_column,
        blocks_crs=args.blocks_crs,
    )

    outlets = load_outlets(
        outlets_file=args.outlets_file,
        watershed_id_column=args.watershed_id_column,
        outlets_crs=args.outlets_crs,
        outlet_x_column=args.outlet_x_column,
        outlet_y_column=args.outlet_y_column,
    )

    with rasterio.open(args.dem_raster) as dem_src:
        dem_crs = dem_src.crs
        if dem_crs is None:
            raise ValueError("DEM raster has no CRS metadata")
        if not dem_crs.is_projected:
            raise ValueError(f"DEM CRS must be projected for area/distance features; got {dem_crs}")

        dem_array = dem_src.read(1).astype(np.float32)
        dem_transform = dem_src.transform
        nodata = dem_src.nodata
        if nodata is not None:
            dem_array[dem_array == nodata] = np.nan
        dem_array[~np.isfinite(dem_array)] = np.nan
        dem_array[dem_array > 1e30] = np.nan

    LOGGER.info("DEM loaded from %s with shape=%s and CRS=%s", args.dem_raster, dem_array.shape, dem_crs)

    blocks = blocks.to_crs(dem_crs)
    outlets = outlets.to_crs(dem_crs)
    LOGGER.info("Reprojected blocks and outlets to DEM CRS")

    slope_array = build_slope_array(dem_array, dem_transform)
    LOGGER.info("Computed slope raster")

    block_features = compute_block_features(
        blocks_gdf=blocks,
        outlets_gdf=outlets,
        dem_array=dem_array,
        slope_array=slope_array,
        dem_transform=dem_transform,
    )

    expected = ["watershed_id", "block_id", *FEATURE_COLUMNS]
    block_features = block_features[expected].copy()
    block_features = block_features.sort_values(["watershed_id", "block_id"]).reset_index(drop=True)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        block_features.to_parquet(args.output_parquet, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet support is unavailable. Install 'pyarrow' (preferred) or 'fastparquet' "
            "in the active Python environment to write blocks.parquet."
        ) from exc
    LOGGER.info("Wrote %d block rows to %s", len(block_features), args.output_parquet)


if __name__ == "__main__":
    main()
