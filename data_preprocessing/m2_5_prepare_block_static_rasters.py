import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


LOGGER = logging.getLogger("m2_5_prepare_block_static_rasters")


@dataclass
class BlockWindow:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    height: int
    width: int


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_kv_specs(raw_specs: Optional[Sequence[str]], name: str) -> Dict[str, Path]:
    specs: Dict[str, Path] = {}
    for raw in raw_specs or []:
        if "=" not in raw:
            raise ValueError(f"Invalid {name} spec '{raw}'. Expected format: key=/path/to/file")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid {name} spec '{raw}'. Empty key")
        if not value:
            raise ValueError(f"Invalid {name} spec '{raw}'. Empty path")
        specs[key] = Path(value).expanduser().resolve()
    return specs


def load_labels_metadata(labels_10m_dir: Path) -> Tuple[Affine, str, int, int]:
    metadata_path = labels_10m_dir / "labels_10m_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    transform_values = metadata.get("transform")
    if not transform_values or len(transform_values) != 6:
        raise ValueError(f"Expected 6-value affine transform in {metadata_path}")

    transform = Affine(*[float(v) for v in transform_values])
    dst_crs = str(metadata["nc_crs"])
    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    return transform, dst_crs, rows, cols


def load_aligned_raster(
    raster_path: Path,
    dst_shape: Tuple[int, int],
    dst_transform: Affine,
    dst_crs: str,
    categorical: bool,
    raster_crs_override: Optional[str],
) -> np.ndarray:
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        src_array = src.read(1).astype(np.float32)
        src_crs = src.crs
        if src_crs is None:
            if raster_crs_override is None:
                raise ValueError(
                    f"Raster {raster_path} has no CRS. Pass --raster-crs-override to set source CRS explicitly"
                )
            src_crs = raster_crs_override

        src_nodata = src.nodata
        if src_nodata is not None:
            src_array = np.where(src_array == src_nodata, np.nan, src_array)

        dst_array = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )

    return dst_array


def compute_block_windows(block_index_grid: np.ndarray, block_lookup_df: pd.DataFrame) -> Dict[int, BlockWindow]:
    valid = block_index_grid >= 0
    rows, cols = np.nonzero(valid)
    indices = block_index_grid[valid].astype(np.int64)

    if len(indices) == 0:
        raise ValueError("block_index_10m.npy contains no valid cells")

    max_index = int(block_lookup_df["block_index"].max())
    row_min = np.full(max_index + 1, block_index_grid.shape[0], dtype=np.int32)
    row_max = np.full(max_index + 1, -1, dtype=np.int32)
    col_min = np.full(max_index + 1, block_index_grid.shape[1], dtype=np.int32)
    col_max = np.full(max_index + 1, -1, dtype=np.int32)

    np.minimum.at(row_min, indices, rows)
    np.maximum.at(row_max, indices, rows)
    np.minimum.at(col_min, indices, cols)
    np.maximum.at(col_max, indices, cols)

    windows: Dict[int, BlockWindow] = {}
    for row in block_lookup_df.itertuples(index=False):
        block_index = int(row.block_index)
        if row_max[block_index] < 0:
            raise ValueError(f"Block index {block_index} has zero cells in block_index_10m.npy")

        windows[block_index] = BlockWindow(
            row_start=int(row_min[block_index]),
            row_stop=int(row_max[block_index] + 1),
            col_start=int(col_min[block_index]),
            col_stop=int(col_max[block_index] + 1),
            height=int(row_max[block_index] - row_min[block_index] + 1),
            width=int(col_max[block_index] - col_min[block_index] + 1),
        )

    return windows


def center_pad(patch: np.ndarray, out_rows: int, out_cols: int, fill_value: float = 0.0) -> np.ndarray:
    out = np.full((out_rows, out_cols), fill_value, dtype=np.float32)
    row0 = (out_rows - patch.shape[0]) // 2
    col0 = (out_cols - patch.shape[1]) // 2
    out[row0 : row0 + patch.shape[0], col0 : col0 + patch.shape[1]] = patch
    return out


def add_distance_to_stream(stream_mask: np.ndarray, xres: float, yres: float) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError(
            "scipy is required to compute distance-to-stream. Install scipy or provide precomputed dist raster"
        ) from exc

    stream_bool = np.isfinite(stream_mask) & (stream_mask > 0.5)
    if not np.any(stream_bool):
        raise ValueError("Stream mask has no positive cells after alignment")

    distance = distance_transform_edt(~stream_bool, sampling=(yres, xres)).astype(np.float32)
    distance[~np.isfinite(stream_mask)] = np.nan
    return distance


def save_feature_names(path: Path, names: List[str]) -> None:
    payload = {"feature_names": names}
    path.write_text(json.dumps(payload, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 2.5: Build aligned 10m raster-level static features and block-local 80x80 feature tensors"
        )
    )
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--static-raster",
        action="append",
        default=[],
        help="Static raster layer as name=/path/to/raster.tif. Repeat for multiple layers",
    )
    parser.add_argument(
        "--categorical-layer",
        action="append",
        default=[],
        help="Layer names to resample with nearest-neighbor (e.g., stream_mask)",
    )
    parser.add_argument(
        "--raster-crs-override",
        type=str,
        default=None,
        help="Use this CRS for input rasters missing CRS metadata",
    )

    parser.add_argument(
        "--stream-mask-raster",
        type=Path,
        default=None,
        help="Optional stream mask raster; used to derive distance-to-stream",
    )
    parser.add_argument(
        "--stream-mask-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to aligned stream mask before distance transform",
    )
    parser.add_argument(
        "--add-distance-to-stream",
        action="store_true",
        help="Derive distance-to-stream layer from stream mask",
    )
    parser.add_argument(
        "--distance-layer-name",
        type=str,
        default="distance_to_stream",
    )

    parser.add_argument(
        "--add-relative-elevation",
        action="store_true",
        help="Add block-normalized DEM channel: dem - mean(dem over block cells)",
    )
    parser.add_argument(
        "--dem-layer-name",
        type=str,
        default="dem",
        help="Layer name used as DEM source for relative elevation and slope",
    )
    parser.add_argument(
        "--relative-elevation-name",
        type=str,
        default="relative_elevation",
    )
    parser.add_argument(
        "--add-slope",
        action="store_true",
        help="Derive 10m slope raster from DEM using finite differences",
    )
    parser.add_argument(
        "--slope-layer-name",
        type=str,
        default="slope",
    )

    parser.add_argument("--target-rows", type=int, default=80)
    parser.add_argument("--target-cols", type=int, default=80)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    labels_10m_dir = args.labels_10m_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    block_index_path = labels_10m_dir / "block_index_10m.npy"
    block_lookup_path = labels_10m_dir / "block_index_lookup.parquet"

    if not block_index_path.exists():
        raise FileNotFoundError(f"Missing {block_index_path}")
    if not block_lookup_path.exists():
        raise FileNotFoundError(f"Missing {block_lookup_path}")

    block_index_grid = np.load(block_index_path)
    block_lookup_df = pd.read_parquet(block_lookup_path)
    if "block_index" not in block_lookup_df.columns:
        raise ValueError("block_index_lookup.parquet must include column 'block_index'")

    dst_transform, dst_crs, dst_rows, dst_cols = load_labels_metadata(labels_10m_dir)
    if block_index_grid.shape != (dst_rows, dst_cols):
        raise ValueError(
            "Grid shape mismatch between labels_10m_metadata.json and block_index_10m.npy: "
            f"metadata=({dst_rows}, {dst_cols}) block_index={block_index_grid.shape}"
        )

    categorical_layers = set(args.categorical_layer)
    layer_specs = parse_kv_specs(args.static_raster, "static-raster")

    if args.stream_mask_raster is not None:
        layer_specs.setdefault("stream_mask", args.stream_mask_raster.resolve())
        categorical_layers.add("stream_mask")

    if not layer_specs:
        raise ValueError("No static raster layers provided. Use --static-raster name=/path/to/raster.tif")

    LOGGER.info("Loading and aligning %d static raster layers", len(layer_specs))
    layer_grids: Dict[str, np.ndarray] = {}
    for layer_name, layer_path in layer_specs.items():
        layer_grids[layer_name] = load_aligned_raster(
            raster_path=layer_path,
            dst_shape=(dst_rows, dst_cols),
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            categorical=layer_name in categorical_layers,
            raster_crs_override=args.raster_crs_override,
        )
        LOGGER.info("Aligned layer %s from %s", layer_name, layer_path)

    if args.add_distance_to_stream:
        if "stream_mask" not in layer_grids:
            raise ValueError(
                "--add-distance-to-stream requires stream mask. Provide --stream-mask-raster or --static-raster stream_mask=..."
            )
        stream_mask = np.where(np.isfinite(layer_grids["stream_mask"]), layer_grids["stream_mask"], np.nan)
        stream_mask = np.where(stream_mask > args.stream_mask_threshold, 1.0, 0.0).astype(np.float32)
        layer_grids["stream_mask"] = stream_mask

        xres = abs(float(dst_transform.a))
        yres = abs(float(dst_transform.e))
        layer_grids[args.distance_layer_name] = add_distance_to_stream(stream_mask, xres=xres, yres=yres)
        LOGGER.info("Derived layer %s from stream mask", args.distance_layer_name)

    if args.add_slope:
        if args.dem_layer_name not in layer_grids:
            raise ValueError(
                f"--add-slope requested but DEM layer '{args.dem_layer_name}' is missing"
            )
        dem_full = layer_grids[args.dem_layer_name]
        xres = abs(float(dst_transform.a))
        yres = abs(float(dst_transform.e))
        grad_y, grad_x = np.gradient(np.where(np.isfinite(dem_full), dem_full, 0.0), yres, xres)
        slope_grid = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)
        slope_grid[~np.isfinite(dem_full)] = np.nan
        layer_grids[args.slope_layer_name] = slope_grid
        LOGGER.info("Derived slope layer from %s", args.dem_layer_name)

    feature_names = list(layer_grids.keys())
    if args.add_relative_elevation:
        if args.dem_layer_name not in layer_grids:
            raise ValueError(
                f"--add-relative-elevation requested but DEM layer '{args.dem_layer_name}' is missing"
            )
        feature_names.append(args.relative_elevation_name)

    windows = compute_block_windows(block_index_grid=block_index_grid, block_lookup_df=block_lookup_df)

    too_large = [
        block_index
        for block_index, window in windows.items()
        if window.height > args.target_rows or window.width > args.target_cols
    ]
    if too_large:
        raise ValueError(
            f"Some blocks exceed target shape {args.target_rows}x{args.target_cols}. Example block_index={too_large[:5]}"
        )

    n_blocks = len(block_lookup_df)
    n_channels = len(feature_names)
    output_path = output_dir / "block_static_features.npy"
    feature_tensor = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_blocks, n_channels, args.target_rows, args.target_cols),
    )

    mask_tensor_path = output_dir / "block_static_masks.npy"
    mask_tensor = np.lib.format.open_memmap(
        mask_tensor_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_blocks, args.target_rows, args.target_cols),
    )

    channel_sum = np.zeros(n_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(n_channels, dtype=np.float64)
    channel_count = np.zeros(n_channels, dtype=np.int64)

    ordered_base_layers = list(layer_grids.keys())
    for row in block_lookup_df.itertuples(index=False):
        block_index = int(row.block_index)
        window = windows[block_index]

        block_mask = (
            block_index_grid[window.row_start : window.row_stop, window.col_start : window.col_stop] == block_index
        ).astype(np.float32)
        padded_mask = center_pad(block_mask, args.target_rows, args.target_cols, fill_value=0.0)
        mask_tensor[block_index] = padded_mask

        channel_maps: List[np.ndarray] = []
        for layer_name in ordered_base_layers:
            layer_patch = layer_grids[layer_name][window.row_start : window.row_stop, window.col_start : window.col_stop]
            layer_patch = np.where(np.isfinite(layer_patch), layer_patch, args.fill_value).astype(np.float32)
            layer_patch = layer_patch * block_mask
            channel_maps.append(layer_patch)

        if args.add_relative_elevation:
            dem_patch = layer_grids[args.dem_layer_name][window.row_start : window.row_stop, window.col_start : window.col_stop]
            valid = np.isfinite(dem_patch) & (block_mask > 0.5)
            if np.any(valid):
                mean_dem = float(np.mean(dem_patch[valid]))
                rel_patch = np.where(np.isfinite(dem_patch), dem_patch - mean_dem, args.fill_value).astype(np.float32)
            else:
                rel_patch = np.full_like(block_mask, args.fill_value, dtype=np.float32)
            rel_patch = rel_patch * block_mask
            channel_maps.append(rel_patch)

        for channel_index, channel_patch in enumerate(channel_maps):
            padded = center_pad(channel_patch, args.target_rows, args.target_cols, fill_value=args.fill_value)
            feature_tensor[block_index, channel_index] = padded

            valid_cells = padded_mask > 0.5
            values = padded[valid_cells]
            channel_sum[channel_index] += float(np.sum(values, dtype=np.float64))
            channel_sq_sum[channel_index] += float(np.sum(values.astype(np.float64) ** 2))
            channel_count[channel_index] += int(values.size)

    feature_tensor.flush()
    mask_tensor.flush()

    block_lookup_df.to_parquet(output_dir / "block_static_lookup.parquet", index=False)
    save_feature_names(output_dir / "block_static_feature_names.json", feature_names)

    means = channel_sum / np.maximum(channel_count, 1)
    variances = channel_sq_sum / np.maximum(channel_count, 1) - means ** 2
    stds = np.sqrt(np.maximum(variances, 0.0))
    stats = {
        "feature_names": feature_names,
        "mean": [float(v) for v in means],
        "std": [float(v) for v in stds],
        "count": [int(v) for v in channel_count],
        "target_rows": int(args.target_rows),
        "target_cols": int(args.target_cols),
        "fill_value": float(args.fill_value),
        "labels_10m_dir": str(labels_10m_dir),
    }
    (output_dir / "block_static_feature_stats.json").write_text(json.dumps(stats, indent=2))

    LOGGER.info("Wrote static feature tensor: %s (shape=%s)", output_path, feature_tensor.shape)
    LOGGER.info("Wrote static mask tensor: %s (shape=%s)", mask_tensor_path, mask_tensor.shape)
    LOGGER.info("Wrote feature names: %s", output_dir / "block_static_feature_names.json")
    LOGGER.info("Wrote feature stats: %s", output_dir / "block_static_feature_stats.json")


if __name__ == "__main__":
    main()
