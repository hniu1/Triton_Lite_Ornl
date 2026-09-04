#!/usr/bin/env python3
"""Generate event netCDF files directly from raw Triton Lite ZIP outputs."""

import argparse
import glob
import logging
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import netCDF4 as nc4
import numpy as np


LOGGER = logging.getLogger("m0_generate_netcdf_from_zip")


CONASAUGA_DEFAULT_ZIP_DIR = (
    "/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/"
    "4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/"
    "ACCESS_RegCM_baseline_flood_3hr"
)
CONASAUGA_DEFAULT_COLS = 5474
CONASAUGA_DEFAULT_ROWS = 7976
CONASAUGA_DEFAULT_GT = (
    676222.46162369,
    9.50608224323499,
    0.0,
    3891564.837097742,
    0.0,
    -9.506082243234989,
)
CONASAUGA_DEFAULT_REF_TIME_UNITS = "hours since 2013-02-02 03:00:00 -5:00"
CONASAUGA_DEFAULT_TIME_ORIGIN = "1966-02-02T03:00:00-05:00"
CONASAUGA_DEFAULT_CRS_WKT = (
    'PROJCS["NAD_1983_UTM_Zone_16N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],'
    'AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-87],PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["EPSG","26916"]]'
)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_time_origin(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


# Maps output_type prefix to netCDF variable name and long_name
OUTPUT_TYPE_META: Dict[str, Dict[str, str]] = {
    "H": {"varname": "output_depth", "long_name": "flood_depth", "units": "m", "standard_name": "Depth"},
    # Legacy variable names are retained for compatibility. The archived
    # TRITON source proves that U/V are the conserved HU/HV fields, i.e.
    # depth-integrated momentum (unit discharge), rather than velocity.
    "U": {"varname": "output_velocity_x", "long_name": "flood_unit_discharge_x", "units": "m2 s-1", "standard_name": "UnitDischargeX"},
    "V": {"varname": "output_velocity_y", "long_name": "flood_unit_discharge_y", "units": "m2 s-1", "standard_name": "UnitDischargeY"},
}


def unzip_selected_multi(zip_path: Path, output_path: Path, output_types: Sequence[str]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for output_type in output_types:
            prefix = f"{output_type}_"
            selected = [
                name for name in zf.namelist()
                if name.split("/")[-1].startswith(prefix) and name.endswith(".dat")
            ]
            for name in selected:
                zf.extract(name, output_path)
            LOGGER.info("Extracted %d %s*.dat files from %s", len(selected), output_type, zip_path)


def get_output_files(dat_path: Path, output_type: str) -> List[Path]:
    files = sorted(
        Path(p) for p in glob.glob(str(dat_path / "*" / "output" / "flood2d" / "bin" / f"{output_type}_*.dat"))
    )
    LOGGER.info("Found %d binary output files for type %s", len(files), output_type)
    return files


def bin2array(bin_file: Path, rows: int, cols: int, threshold: float, pad: int) -> np.ndarray:
    full_rows = rows + 2 * pad
    full_cols = cols + 2 * pad
    expected_cells = full_rows * full_cols
    file_size = bin_file.stat().st_size

    if file_size == expected_cells * 4:
        dtype = np.float32
    elif file_size == expected_cells * 8:
        dtype = np.float64
    else:
        raise ValueError(
            f"Unexpected file size for {bin_file}: {file_size} bytes; "
            f"expected {expected_cells * 4} (float32) or {expected_cells * 8} (float64)"
        )

    array = np.fromfile(bin_file, dtype=dtype)
    array = array.reshape((full_rows, full_cols)).astype(np.float32, copy=False)
    array = array[pad : rows + pad, pad : cols + pad]
    array = np.where(array > threshold, array, np.nan)
    return array


def build_coordinate_arrays(cols: int, rows: int, gt: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xres, _, ymax, _, yres = gt
    xarr = np.array([xmin + xres * 0.5 + i * xres for i in range(cols)], dtype=np.float64)
    yarr = np.array([ymax + yres * 0.5 + i * yres for i in range(rows)], dtype=np.float64)
    return xarr, yarr


def add_crs_variable(nc: nc4.Dataset, crs_wkt: str) -> str:
    grid_mapping_name = "transverse_mercator"
    crs = nc.createVariable(grid_mapping_name, "i2")
    crs.grid_mapping_name = grid_mapping_name
    crs.false_easting = 500000.0
    crs.false_northing = 0.0
    crs.inverse_flattening = 298.257222101004
    crs.crs_wkt = crs_wkt
    crs.longitude_of_central_meridian = -87.0
    crs.latitude_of_projection_origin = 0.0
    crs.semi_major_axis = 6378137.0
    crs.semi_minor_axis = 6356752.314140356
    crs.scale_factor_at_central_meridian = 0.9996
    return grid_mapping_name


def add_time_variable(
    nc: nc4.Dataset,
    dim_name: str,
    ref_time_units: str,
    time_origin: datetime,
    n_steps: int,
    interval_sec: int,
    start_offset_steps: int,
    bound_width_hr: float,
) -> None:
    nc.createDimension(dim_name, n_steps)
    time_var = nc.createVariable(dim_name, "f4", (dim_name,))
    time_var.standard_name = "time"
    time_var.long_name = "time"
    time_var.calendar = "standard"
    time_var.units = ref_time_units
    time_var.bounds = f"{dim_name}_bnds"

    bounds = nc.createVariable(f"{dim_name}_bnds", "f4", (dim_name, "nv"))
    bounds.units = ref_time_units

    for i in range(n_steps):
        date = time_origin + timedelta(seconds=(i + start_offset_steps) * interval_sec)
        delta = date - time_origin
        hours = delta.days * 24 + delta.seconds / 3600
        time_var[i] = hours
        bounds[i] = [hours - bound_width_hr, hours + bound_width_hr]


def create_netcdf(
    output_path: Path,
    output_files_by_type: Dict[str, List[Path]],
    cols: int,
    rows: int,
    gt: Sequence[float],
    ref_time_units: str,
    time_origin: datetime,
    threshold: float,
    pad: int,
    in_steps: int,
    in_interval_sec: int,
    out_interval_sec: int,
    crs_wkt: str,
    compression_level: int,
    sync_every: int,
) -> None:
    """Write all output types (H, U, V, ...) into one netCDF file.

    Performance notes:
    - Lower compression_level speeds up writing on large events.
    - sync_every=0 disables periodic sync and only flushes when file is closed.
    """
    # Use depth (H) file count for out_time dimension; all types must have same count.
    primary_type = next(iter(output_files_by_type))
    n_out_steps = len(output_files_by_type[primary_type])
    xarr, yarr = build_coordinate_arrays(cols, rows, gt)

    with nc4.Dataset(output_path, "w", format="NETCDF4", diskless=False) as nc:
        nc.Conventions = "CF-1.6"
        nc.createDimension("x", cols)
        nc.createDimension("y", rows)
        nc.createDimension("nv", 2)

        x = nc.createVariable("x", "f8", ("x",))
        x.units = "m"
        x.long_name = "x coordinate of projection"
        x.standard_name = "projection_x_coordinate"
        x[:] = xarr

        y = nc.createVariable("y", "f8", ("y",))
        y.units = "m"
        y.long_name = "y coordinate of projection"
        y.standard_name = "projection_y_coordinate"
        y[:] = yarr

        grid_mapping_name = add_crs_variable(nc, crs_wkt)

        add_time_variable(
            nc,
            dim_name="in_time",
            ref_time_units=ref_time_units,
            time_origin=time_origin,
            n_steps=in_steps,
            interval_sec=in_interval_sec,
            start_offset_steps=0,
            bound_width_hr=0.5,
        )
        add_time_variable(
            nc,
            dim_name="out_time",
            ref_time_units=ref_time_units,
            time_origin=time_origin,
            n_steps=n_out_steps,
            interval_sec=out_interval_sec,
            start_offset_steps=1,
            bound_width_hr=0.25,
        )

        for output_type, output_files in output_files_by_type.items():
            meta = OUTPUT_TYPE_META.get(output_type, {
                "varname": f"output_{output_type.lower()}",
                "long_name": f"flood_{output_type.lower()}",
                "units": "unknown",
                "standard_name": output_type,
            })
            use_compression = compression_level > 0
            nc_var = nc.createVariable(
                meta["varname"],
                "f4",
                ("out_time", "y", "x"),
                zlib=use_compression,
                complevel=max(1, compression_level) if use_compression else 1,
                fill_value=-9999.0,
            )
            nc_var.grid_mapping = grid_mapping_name
            nc_var.standard_name = meta["standard_name"]
            nc_var.long_name = f"{meta['long_name']}_{output_path.name}"
            nc_var.units = meta["units"]
            if output_type in {"U", "V"}:
                nc_var.triton_component_semantics = "unit_discharge"
                nc_var.legacy_variable_name = "true"

            if len(output_files) != n_out_steps:
                raise ValueError(
                    f"Output type '{output_type}' has {len(output_files)} files but primary type "
                    f"'{primary_type}' has {n_out_steps}; counts must match."
                )

            for i, output_file in enumerate(output_files):
                arr = bin2array(output_file, rows=rows, cols=cols, threshold=threshold, pad=pad)
                arr[np.isnan(arr)] = -9999.0
                nc_var[i, :, :] = arr
                if (i + 1) % 50 == 0 or (i + 1) == n_out_steps:
                    LOGGER.info(
                        "Wrote %d/%d slices for var '%s' to %s",
                        i + 1, n_out_steps, meta["varname"], output_path,
                    )
                if sync_every > 0 and (i + 1) % sync_every == 0:
                    nc.sync()

        if sync_every > 0:
            nc.sync()


def process_zip(
    zip_path: Path,
    output_dir: Path,
    output_types: Sequence[str],
    cols: int,
    rows: int,
    gt: Sequence[float],
    ref_time_units: str,
    time_origin: datetime,
    threshold: float,
    pad: int,
    in_steps: int,
    in_interval_sec: int,
    out_interval_sec: int,
    crs_wkt: str,
    compression_level: int,
    sync_every: int,
    temp_root_dir: Path | None,
) -> Path:
    event_id = zip_path.stem
    nc_name = f"{event_id}_ACC_future.nc"
    output_path = output_dir / nc_name

    # Use configurable temp root so large unzip staging does not exhaust node-local /tmp.
    with tempfile.TemporaryDirectory(
        prefix=f"{event_id}_",
        dir=str(temp_root_dir) if temp_root_dir is not None else None,
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        unzip_selected_multi(zip_path, temp_dir, output_types)
        output_files_by_type: Dict[str, List[Path]] = {}
        for output_type in output_types:
            files = get_output_files(temp_dir, output_type)
            if not files:
                raise FileNotFoundError(
                    f"No {output_type}_*.dat files found after extracting {zip_path}"
                )
            output_files_by_type[output_type] = files
        create_netcdf(
            output_path=output_path,
            output_files_by_type=output_files_by_type,
            cols=cols,
            rows=rows,
            gt=gt,
            ref_time_units=ref_time_units,
            time_origin=time_origin,
            threshold=threshold,
            pad=pad,
            in_steps=in_steps,
            in_interval_sec=in_interval_sec,
            out_interval_sec=out_interval_sec,
            crs_wkt=crs_wkt,
            compression_level=compression_level,
            sync_every=sync_every,
        )

    LOGGER.info("Saved netCDF: %s", output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate event netCDF files from raw Triton Lite ZIP outputs")
    parser.add_argument("--zip-dir", type=Path, default=Path(CONASAUGA_DEFAULT_ZIP_DIR))
    parser.add_argument("--zip-pattern", type=str, default="D*.zip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-types",
        nargs="+",
        default=["H"],
        help="Output type prefixes to extract and write; e.g. H U V (default: H only for backward compat)",
    )
    parser.add_argument("--cols", type=int, default=CONASAUGA_DEFAULT_COLS)
    parser.add_argument("--rows", type=int, default=CONASAUGA_DEFAULT_ROWS)
    parser.add_argument("--gt", nargs=6, type=float, default=list(CONASAUGA_DEFAULT_GT))
    parser.add_argument("--ref-time-units", type=str, default=CONASAUGA_DEFAULT_REF_TIME_UNITS)
    parser.add_argument("--time-origin", type=str, default=CONASAUGA_DEFAULT_TIME_ORIGIN)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--in-steps", type=int, default=80)
    parser.add_argument("--in-interval-sec", type=int, default=10800)
    parser.add_argument("--out-interval-sec", type=int, default=1800)
    parser.add_argument("--crs-wkt", type=str, default=CONASAUGA_DEFAULT_CRS_WKT)
    parser.add_argument(
        "--temp-root-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary unzip staging (recommended on large runs)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="NetCDF zlib compression level (0 disables compression; faster write at cost of larger files)",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=0,
        help="Call nc.sync() every N slices per variable (0 disables periodic sync for faster IO)",
    )
    parser.add_argument("--min-event-index", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root_dir = args.temp_root_dir.resolve() if args.temp_root_dir is not None else None
    if temp_root_dir is not None:
        temp_root_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(args.zip_dir.glob(args.zip_pattern))
    if args.min_event_index is not None:
        filtered = []
        for path in zip_files:
            stem = path.stem
            if len(stem) >= 4 and stem[0].upper() == "D" and stem[1:4].isdigit() and int(stem[1:4]) >= args.min_event_index:
                filtered.append(path)
        zip_files = filtered
    if args.max_events is not None:
        zip_files = zip_files[: args.max_events]
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files matched pattern {args.zip_pattern} in {args.zip_dir}")

    time_origin = parse_time_origin(args.time_origin)

    for zip_path in zip_files:
        output_path = output_dir / f"{zip_path.stem}_ACC_future.nc"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("Skipping existing netCDF: %s", output_path)
            continue
        process_zip(
            zip_path=zip_path,
            output_dir=output_dir,
            output_types=args.output_types,
            cols=args.cols,
            rows=args.rows,
            gt=args.gt,
            ref_time_units=args.ref_time_units,
            time_origin=time_origin,
            threshold=args.threshold,
            pad=args.pad,
            in_steps=args.in_steps,
            in_interval_sec=args.in_interval_sec,
            out_interval_sec=args.out_interval_sec,
            crs_wkt=args.crs_wkt,
            compression_level=args.compression_level,
            sync_every=args.sync_every,
            temp_root_dir=temp_root_dir,
        )


if __name__ == "__main__":
    main()
