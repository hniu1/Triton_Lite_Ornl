#!/usr/bin/env python3
"""Build a lightweight manifest for timestamp-conditioned TRITON training.

The dynamic fields remain in their source netCDF files.  This script validates
the data contract and records enough metadata for streaming block/time slices
during training; it intentionally does not materialize timestamp patches.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import netCDF4 as nc4
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage-1 dynamic netCDF manifest")
    parser.add_argument("--netcdf-dir", type=Path, required=True)
    parser.add_argument("--netcdf-pattern", default="D*.nc")
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--labels-10m-dir", type=Path, required=True)
    parser.add_argument("--watershed-id", default="conasauga")
    parser.add_argument("--depth-var", default="output_depth")
    parser.add_argument("--component-x-var", default="output_velocity_x")
    parser.add_argument("--component-y-var", default="output_velocity_y")
    parser.add_argument("--time-var", default="out_time")
    parser.add_argument(
        "--component-semantics",
        choices=["velocity", "unit_discharge", "unknown"],
        default="unit_discharge",
        help="Physical meaning of the two signed component variables. Audit this before publication.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip unreadable/incomplete netCDF files and record them in rejected_events.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def event_id_from_path(path: Path) -> str:
    match = re.search(r"(D\d{3})", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse D### event ID from {path}")
    return match.group(1).upper()


def inspect_file(
    path: Path,
    event_id: str,
    watershed_id: str,
    variable_names: Dict[str, str],
    expected_grid_shape: tuple,
) -> Dict[str, object]:
    with nc4.Dataset(path, "r") as ds:
        missing = [name for name in variable_names.values() if name not in ds.variables]
        if missing:
            raise ValueError(f"{path} is missing variables: {missing}")

        depth = ds.variables[variable_names["depth"]]
        component_x = ds.variables[variable_names["component_x"]]
        component_y = ds.variables[variable_names["component_y"]]
        time = ds.variables[variable_names["time"]]
        if depth.ndim != 3:
            raise ValueError(f"{path}:{variable_names['depth']} must have dimensions time,y,x")
        if component_x.shape != depth.shape or component_y.shape != depth.shape:
            raise ValueError(f"Dynamic variable shape mismatch in {path}")
        if depth.shape[1:] != expected_grid_shape:
            raise ValueError(
                f"Grid mismatch in {path}: netCDF={depth.shape[1:]}, block grid={expected_grid_shape}"
            )
        if len(time) != depth.shape[0]:
            raise ValueError(f"Time coordinate length mismatch in {path}")

        time_values = np.asarray(time[:], dtype=np.float64)
        if len(time_values) < 2:
            raise ValueError(f"Need at least two output timestamps in {path}")
        deltas = np.diff(time_values)
        if not np.allclose(deltas, deltas[0], rtol=1e-5, atol=1e-7):
            raise ValueError(f"Output timestamps are not uniformly spaced in {path}")

        return {
            "event_id": event_id,
            "watershed_id": watershed_id,
            "path_to_netcdf": str(path.resolve()),
            "n_times": int(depth.shape[0]),
            "rows": int(depth.shape[1]),
            "cols": int(depth.shape[2]),
            "time_start": float(time_values[0]),
            "time_end": float(time_values[-1]),
            "time_step": float(deltas[0]),
            "time_units": str(getattr(time, "units", "")),
            "depth_units": str(getattr(depth, "units", "")),
            "component_x_units": str(getattr(component_x, "units", "")),
            "component_y_units": str(getattr(component_y, "units", "")),
        }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv.resolve())
    required_event_columns = {"event_id", "watershed_id", "path_to_X_event", "T", "F"}
    missing = required_event_columns - set(events.columns)
    if missing:
        raise ValueError(f"events.csv is missing columns: {sorted(missing)}")
    events["event_id"] = events["event_id"].astype(str).str.upper()

    block_index = np.load(args.labels_10m_dir.resolve() / "block_index_10m.npy", mmap_mode="r")
    expected_grid_shape = tuple(int(value) for value in block_index.shape)
    variable_names = {
        "depth": args.depth_var,
        "component_x": args.component_x_var,
        "component_y": args.component_y_var,
        "time": args.time_var,
    }

    paths = sorted(args.netcdf_dir.resolve().glob(args.netcdf_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No netCDF files matched {args.netcdf_pattern!r} under {args.netcdf_dir}"
        )

    records: List[Dict[str, object]] = []
    rejected: List[Dict[str, str]] = []
    seen = set()
    for path in paths:
        event_id = event_id_from_path(path)
        if event_id in seen:
            raise ValueError(f"Duplicate netCDF event ID: {event_id}")
        seen.add(event_id)
        event_rows = events.loc[
            (events["event_id"] == event_id) & (events["watershed_id"] == args.watershed_id)
        ]
        if len(event_rows) != 1:
            raise ValueError(
                f"Expected exactly one events.csv row for {args.watershed_id}/{event_id}, got {len(event_rows)}"
            )
        try:
            record = inspect_file(
                path=path,
                event_id=event_id,
                watershed_id=args.watershed_id,
                variable_names=variable_names,
                expected_grid_shape=expected_grid_shape,
            )
        except Exception as exc:
            if not args.skip_incomplete:
                raise
            rejected.append(
                {"event_id": event_id, "path": str(path.resolve()), "reason": str(exc)}
            )
            print(f"Skipping {event_id}: {exc}")
            continue
        record["path_to_X_event"] = str(event_rows.iloc[0]["path_to_X_event"])
        if args.component_semantics == "unit_discharge":
            record["source_component_x_units"] = record["component_x_units"]
            record["source_component_y_units"] = record["component_y_units"]
            record["component_x_units"] = "m2 s-1"
            record["component_y_units"] = "m2 s-1"
        record["forcing_T"] = int(event_rows.iloc[0]["T"])
        record["forcing_F"] = int(event_rows.iloc[0]["F"])
        if record["forcing_T"] != record["n_times"]:
            raise ValueError(
                f"Timestamp count mismatch for {event_id}: forcing={record['forcing_T']} "
                f"and netCDF={record['n_times']}"
            )
        records.append(record)

    manifest = pd.DataFrame(records).sort_values(["watershed_id", "event_id"]).reset_index(drop=True)
    manifest.to_parquet(output_dir / "dynamic_manifest.parquet", index=False)
    metadata = {
        "format_version": 1,
        "variable_names": variable_names,
        "component_semantics": args.component_semantics,
        "grid_shape": list(expected_grid_shape),
        "n_events": int(len(manifest)),
        "event_ids": manifest["event_id"].tolist(),
        "component_semantics_evidence": (
            "Archived TRITON source assigns g_HUa=HU and g_HVa=HV, writes those "
            "arrays directly as U/V, and computes velocity internally as HU/H and HV/H."
        ),
    }
    (output_dir / "dynamic_metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "rejected_events.json").write_text(json.dumps(rejected, indent=2))
    print(f"Wrote {len(manifest)} events to {output_dir / 'dynamic_manifest.parquet'}")


if __name__ == "__main__":
    main()
