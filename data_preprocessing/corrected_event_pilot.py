"""Isolated, streaming event rebuild and independent raw-patch validation."""
import argparse
import hashlib
import json
import logging
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np

from data_preprocessing import m0_generate_netcdf_from_zip as converter
from data_preprocessing.m3_build_dynamic_manifest import inspect_file
from stage2.audit_raw_targets import archive_members, raw_patch


def read_member(archive, member, rows, cols, threshold, pad, preserve_signed=False):
    cells = (rows + 2 * pad) * (cols + 2 * pad)
    size = archive.getinfo(str(member)).file_size
    if size not in (cells * 4, cells * 8):
        raise ValueError(f"Unexpected binary size: {member}: {size}")
    dtype = np.dtype("<f4" if size == cells * 4 else "<f8")
    with archive.open(str(member)) as stream:
        payload = stream.read()
    if len(payload) != size:
        raise ValueError(f"Truncated binary: {member}")
    values = np.frombuffer(payload, dtype=dtype).reshape(rows + 2*pad, cols + 2*pad)
    values = values[pad:rows+pad, pad:cols+pad].astype(np.float32)
    valid = np.isfinite(values) & (values != -9999)
    if not preserve_signed:
        valid &= values > threshold
    values[~valid] = np.nan
    return values


def validate(output, archive, members, windows, rows, cols, indices):
    checks = []
    record = inspect_file(output, output.name[:4], "conasauga",
                          dict(depth="output_depth", component_x="output_velocity_x",
                               component_y="output_velocity_y", time="out_time"), (rows, cols))
    with netCDF4.Dataset(output) as nc:
        np.testing.assert_array_equal(nc.variables["out_time"][:], np.arange(1, len(members["H"])+1)*.5)
        x, y = converter.build_coordinate_arrays(cols, rows, converter.CONASAUGA_DEFAULT_GT)
        np.testing.assert_array_equal(nc.variables["x"][:], x)
        np.testing.assert_array_equal(nc.variables["y"][:], y)
        for prefix in "HUV":
            var = nc.variables[converter.OUTPUT_TYPE_META[prefix]["varname"]]
            if var.units != converter.OUTPUT_TYPE_META[prefix]["units"]:
                raise ValueError("Incorrect physical units")
            for index in indices:
                for label, window in windows:
                    raw = raw_patch(archive, str(members[prefix][index]), window, rows, cols)
                    valid = np.isfinite(raw) & (raw != -9999)
                    if prefix == "H":
                        valid &= raw > 0
                    actual = var[index, window.row_start:window.row_start+window.height,
                                 window.col_start:window.col_start+window.width]
                    np.testing.assert_array_equal(np.ma.getmaskarray(actual), ~valid)
                    np.testing.assert_array_equal(np.asarray(actual)[valid], raw[valid])
                    checks.append(dict(component=prefix, counter=index+1, block_label=label,
                                       valid_cells=int(valid.sum()),
                                       negative_cells=int(((raw < 0) & valid).sum()), max_abs_error=0.0))
    return record, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="D029", choices=[f"D{i:03d}" for i in range(1, 41)])
    parser.add_argument("--zip-dir", type=Path, default=Path(converter.CONASAUGA_DEFAULT_ZIP_DIR))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-scope", choices=["single_event_pilot", "full_event_rebuild"],
                        default="single_event_pilot")
    parser.add_argument("--forcing-audit", type=Path)
    args = parser.parse_args()
    converter.configure_logging("INFO")
    forcing_evidence = None
    if args.forcing_audit is not None:
        audit = json.loads(args.forcing_audit.read_text())
        matches = [event for event in audit["events"] if event["event_id"] == args.event]
        if (audit.get("status") != "forcing_alignment_passed" or len(matches) != 1
                or not matches[0].get("exact_float32_match")):
            raise ValueError(f"Missing passed forcing audit for {args.event}")
        forcing_evidence = dict(status="forcing_alignment_passed", event_id=args.event,
                                audit_path=str(args.forcing_audit.resolve()),
                                audit_sha256=hashlib.sha256(args.forcing_audit.read_bytes()).hexdigest())
    if args.run_scope == "full_event_rebuild" and forcing_evidence is None:
        raise ValueError("Full rebuild requires the independent forcing audit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.zip_dir / f"{args.event}.zip"
    output = args.output_dir / f"{args.event}_ACC_future.nc"
    partial = output.with_suffix(".nc.partial")
    started = time.monotonic()
    with zipfile.ZipFile(source) as archive:
        members = {prefix: [Path(name) for name in archive_members(archive, prefix)[1]] for prefix in "HUV"}
        if any(len(paths) != 480 for paths in members.values()):
            raise ValueError("Rebuild requires 480 aligned H/U/V snapshots")
        converter.create_netcdf(
            partial, members, converter.CONASAUGA_DEFAULT_COLS, converter.CONASAUGA_DEFAULT_ROWS,
            converter.CONASAUGA_DEFAULT_GT, converter.CONASAUGA_DEFAULT_REF_TIME_UNITS,
            converter.parse_time_origin(converter.CONASAUGA_DEFAULT_TIME_ORIGIN), 0., 1, 80,
            10800, 1800, converter.CONASAUGA_DEFAULT_CRS_WKT, 1, 0,
            array_reader=lambda member, **kw: read_member(archive, member, **kw))
        conversion_seconds = time.monotonic() - started
        with netCDF4.Dataset(partial, "a") as nc:
            nc.source_archive = str(source.resolve())
            nc.source_archive_bytes = str(source.stat().st_size)
            nc.time_interpretation = "Event-relative hours; CF epoch is a compatibility label, not verified event calendar date"
        grid = np.load("processed_data/timestamp_stage1/m3_labels_10m/block_index_10m.npy", mmap_mode="r")
        windows = []
        for label in (0, 2264, 5000, 6899):
            row, col = np.where(grid == label)
            if not len(row):
                raise ValueError(f"Missing validation block {label}")
            windows.append((label, SimpleNamespace(row_start=int(row.min()), col_start=int(col.min()),
                                                   height=int(row.max()-row.min()+1), width=int(col.max()-col.min()+1))))
        record, checks = validate(partial, archive, members, windows, grid.shape[0], grid.shape[1],
                                  (0, 8, 9, 29, 98, 99, 239, 240, 479))
        negative_cells = sum(check["negative_cells"] for check in checks)
        if negative_cells == 0:
            raise ValueError("Validation did not exercise negative discharge")
    partial.rename(output)
    record["path_to_netcdf"] = str(output.resolve())
    report = dict(status=("hydraulic_validation_passed" if forcing_evidence else
                          "hydraulic_pilot_passed_forcing_provenance_pending"), event_id=args.event,
                  source_archive=str(source), source_bytes=source.stat().st_size,
                  source_mtime_ns=source.stat().st_mtime_ns, conversion_seconds=conversion_seconds,
                  total_seconds=time.monotonic()-started, output_bytes=output.stat().st_size,
                  run_scope=args.run_scope, active_manifest_changed=False,
                  forcing_alignment=(forcing_evidence or "Original timestamp provenance pending"),
                  calendar_epoch="Compatibility label only; event-relative output hours validated",
                  sampled_negative_component_occurrences=negative_cells,
                  converter_sha256=hashlib.sha256(Path(converter.__file__).read_bytes()).hexdigest(),
                  manifest_record=record, checks=checks)
    report_name = ("HYDRAULIC_VALIDATION.json" if args.run_scope == "full_event_rebuild"
                   else "PILOT_VALIDATION.json")
    (args.output_dir / report_name).write_text(json.dumps(report, indent=2)+"\n")
    logging.info("Event validated: %s; %.1f seconds; %.2f GiB", output, report["total_seconds"],
                 report["output_bytes"]/2**30)


if __name__ == "__main__":
    main()
