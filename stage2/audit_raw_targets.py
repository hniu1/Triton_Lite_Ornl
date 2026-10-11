"""Read-only comparison of legacy labels with raw TRITON binary archive patches."""

import argparse
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np

from data_preprocessing.m0_generate_netcdf_from_zip import CONASAUGA_DEFAULT_ZIP_DIR, output_step


def raw_patch(archive, member, window, rows, cols, pad=1):
    cells = (rows + 2*pad) * (cols + 2*pad)
    size = archive.getinfo(member).file_size
    if size not in (cells*4, cells*8):
        raise ValueError(f"Unexpected raw size: {member}: {size}")
    dtype = np.dtype("<f4" if size == cells*4 else "<f8")
    result = np.empty((window.height, window.width), dtype=np.float32)
    with archive.open(member) as stream:
        for row in range(window.height):
            offset = ((window.row_start + row + pad)*(cols+2*pad)+window.col_start+pad)*dtype.itemsize
            stream.seek(offset)
            values = stream.read(window.width*dtype.itemsize)
            if len(values) != window.width*dtype.itemsize:
                raise ValueError("Truncated binary patch")
            result[row] = np.frombuffer(values, dtype=dtype)
    return result


def archive_members(archive, prefix):
    legacy = sorted(name for name in archive.namelist()
                    if Path(name).name.startswith(prefix + "_") and name.endswith(".dat"))
    numeric = sorted(legacy, key=lambda name: output_step(Path(name)))
    if [output_step(Path(name)) for name in numeric] != list(range(1, len(numeric)+1)):
        raise ValueError("Archive counters are not unique and contiguous")
    return legacy, numeric


def main():
    # Conversion reuses the raw readers without needing plotting or PyTorch.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stage2.data import DirectHorizonDataset
    from stage2.train_direct import build_bundle
    from triton_core.data import _center_pad

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-run", type=Path, default=Path("results/stage2_v5d_learning_check/job_186110"))
    parser.add_argument("--reference-run", type=Path, default=Path("results/stage2_v5b_robustness_fold0/halo/seed_45"))
    parser.add_argument("--zip-dir", type=Path, default=Path(CONASAUGA_DEFAULT_ZIP_DIR))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.reference_run/"run_config.json").read_text())
    base_config = json.loads((Path(reference["initial_run_dir"])/"run_config.json").read_text())
    bundle_args = SimpleNamespace(data_split_seed=reference["data_split_seed"], test_events=reference["test_events"],
                                  batch_size=1, train_batches_per_epoch=1, selection_batches=1,
                                  selection_start_time_stride=12, max_horizon_steps=48,
                                  netcdf_chunk_cache_mb=32, max_open_netcdf_handles=4)
    bundle = build_bundle(base_config, bundle_args, Path(reference["sampling_index_dir"]))
    dataset = DirectHorizonDataset(bundle.train_dataset)
    records = json.loads((args.diagnostic_run/"fixed_training_cohort.json").read_text())
    checks, ordering = [], []
    # Inspect naming across every event; raw value checks are confined to the
    # selected training examples, never held-out target-based model selection.
    all_events = sorted(set(event for split in bundle.split_events.values() for event in split))
    for event_id in all_events:
        with zipfile.ZipFile(args.zip_dir/f"{event_id}.zip") as archive:
            entries = {}
            for prefix in ("H", "U", "V"):
                legacy, numeric = archive_members(archive, prefix)
                entries[prefix] = dict(count=len(legacy), mismatch_positions=sum(a != b for a,b in zip(legacy,numeric)),
                                       first_legacy_steps=[output_step(Path(name)) for name in legacy[:12]])
            ordering.append(dict(event_id=event_id, components=entries))
    figure, axes = plt.subplots(len(records), 1, figsize=(10, 3*len(records)))
    for record, axis in zip(records, np.atleast_1d(axes)):
        event_id = record["event_id"]
        position = dataset.events.index[dataset.events.event_id.astype(str)==event_id].tolist()[0]
        block = record["block_position"]
        label = int(dataset.block_rows.iloc[block]["block_index"])
        window = dataset.block_windows[label]
        raster_mask = dataset.block_index_grid[window.row_start:window.row_stop, window.col_start:window.col_stop] == label
        event = dataset.events.iloc[position]
        with zipfile.ZipFile(args.zip_dir/f"{event_id}.zip") as archive, netCDF4.Dataset(event.resolved_netcdf_path) as nc:
            source_time = np.asarray(nc.variables[dataset.variable_names["time"]][:], dtype=float)
            step_hours = float(event.time_step)
            if not np.allclose(np.diff(source_time), step_hours):
                raise ValueError("Nonuniform NetCDF time coordinate")
            for prefix, variable, field in (("H", "depth", "depth"), ("U", "component_x", "component_x"),
                                            ("V", "component_y", "component_y")):
                legacy, numeric = archive_members(archive, prefix)
                if len(numeric) != len(source_time):
                    raise ValueError("Raw archive and NetCDF time counts differ")
                for index in (record["origin_time_index"], record["target_time_index"]):
                    raw_legacy = raw_patch(archive, legacy[index], window, int(event.rows), int(event.cols))
                    raw_numeric = raw_patch(archive, numeric[index], window, int(event.rows), int(event.cols))
                    processed = dataset.base_dataset[(position, index, block)][field].numpy()
                    legacy_filtered = np.where(np.isfinite(raw_legacy) & (raw_legacy > 0), raw_legacy, 0) * raster_mask
                    expected = _center_pad(legacy_filtered, dataset.target_rows, dataset.target_cols)
                    numeric_signed = np.where(np.isfinite(raw_numeric) & (raw_numeric != -9999), raw_numeric, 0)*raster_mask
                    corrected = _center_pad(np.maximum(numeric_signed, 0) if prefix=="H" else numeric_signed,
                                            dataset.target_rows, dataset.target_cols)
                    negative = (raw_legacy < 0) & (raw_legacy != -9999) & raster_mask
                    check = dict(event_id=event_id, block_label=label, variable=variable, index=index,
                                 declared_time_hours=float(source_time[index]),
                                 lexical_source_step=output_step(Path(legacy[index])),
                                 numeric_source_step=output_step(Path(numeric[index])),
                                 lexical_source_time_hours=output_step(Path(legacy[index]))*step_hours,
                                 raw_lexical_file=legacy[index], raw_numeric_file=numeric[index],
                                 adapter_matches_legacy_max_abs=float(np.max(np.abs(processed-expected))),
                                 differs_from_correct_numeric_max_abs=float(np.max(np.abs(processed-corrected))),
                                 raw_negative_cells=int(negative.sum()),
                                 negative_cells_clipped_to_zero=int(((raw_legacy<0) & (raw_legacy!=-9999)
                                                                    & raster_mask & (legacy_filtered==0)).sum()))
                    checks.append(check)
                    print(json.dumps(check), flush=True)
            legacy, numeric = archive_members(archive, "H")
            mapping = {name: index for index,name in enumerate(legacy)}
            times = range(max(0, record["origin_time_index"]-2), min(len(legacy), record["target_time_index"]+3))
            means_legacy, means_numeric = [], []
            for index in times:
                for output, source_index in ((means_legacy, index), (means_numeric, mapping[numeric[index]])):
                    values = nc.variables[dataset.variable_names["depth"]][source_index,
                             window.row_start:window.row_stop, window.col_start:window.col_stop]
                    values = np.ma.filled(values, 0)
                    output.append(float(values[raster_mask].mean()))
            hours = np.asarray(list(times))*step_hours + float(source_time[0])
            axis.plot(hours, means_legacy, label="Existing labels (lexical order)")
            axis.plot(hours, means_numeric, label="Numeric source-counter order")
            axis.set_title(f"{event_id}, block {label}; forecast lead {record['lead_steps']*step_hours:g} h")
            axis.set_ylabel("Mean depth (m)")
            axis.legend(fontsize=8)
    axes[-1].set_xlabel("Output time (hours from event origin)")
    figure.tight_layout()
    figure.savefig(args.output_dir/"depth_time_order_audit.png", dpi=150)
    plt.close(figure)
    matched = all(check["adapter_matches_legacy_max_abs"] <= 1e-5 for check in checks)
    mismatch = any(check["differs_from_correct_numeric_max_abs"] > 1e-5 for check in checks)
    clipped = sum(check["negative_cells_clipped_to_zero"] for check in checks if check["variable"] != "depth")
    report = dict(status="legacy_label_contamination_confirmed" if matched and mismatch else "requires_review",
                  raw_archive_directory=str(args.zip_dir), adapter_matches_lexical_positive_filter=matched,
                  differs_from_numeric_signed_targets=mismatch, clipped_negative_component_cells=clipped,
                  archive_ordering=ordering, raw_patch_checks=checks,
                  warning="Do not interpret legacy metrics as chronological hydraulic forecasting skill")
    (args.output_dir/"TARGET_AUDIT.json").write_text(json.dumps(report, indent=2))
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
