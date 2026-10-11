"""Read-only audit of stored forcing against original simulation hydrographs."""
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.m0_generate_netcdf_from_zip import CONASAUGA_DEFAULT_ZIP_DIR


def reconstruct_forcing(raw, times):
    if raw.ndim != 2 or raw.shape[1] != 301 or not np.isfinite(raw).all():
        raise ValueError("Expected finite hydrograph time plus 300 source columns")
    if not np.all(np.diff(raw[:, 0]) > 0):
        raise ValueError("Source hydrograph times must be strictly increasing")
    if times[0] < raw[0, 0] or times[-1] > raw[-1, 0]:
        raise ValueError("Output times extend beyond forcing support")
    # Existing m1b first converts source discharge to float32, then interpolates.
    source = raw[:, 1:].astype(np.float32)
    return np.stack([np.interp(times, raw[:, 0], source[:, i]) for i in range(300)], axis=1).astype(np.float32)


def config_number(text, name):
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^\s#]+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing simulator configuration field: {name}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=Path(CONASAUGA_DEFAULT_ZIP_DIR))
    parser.add_argument("--events-csv", type=Path, default=Path("processed_data/timestamp_stage1/m1_events/events.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    events = pd.read_csv(args.events_csv)
    reports = []
    for event in events.itertuples(index=False):
        archive_path = args.zip_dir / f"{event.event_id}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            candidates = [n for n in archive.namelist() if "/input/hyg/" in n and n.endswith(".txt")]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous hydrograph for {event.event_id}: {candidates}")
            member = candidates[0]
            payload = archive.read(member)
            text = payload.decode("utf-8")
            raw = np.loadtxt(io.StringIO(text), delimiter=",", comments="%")
            input_cfg = archive.read(f"{event.event_id}/input/cfg/D001.cfg").decode()
            interval = config_number(input_cfg, "print_interval") / 3600
            duration = config_number(input_cfg, "sim_duration") / 3600
            start = config_number(input_cfg, "sim_start_time") / 3600
            if start != 0 or interval != .5 or duration != 240:
                raise ValueError(f"Unexpected event-relative simulation timing for {event.event_id}")
            times = np.arange(1, 481, dtype=np.float64) * interval
            checkpoint_checks = []
            for counter in (1, 9, 10, 30, 99, 100, 240, 241, 480):
                cfg = archive.read(f"{event.event_id}/output/flood2d/cfg/config_{counter}.cfg").decode()
                actual_hours = config_number(cfg, "sim_start_time") / 3600
                actual_counter = config_number(cfg, "checkpoint_id")
                if not np.isclose(actual_hours, times[counter-1], rtol=0, atol=1e-8) or actual_counter != counter:
                    raise ValueError(f"Snapshot counter/time mismatch: {event.event_id}, {counter}")
                checkpoint_checks.append(dict(counter=counter, simulator_hours=actual_hours))
            expected = reconstruct_forcing(raw, times)
            stored = np.load(event.path_to_X_event)
            if stored.shape != expected.shape or not np.isfinite(stored).all():
                raise ValueError(f"Stored forcing shape/values invalid for {event.event_id}")
            exact = bool(np.array_equal(stored, expected))
            max_error = float(np.max(np.abs(stored.astype(np.float64)-expected)))
            reports.append(dict(event_id=event.event_id, source_archive=str(archive_path),
                                source_member=member, source_hydrograph_sha256=hashlib.sha256(payload).hexdigest(),
                                stored_tensor=event.path_to_X_event,
                                stored_tensor_sha256=hashlib.sha256(Path(event.path_to_X_event).read_bytes()).hexdigest(),
                                source_rows=int(raw.shape[0]), source_start_hours=float(raw[0, 0]),
                                source_end_hours=float(raw[-1, 0]), output_start_hours=float(times[0]),
                                output_end_hours=float(times[-1]), source_columns="Loc1 through Loc300, source order",
                                raw_header_comments=[line for line in text.splitlines() if line.startswith("%")],
                                exact_float32_match=exact, max_abs_error=max_error,
                                simulator_checkpoint_checks=checkpoint_checks))
            print(f"{event.event_id}: exact={exact}, max_abs_error={max_error:.9g}", flush=True)
    passed = len(reports) == 40 and all(r["exact_float32_match"] for r in reports)
    report = dict(status="forcing_alignment_passed" if passed else "forcing_alignment_failed",
                  events_checked=len(reports), expected_events=40, output_hours="0.5 through 240, every 0.5 hours",
                  interpolation="Linear; float32 source values, float64 interpolation, float32 output",
                  calendar_epoch_verified=False, existing_tensors_modified=False,
                  unit_caveat="Archive hydrograph comments declare cfs; physical forcing-unit handling requires separate source-code audit before conservation losses",
                  events=reports)
    (args.output_dir / "FORCING_ALIGNMENT.json").write_text(json.dumps(report, indent=2)+"\n")
    if not passed:
        raise SystemExit("Forcing alignment audit failed; inspect report")


if __name__ == "__main__":
    main()
