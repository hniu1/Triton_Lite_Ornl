"""Prepare an isolated corrected manifest and preserve the original fold-0 split."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from stage2.target_integrity import require_corrected_targets
from triton_core.data import _split_event_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    expected = {f"D{i:03d}" for i in range(1, 41)}
    reports = {}
    for job in (186578, 187031):
        for report in Path(f"processed_data/stage2_corrected_rebuild/job_{job}").glob(
                "*/HYDRAULIC_VALIDATION.json"):
            data = json.loads(report.read_text())
            event = data["event_id"]
            nc = report.parent / f"{event}_ACC_future.nc"
            if (data["status"] != "hydraulic_validation_passed" or not nc.exists()
                    or len(data["checks"]) != 108
                    or any(check["max_abs_error"] != 0 for check in data["checks"])
                    or data["forcing_alignment"]["status"] != "forcing_alignment_passed"):
                raise ValueError(f"Event validation failed: {report}")
            if event in reports:
                raise ValueError(f"Duplicate validated event: {event}")
            reports[event] = (nc.resolve(), report.resolve())
    if set(reports) != expected:
        raise ValueError(f"Missing events: {sorted(expected - set(reports))}")
    root.mkdir(parents=True, exist_ok=False)
    links = root / "validated_netcdf"
    links.mkdir()
    for event, (nc, _) in sorted(reports.items()):
        (links / nc.name).symlink_to(nc)
    manifest = root / "manifest"
    subprocess.run([
        sys.executable, "-m", "data_preprocessing.m3_build_dynamic_manifest",
        "--netcdf-dir", str(links),
        "--events-csv", "processed_data/timestamp_stage1/m1_events/events.csv",
        "--labels-10m-dir", "processed_data/timestamp_stage1/m3_labels_10m",
        "--component-semantics", "unit_discharge", "--output-dir", str(manifest),
    ], check=True)
    require_corrected_targets(manifest)
    frame = pd.read_parquet(manifest / "dynamic_manifest.parquet")
    if set(frame.event_id) != expected or set(frame.n_times) != {480}:
        raise ValueError("Corrected manifest event/time coverage is wrong")
    previous = json.loads(Path(
        "results/stage2_v5b_robustness_fold0/baseline/seed_45/run_config.json").read_text())
    split = _split_event_ids(sorted(expected), previous["test_events"], .2, 4300)
    if split != previous["split_events"]:
        raise ValueError("Corrected fold-0 split differs from original experiment")
    config = json.loads(Path("results/stage1_transition_multistep_v4b/run_config.json").read_text())
    config.update(manifest_dir=str(manifest), sampling_index_dir=str(root / "sampling"),
                  test_events=split["test"], seed=45, data_split_seed=4300,
                  split_events=split, corrected_data=True)
    # This directory contains configuration only. Fresh training never opens
    # a legacy checkpoint or transfers its learned weights.
    source = root / "source_config"
    source.mkdir()
    (source / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "split_events.json").write_text(json.dumps(split, indent=2) + "\n")
    (root / "train_events.txt").write_text("\n".join(split["train"]) + "\n")
    provenance = dict(status="corrected_manifest_ready", n_events=40,
                      conversion_version="numeric-order-signed-components-v1",
                      split_events=split, legacy_weights_used=False,
                      sampling_status="pending", reports={
                          e: dict(netcdf=str(nc), validation_report=str(report),
                                  report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
                          for e, (nc, report) in sorted(reports.items())})
    (root / "PREPARATION.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Corrected manifest ready: {root}; split 25 train / 7 validation / 8 test", flush=True)


if __name__ == "__main__":
    main()
