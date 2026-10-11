"""Verify sampler coverage, train-only forcing normalization, and real data batches."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from stage2.data import DirectHorizonDataset
from stage2.target_integrity import require_corrected_sampling, require_corrected_targets
from stage2.train_direct import build_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    config = json.loads((root / "source_config/run_config.json").read_text())
    require_corrected_targets(root / "manifest")
    require_corrected_sampling(root / "sampling", root / "manifest")
    split = json.loads((root / "split_events.json").read_text())
    candidates = pd.read_parquet(root / "sampling/sampling_candidates.parquet")
    if set(candidates.event_id) != set(split["train"]):
        raise ValueError("Sampling index must contain exactly the 25 training events")
    bundle_args = SimpleNamespace(data_split_seed=4300, netcdf_chunk_cache_mb=32,
        max_open_netcdf_handles=4, test_events=split["test"], batch_size=8,
        train_batches_per_epoch=20, selection_batches=8, selection_start_time_stride=12,
        max_horizon_steps=48)
    bundle = build_bundle(config, bundle_args, root / "sampling")
    if bundle.split_events != split:
        raise ValueError("Training bundle changed event splits")
    sampled_positions = {position for position, _ in bundle.train_sampler.groups}
    represented = {split["train"][position] for position in sampled_positions}
    if represented != set(split["train"]):
        raise ValueError(f"No feasible batches for training events: {set(split['train']) - represented}")
    train_arrays = [np.load(row.resolved_event_path) for row in bundle.train_dataset.events.itertuples()]
    stacked = np.concatenate(train_arrays, axis=0)
    np.testing.assert_allclose(bundle.normalization.event_mean, stacked.mean(axis=0), rtol=1e-6)
    expected_std = stacked.std(axis=0)
    expected_std[expected_std < 1e-6] = 1
    np.testing.assert_allclose(bundle.normalization.event_std, expected_std, rtol=1e-6)
    for name, array in vars(bundle.normalization).items():
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite normalization: {name}")
    np.savez(root / "normalization.npz", **vars(bundle.normalization))
    dataset = DirectHorizonDataset(bundle.train_dataset, max_horizon_steps=48,
                                  step_hours=.5, history_states=2)
    leads_checked = []
    negative = 0
    for batch_keys, lead in zip(bundle.train_sampler, (1, 6, 12, 24, 48)):
        batch = default_collate([dataset[(*key, lead)] for key in batch_keys])
        for key in ("event", "depth", "component_x", "component_y", "previous_depth",
                    "previous_component_x", "previous_component_y", "older_depth"):
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f"Non-finite real batch: {key}")
        negative += int((batch["component_x"] < 0).sum() + (batch["component_y"] < 0).sum())
        leads_checked.append(lead)
    if leads_checked != [1, 6, 12, 24, 48]:
        raise ValueError("Insufficient forecast-batch checks")
    report = dict(status="passed", split_events=split, n_candidates=len(candidates),
                  feasible_event_time_groups=len(bundle.train_sampler.groups),
                  training_events_with_feasible_batches=sorted(represented),
                  normalization_fit_events=split["train"],
                  normalization_scope="Forcing statistics fit on training events; block/static terrain statistics shared across events",
                  checked_lead_steps=leads_checked, sampled_negative_component_cells=negative,
                  manifest_sha256=hashlib.sha256((root / "manifest/dynamic_manifest.parquet").read_bytes()).hexdigest(),
                  sampling_sha256=hashlib.sha256((root / "sampling/sampling_candidates.parquet").read_bytes()).hexdigest(),
                  test_hydraulic_batches_evaluated=0)
    (root / "TRAINING_PRECHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    preparation = json.loads((root / "PREPARATION.json").read_text())
    preparation.update(status="corrected_training_inputs_ready", sampling_status="passed",
                       training_precheck=str(root / "TRAINING_PRECHECK.json"))
    (root / "PREPARATION.json").write_text(json.dumps(preparation, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
