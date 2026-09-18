#!/usr/bin/env python3
"""Aggregate completed five-fold V5-B diagnostic evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "depth_wet_rmse", "depth_all_rmse", "wet_f1", "wet_precision",
    "wet_recall", "dry_specificity", "dry_false_positive_rate",
    "component_rmse", "derived_velocity_rmse", "physical_score",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage2_v5b_crossval"))
    return parser.parse_args()


def run_dir(root, fold):
    if fold == 0:
        return Path("results/stage2_v5b_robustness_fold0/halo/seed_45")
    return root / f"fold_{fold}" / "seed_45"


def main():
    args = parse_args()
    root = args.output_dir.resolve()
    records = {}
    for policy in ("learned", "velocity_persistence"):
        folds = []
        for fold in range(5):
            path = run_dir(root, fold) / f"test_diagnostic_{policy}.json"
            folds.append(json.loads(path.read_text()))
        summary = {}
        for metric in METRICS:
            fold_means = [
                float(np.mean([v["direct"][metric] for v in payload["metrics"].values()]))
                for payload in folds
            ]
            persistence_means = [
                float(np.mean([v["persistence"][metric] for v in payload["metrics"].values()]))
                for payload in folds
            ]
            summary[metric] = {
                "fold_mean": float(np.mean(fold_means)),
                "fold_standard_deviation": float(np.std(fold_means, ddof=1)),
                "fold_values": fold_means,
                "persistence_fold_mean": float(np.mean(persistence_means)),
            }
        records[policy] = {
            "summary": summary,
            "extent_selections": [
                {
                    "fold": fold,
                    "mode": payload["wet_classification_mode"],
                    "threshold": payload["wet_probability_threshold"],
                }
                for fold, payload in enumerate(folds)
            ],
            "test_events": [payload["split_events"] for payload in folds],
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "CROSSVAL_SUMMARY.json").write_text(json.dumps(records, indent=2))
    learned = records["learned"]["summary"]
    velocity = records["velocity_persistence"]["summary"]
    lines = [
        "# V5-B Five-Fold Cross-Validation Results", "",
        "Each of the 40 events is held out exactly once. Values are the mean and",
        "sample standard deviation of five fold-level means across all 48 leads.", "",
        "| Metric | Learned HU/HV | Velocity persistence | Full-state persistence |", "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        a = learned[metric]
        b = velocity[metric]
        lines.append(
            f"| {metric} | {a['fold_mean']:.6f} ± {a['fold_standard_deviation']:.6f} "
            f"| {b['fold_mean']:.6f} ± {b['fold_standard_deviation']:.6f} "
            f"| {a['persistence_fold_mean']:.6f} |"
        )
    lines.extend(["", "## Validation-selected extent rules", ""])
    for item in records["learned"]["extent_selections"]:
        lines.append(
            f"- Fold {item['fold']}: `{item['mode']}` at probability threshold "
            f"{item['threshold']:.3f}"
        )
    (root / "CROSSVAL_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {root / 'CROSSVAL_SUMMARY.json'}")
    print(f"Wrote {root / 'CROSSVAL_RESULTS.md'}")


if __name__ == "__main__":
    main()
