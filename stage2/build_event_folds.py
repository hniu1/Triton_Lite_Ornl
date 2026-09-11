#!/usr/bin/env python3
"""Create severity-balanced, complete-event cross-validation folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(csv.DictReader(args.events_csv.open()))
    scored = []
    for row in rows:
        path = Path(row["path_to_X_event"])
        event = np.load(path, mmap_mode="r")
        forcing = np.maximum(np.asarray(event), 0.0)
        score = float(forcing.max()) + float(forcing.sum() / forcing.size)
        scored.append((score, row["event_id"]))
    scored.sort(reverse=True)
    folds = [[] for _ in range(args.folds)]
    for group_start in range(0, len(scored), args.folds):
        group = scored[group_start : group_start + args.folds]
        if (group_start // args.folds) % 2:
            group = list(reversed(group))
        for index, (_, event_id) in enumerate(group):
            folds[index].append(event_id)
    payload = {
        "method": "snake-balanced by peak plus mean nonnegative forcing",
        "folds": [sorted(fold) for fold in folds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
