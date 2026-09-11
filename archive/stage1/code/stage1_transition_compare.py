#!/usr/bin/env python3
"""Apply controlled acceptance gates to a candidate transition run."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--max-f1-regression", type=float, default=0.005)
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args()


def evaluate_acceptance(candidate, reference, max_f1_regression=0.005):
    test = candidate["test"]
    persistence = candidate["persistence_test"]
    reference_test = reference["test"]
    required = ("depth_wet_rmse", "component_rmse", "wet_f1", "physical_score")
    finite = all(
        math.isfinite(float(metrics[name]))
        for metrics in (test, persistence, reference_test)
        for name in required
    )
    gates = {
        "all_primary_metrics_finite": finite,
        "physical_score_beats_reference": test["physical_score"]
        < reference_test["physical_score"],
        "wet_depth_rmse_beats_reference": test["depth_wet_rmse"]
        < reference_test["depth_wet_rmse"],
        "physical_score_beats_persistence": test["physical_score"]
        < persistence["physical_score"],
        "wet_depth_rmse_beats_persistence": test["depth_wet_rmse"]
        < persistence["depth_wet_rmse"],
        "component_rmse_not_worse_than_persistence": test["component_rmse"]
        <= persistence["component_rmse"],
        "wet_f1_within_persistence_tolerance": test["wet_f1"]
        >= persistence["wet_f1"] - float(max_f1_regression),
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "candidate_test": {name: test[name] for name in required},
        "candidate_persistence": {name: persistence[name] for name in required},
        "reference_test": {name: reference_test[name] for name in required},
        "max_f1_regression": float(max_f1_regression),
    }


def main():
    args = parse_args()
    candidate = json.loads(args.candidate.resolve().read_text())
    reference = json.loads(args.reference.resolve().read_text())
    result = evaluate_acceptance(candidate, reference, args.max_f1_regression)
    output_path = args.output_path or args.candidate.resolve().with_name(
        "acceptance.json"
    )
    output_path.resolve().write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if args.fail_on_reject and not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

