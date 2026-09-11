#!/usr/bin/env python3
"""Gate whole-area D030 key times against persistence and a reference model."""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--stable-times", type=int, nargs="+", default=[60, 440])
    parser.add_argument("--dynamic-times", type=int, nargs="+", default=[140, 240, 360])
    parser.add_argument("--stable-depth-tolerance", type=float, default=0.05)
    parser.add_argument("--stable-f1-tolerance", type=float, default=0.02)
    parser.add_argument("--stable-velocity-absolute-tolerance", type=float, default=0.01)
    parser.add_argument("--dynamic-velocity-relative-tolerance", type=float, default=0.25)
    parser.add_argument("--reference-score-relative-tolerance", type=float, default=0.02)
    return parser.parse_args()


def physical_score(row, prefix):
    return (
        float(row[f"{prefix}_depth_wet_rmse"])
        + 2.0 * float(row[f"{prefix}_component_rmse"])
        + 0.5 * (1.0 - float(row[f"{prefix}_wet_f1"]))
    )


def make_check(name, passed, actual, limit, comparison):
    return {
        "name": name,
        "passed": bool(passed),
        "actual": float(actual),
        "limit": float(limit),
        "comparison": comparison,
    }


def evaluate_whole_area_gate(
    candidate,
    reference,
    stable_times=(60, 440),
    dynamic_times=(140, 240, 360),
    stable_depth_tolerance=0.05,
    stable_f1_tolerance=0.02,
    stable_velocity_absolute_tolerance=0.01,
    dynamic_velocity_relative_tolerance=0.25,
    reference_score_relative_tolerance=0.02,
):
    candidate = candidate.set_index("time_index", drop=False)
    reference = reference.set_index("time_index", drop=False)
    checks = []
    details = {}
    for time_index in (*stable_times, *dynamic_times):
        if time_index not in candidate.index or time_index not in reference.index:
            checks.append(
                make_check(f"t{time_index}_available", False, 0, 1, "required")
            )
            continue
        row = candidate.loc[time_index]
        baseline = reference.loc[time_index]
        candidate_score = physical_score(row, "gated")
        persistence_score = physical_score(row, "persistence")
        reference_score = physical_score(baseline, "gated")
        details[str(time_index)] = {
            "candidate_score": candidate_score,
            "persistence_score": persistence_score,
            "reference_score": reference_score,
        }
        checks.append(
            make_check(
                f"t{time_index}_score_not_worse_than_reference",
                candidate_score
                <= reference_score * (1.0 + reference_score_relative_tolerance),
                candidate_score,
                reference_score * (1.0 + reference_score_relative_tolerance),
                "less_than_or_equal",
            )
        )
        if time_index in stable_times:
            checks.extend(
                [
                    make_check(
                        f"t{time_index}_stable_depth_near_persistence",
                        row["gated_depth_wet_rmse"]
                        <= row["persistence_depth_wet_rmse"] + stable_depth_tolerance,
                        row["gated_depth_wet_rmse"],
                        row["persistence_depth_wet_rmse"] + stable_depth_tolerance,
                        "less_than_or_equal",
                    ),
                    make_check(
                        f"t{time_index}_stable_f1_near_persistence",
                        row["gated_wet_f1"]
                        >= row["persistence_wet_f1"] - stable_f1_tolerance,
                        row["gated_wet_f1"],
                        row["persistence_wet_f1"] - stable_f1_tolerance,
                        "greater_than_or_equal",
                    ),
                    make_check(
                        f"t{time_index}_stable_velocity_bounded",
                        row["gated_derived_velocity_rmse"]
                        <= row["persistence_derived_velocity_rmse"]
                        + stable_velocity_absolute_tolerance,
                        row["gated_derived_velocity_rmse"],
                        row["persistence_derived_velocity_rmse"]
                        + stable_velocity_absolute_tolerance,
                        "less_than_or_equal",
                    ),
                ]
            )
        else:
            checks.extend(
                [
                    make_check(
                        f"t{time_index}_dynamic_depth_beats_persistence",
                        row["gated_depth_wet_rmse"]
                        < row["persistence_depth_wet_rmse"],
                        row["gated_depth_wet_rmse"],
                        row["persistence_depth_wet_rmse"],
                        "less_than",
                    ),
                    make_check(
                        f"t{time_index}_dynamic_score_beats_persistence",
                        candidate_score < persistence_score,
                        candidate_score,
                        persistence_score,
                        "less_than",
                    ),
                    make_check(
                        f"t{time_index}_dynamic_velocity_bounded",
                        row["gated_derived_velocity_rmse"]
                        <= row["persistence_derived_velocity_rmse"]
                        * (1.0 + dynamic_velocity_relative_tolerance),
                        row["gated_derived_velocity_rmse"],
                        row["persistence_derived_velocity_rmse"]
                        * (1.0 + dynamic_velocity_relative_tolerance),
                        "less_than_or_equal",
                    ),
                ]
            )
    return {
        "accepted": all(item["passed"] for item in checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "checks": checks,
        "time_details": details,
    }


def main():
    args = parse_args()
    result = evaluate_whole_area_gate(
        pd.read_csv(args.candidate_csv.resolve()),
        pd.read_csv(args.reference_csv.resolve()),
        stable_times=args.stable_times,
        dynamic_times=args.dynamic_times,
        stable_depth_tolerance=args.stable_depth_tolerance,
        stable_f1_tolerance=args.stable_f1_tolerance,
        stable_velocity_absolute_tolerance=args.stable_velocity_absolute_tolerance,
        dynamic_velocity_relative_tolerance=args.dynamic_velocity_relative_tolerance,
        reference_score_relative_tolerance=args.reference_score_relative_tolerance,
    )
    result["inputs"] = {
        "candidate_csv": str(args.candidate_csv.resolve()),
        "reference_csv": str(args.reference_csv.resolve()),
    }
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    for item in result["checks"]:
        print(
            f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: "
            f"actual={item['actual']:.6g} limit={item['limit']:.6g}"
        )
    print(
        f"accepted={result['accepted']} "
        f"checks={result['passed_checks']}/{result['total_checks']}"
    )


if __name__ == "__main__":
    main()
