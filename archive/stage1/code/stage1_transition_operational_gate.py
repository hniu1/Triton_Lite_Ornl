#!/usr/bin/env python3
"""Gate transition checkpoints using rollout and hydraulic-regime evidence."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rollout", type=Path, required=True)
    parser.add_argument("--reference-rollout", type=Path, required=True)
    parser.add_argument("--candidate-regimes", type=Path, required=True)
    parser.add_argument("--reference-regimes", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--operational-horizons", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--stable-depth-rmse-tolerance", type=float, default=0.05)
    parser.add_argument("--stable-f1-tolerance", type=float, default=0.02)
    parser.add_argument("--reference-score-tolerance-percent", type=float, default=2.0)
    parser.add_argument("--velocity-persistence-tolerance-percent", type=float, default=25.0)
    return parser.parse_args()


def check(name, passed, actual, limit, comparison):
    return {
        "name": name,
        "passed": bool(passed),
        "actual": float(actual),
        "limit": float(limit),
        "comparison": comparison,
    }


def evaluate_operational_gate(
    candidate_rollout,
    reference_rollout,
    candidate_regimes,
    reference_regimes,
    operational_horizons=(6, 12, 24),
    stable_depth_rmse_tolerance=0.05,
    stable_f1_tolerance=0.02,
    reference_score_tolerance_percent=2.0,
    velocity_persistence_tolerance_percent=25.0,
):
    checks = []
    candidate_horizons = candidate_rollout["metrics"]
    reference_horizons = reference_rollout["metrics"]
    for horizon in operational_horizons:
        key = str(horizon)
        if key not in candidate_horizons or key not in reference_horizons:
            checks.append(check(f"h{horizon}_available", False, 0, 1, "required"))
            continue
        candidate = candidate_horizons[key]["autoregressive"]
        persistence = candidate_horizons[key]["persistence"]
        reference = reference_horizons[key]["autoregressive"]
        checks.extend(
            [
                check(
                    f"h{horizon}_depth_beats_persistence",
                    candidate["depth_wet_rmse"] < persistence["depth_wet_rmse"],
                    candidate["depth_wet_rmse"],
                    persistence["depth_wet_rmse"],
                    "less_than",
                ),
                check(
                    f"h{horizon}_score_beats_persistence",
                    candidate["physical_score"] < persistence["physical_score"],
                    candidate["physical_score"],
                    persistence["physical_score"],
                    "less_than",
                ),
                check(
                    f"h{horizon}_score_not_worse_than_reference",
                    candidate["physical_score"]
                    <= reference["physical_score"]
                    * (1.0 + reference_score_tolerance_percent / 100.0),
                    candidate["physical_score"],
                    reference["physical_score"]
                    * (1.0 + reference_score_tolerance_percent / 100.0),
                    "less_than_or_equal",
                ),
            ]
        )
        if "derived_velocity_rmse" in candidate:
            velocity_limit = persistence["derived_velocity_rmse"] * (
                1.0 + velocity_persistence_tolerance_percent / 100.0
            )
            checks.append(
                check(
                    f"h{horizon}_velocity_bounded",
                    candidate["derived_velocity_rmse"] <= velocity_limit,
                    candidate["derived_velocity_rmse"],
                    velocity_limit,
                    "less_than_or_equal",
                )
            )

    candidate_by_regime = candidate_regimes["metrics"]
    reference_by_regime = reference_regimes["metrics"]
    for regime in ("stable", "filling", "draining", "rapid"):
        if regime not in candidate_by_regime or regime not in reference_by_regime:
            checks.append(check(f"regime_{regime}_available", False, 0, 1, "required"))
            continue
        candidate = candidate_by_regime[regime]["model"]
        persistence = candidate_by_regime[regime]["persistence"]
        reference = reference_by_regime[regime]["model"]
        if regime == "stable":
            checks.extend(
                [
                    check(
                        "stable_depth_near_persistence",
                        candidate["depth_wet_rmse"]
                        <= persistence["depth_wet_rmse"] + stable_depth_rmse_tolerance,
                        candidate["depth_wet_rmse"],
                        persistence["depth_wet_rmse"] + stable_depth_rmse_tolerance,
                        "less_than_or_equal",
                    ),
                    check(
                        "stable_f1_near_persistence",
                        candidate["wet_f1"]
                        >= persistence["wet_f1"] - stable_f1_tolerance,
                        candidate["wet_f1"],
                        persistence["wet_f1"] - stable_f1_tolerance,
                        "greater_than_or_equal",
                    ),
                ]
            )
        else:
            checks.extend(
                [
                    check(
                        f"{regime}_depth_beats_persistence",
                        candidate["depth_wet_rmse"] < persistence["depth_wet_rmse"],
                        candidate["depth_wet_rmse"],
                        persistence["depth_wet_rmse"],
                        "less_than",
                    ),
                    check(
                        f"{regime}_score_beats_persistence",
                        candidate["physical_score"] < persistence["physical_score"],
                        candidate["physical_score"],
                        persistence["physical_score"],
                        "less_than",
                    ),
                ]
            )
        checks.append(
            check(
                f"{regime}_score_not_worse_than_reference",
                candidate["physical_score"]
                <= reference["physical_score"]
                * (1.0 + reference_score_tolerance_percent / 100.0),
                candidate["physical_score"],
                reference["physical_score"]
                * (1.0 + reference_score_tolerance_percent / 100.0),
                "less_than_or_equal",
            )
        )
    return {
        "accepted": all(item["passed"] for item in checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "checks": checks,
    }


def main():
    args = parse_args()
    payloads = [
        json.loads(path.resolve().read_text())
        for path in (
            args.candidate_rollout,
            args.reference_rollout,
            args.candidate_regimes,
            args.reference_regimes,
        )
    ]
    result = evaluate_operational_gate(
        *payloads,
        operational_horizons=args.operational_horizons,
        stable_depth_rmse_tolerance=args.stable_depth_rmse_tolerance,
        stable_f1_tolerance=args.stable_f1_tolerance,
        reference_score_tolerance_percent=args.reference_score_tolerance_percent,
        velocity_persistence_tolerance_percent=args.velocity_persistence_tolerance_percent,
    )
    result["inputs"] = {
        "candidate_rollout": str(args.candidate_rollout.resolve()),
        "reference_rollout": str(args.reference_rollout.resolve()),
        "candidate_regimes": str(args.candidate_regimes.resolve()),
        "reference_regimes": str(args.reference_regimes.resolve()),
    }
    args.output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_path.resolve().write_text(json.dumps(result, indent=2))
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
