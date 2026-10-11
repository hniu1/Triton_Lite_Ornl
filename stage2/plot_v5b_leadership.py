"""Generate leadership figures and lead-specific tables from saved V5-B tests."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    destination = Path("docs/stage2/figures")
    destination.mkdir(parents=True, exist_ok=True)
    learned, velocity = [], []
    for fold in range(5):
        directory = (
            Path("results/stage2_v5b_robustness_fold0/halo/seed_45")
            if fold == 0
            else Path(f"results/stage2_v5b_crossval/fold_{fold}/seed_45")
        )
        learned.append(json.loads((directory / "test_diagnostic_learned.json").read_text()))
        velocity.append(json.loads((directory / "test_diagnostic_velocity_persistence.json").read_text()))
    leads = np.arange(1, 49) / 2
    panels = [
        ("depth_wet_rmse", "Wet-cell depth RMSE (m)"),
        ("wet_f1", "Flood-extent F1"),
        ("derived_velocity_rmse", "Velocity component RMSE (m/s)"),
    ]
    rows = []
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, (metric, label) in zip(axes, panels):
        direct = np.array([[d["metrics"][str(h)]["direct"][metric] for h in range(1, 49)] for d in learned])
        persistence = np.array([[d["metrics"][str(h)]["persistence"][metric] for h in range(1, 49)] for d in learned])
        axis.plot(leads, direct.mean(0), color="#1864ab", label="V5-B learned outputs")
        axis.fill_between(leads, direct.min(0), direct.max(0), color="#1864ab", alpha=0.12, label="Range across five folds")
        axis.plot(leads, persistence.mean(0), color="#666666", linestyle="--", label="Full-state persistence")
        if metric == "derived_velocity_rmse":
            hybrid = np.array([[d["metrics"][str(h)]["direct"][metric] for h in range(1, 49)] for d in velocity])
            axis.plot(leads, hybrid.mean(0), color="#2b8a3e", label="Halo depth + velocity persistence")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
        for index, lead in enumerate(leads):
            rows.append({"lead_hours": lead, "metric": metric,
                         "v5b_fold_mean": direct[:, index].mean(),
                         "v5b_fold_sd": direct[:, index].std(ddof=1),
                         "persistence_fold_mean": persistence[:, index].mean()})
    axes[-1].set_xlabel("Forecast lead (hours)")
    axes[-1].set_xticks([0.5, 3, 6, 12, 18, 24])
    figure.suptitle("V5-B legacy-target metrics: NOT validated forecasts\nKnown chronology/sign defects; preserved historical results", fontsize=13)
    figure.tight_layout()
    figure.savefig(destination / "v5b_forecast_horizons.png", dpi=180)
    figure.savefig(destination / "v5b_forecast_horizons.pdf")
    plt.close(figure)
    with (destination / "v5b_forecast_horizons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated PNG, PDF, and CSV in {destination}")


if __name__ == "__main__":
    main()
