# Slide-ready results: dynamic flood evaluation on stream tiles

Completed corrected-data evaluation of the original baseline, Halo and Halo + graph. The new fixed-5-hour localized B model is reported separately in [the completed B results](../../results/stage2_fixed5h_local/20261022_fold0/evaluation/RESULTS_SUMMARY.md); the historical tables below remain the original multi-lead comparison.

## Slide 1 — Evaluation setup

- Eight held-out events, fold 0, seed 45; validation-selected checkpoints.
- Candidate spatial subset: 258 tiles containing mapped streams. Score all valid cells in each selected tile.
- Select target periods with substantial joint depth and velocity change over the preceding 3 hours at both event and tile levels. Thresholds are training-derived positive-change 75th percentiles.
- Freeze the same eligible tile/target cases across models at each lead; use observed origin and origin−0.5 h hydraulic states, and forcing known through the target.
- At most 640 selected tile/target pairs across eight events; coverage decreases with lead because early targets lack earlier forecast origins.
- Raw predictions; wet definition 0.05 m. The 0.10 m visualization filter is not applied to these results.

## Slide 2 — Depth prediction

Wet-depth RMSE in metres; lower is better.

| Lead | Persistence | Baseline | Halo | Halo + graph | Events / tile-target pairs |
|---|---:|---:|---:|---:|---|
| 5 h | 0.349 | 0.239 | 0.238 | 0.206 | 8 / 544 |
| 6 h | 0.395 | 0.269 | 0.245 | 0.226 | 8 / 520 |
| 12 h | 0.683 | 0.517 | 0.431 | 0.518 | 7 / 392 |
| 24 h | 1.059 | 0.890 | 0.848 | 0.974 | 6 / 296 |

Suggested takeaway:

“On selected dynamic flood cases, Halo reduces wet-depth RMSE versus persistence by 32% at 5 h, 37% at 12 h and 20% at 24 h. Graph gives the lowest depth error at 5–6 h among these models, while Halo is best at 12 and 24 h.”

## Slide 3 — Flood extent and limitations

Wet F1 comes from the separate wet-probability head at threshold 0.5; higher is better.

| Lead | Persistence F1 | Halo F1 |
|---|---:|---:|
| 5 h | 0.906 | 0.896 |
| 12 h | 0.820 | 0.879 |
| 24 h | 0.726 | 0.781 |

- Extent skill is mixed: Halo loses to persistence at 5 h but improves on it at 12–24 h.
- Persistence has lower derived velocity RMSE than every learned model at all 48 evaluated leads.
- Raw depth maps still show excessive shallow wet area; a good wet-head F1 does not establish accurate depth-defined extent.
- The 24 h Halo wet-depth error is 0.848 m despite the relative improvement. These results do not establish simulator-replacement accuracy.
- Conditional stream-tile results do not describe whole-domain performance. Different leads have different cohorts; single fold/seed and known future forcing limit generalization.

Suggested closing message:

“The current models capture useful flood-depth evolution, but excessive wet extent and velocity errors prevent a claim of a dependable hydraulic surrogate. The next experiment is a fixed-5-hour, localized-forcing Halo trained for depth and extent.”

## Source artifacts

- [Original comparison](../../results/stage2_stream_flood/20261011_fold0/comparison/COMPARISON.md)
- [All-lead CSV](../../results/stage2_stream_flood/20261011_fold0/comparison/comparison.csv)
- [Absolute metric plots](../../results/stage2_stream_flood/20261011_fold0/comparison/plots/absolute_metrics.png)
- [Coverage by lead](../../results/stage2_stream_flood/20261011_fold0/comparison/plots/coverage_by_lead.png)
- [Detailed protocol](STREAM_FLOOD_EVALUATION.md)
- [New B experiment](FIXED5H_LOCAL_DEPTH_EXTENT.md)

Do not substitute the legacy pre-correction leadership briefing numbers or the whole-domain D005 visualization statistics for this conditional evaluation.
