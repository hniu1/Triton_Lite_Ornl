# Halo map mask calibration

Status (verified 22 October 2026): **COMPLETE and ready for new plots.**
Validation calibration **193841** and D005 audit/preview job **193842** completed.
Two revised previews are available; the original 28-figure series and native
prediction arrays are unchanged.

## Recommendation

Use a **0.10 m predicted-depth cutoff**, with **no wet-probability mask**.
This is the best of the tested candidates under the validation recall
safeguards, not a universal physical flood-depth threshold. Keep the reference
wet definition at **0.05 m** so genuine shallow flooding remains visible and
is not relabelled dry to improve the scores.

- Validation: **50.3% fewer false wet cells**, **96.7% wet recall**;
  worst individual validation-frame recall **91.7%**.
- D005 independent audit of the frozen mask: **55.3% fewer false wet cells**,
  **97.0% wet recall**; precision rises from **52.8% to 70.9%**.
- D005 worst-frame recall is **89.8% at hour 66**, so the filter still misses
  some genuine flooding and is not uniformly safe at every test instant.
- No probability-gated candidate met all validation frame-recall safeguards;
  the wet-head mask is not recommended for these plots.
- Revised errors: **blue overprediction, red underprediction, ±0.3 m** with
  over-range colors indicated. The native D005 wet-union absolute-error 95th
  percentile is approximately **0.25 m**. Raw error remains visible at cells
  removed from the prediction-depth display; raw RMSE is unchanged.

[Threshold report](../../results/stage2_whole_domain_halo/20261022_mask_calibration/REPORT.md)
contains two checked previews at target hours **6** and **76**, plus links to
all native-resolution mask scores. Error display now shows the signed native
error with largest absolute value in each 4×4 display cell, rather than
subtracting independently pooled depths. This avoids hiding local errors when
truth and prediction maxima occur at different native locations.

## Question and interpretation

The full-domain Halo maps show excessive wet width. Native-depth checks at
D005 hours 6, 76 and 106 confirm that many falsely wet cells are shallow.
At hour 76, 54.8% of false wet cells are below 0.10 m and 90.8% below 0.20 m.
Four-by-four maximum pooling also enlarges the displayed footprint of narrow
wet features. A mask can improve presentation or flood-extent classification,
but does not repair the continuous depth model. Filtering must not be claimed
as an improvement in the original raw-depth RMSE.

## Validation design

Use the existing corrected-data Halo checkpoint (validation-selected epoch 6,
seed 45) at the current 5 h forecast lead. Select thresholds on validation
only; use D005 test maps strictly for a subsequent frozen-setting audit.

- Validation events: D001, D003, D008, D010, D016, D025, D030.
- Three target times per event, at evenly spaced ranks among eligible even-hour
  targets passing the existing event-level joint 3 h depth/velocity-change gate.
- Every valid cell in all 6,900 tiles at each target: **21 whole-domain frames**.
  There is no stream-tile sampling bias or dry-background subsampling.
- Use observed hydraulic states at target−5 h and target−5.5 h. Original
  training normalization, halo context and known-through-target forcing remain.
- Depth candidates: 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25,
  0.30, 0.40 and 0.50 m.
- Wet-probability candidates: no probability mask, then 0.05, 0.10, 0.20,
  0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 and 0.95.
- Candidate extent: predicted depth >= depth cutoff AND, where enabled,
  model wet probability >= probability cutoff.

**Truth is always depth >=0.05 m.** Raising the display threshold does not
relabel genuine shallow flooding as dry during scoring. Confusion counts are
computed at native raster resolution, before display pooling. Histograms allow
all 132 candidate masks to be compared from the same predictions.

## Selection criterion

Maximize the equally weighted mean of the seven per-event wet-extent F1
scores, among candidates with:

1. pooled wet recall >=95%, and
2. wet recall >=90% in every validation event, and
3. wet recall >=90% in every individual validation frame.

The third condition is a validation-only safety refinement applied from saved
histograms before the D005 audit; it prevents early-flood omissions from being
hidden by an event average. The original event-level selection remains in
`CALIBRATION.json`; the final refinement is saved as `CALIBRATION_FRAME_SAFE.json`.

These recall constraints protect true flooding while reducing false wet area.
F1 balances precision and recall; dry-cell accuracy is not the selection
objective. Ties prefer higher pooled recall, then less restrictive thresholds.
Report the best depth-only setting and best probability-gated setting
separately. If no candidate meets the constraints, report that explicitly and
do not declare plot settings ready or silently relax constraints.

This estimates a defensible mask for these validation events and sampled flood
phases. It is not a universal hydrologic wetness threshold or an operational
calibration. The 5 h lead itself was chosen using already inspected conditional
test diagnostics; this limits claims of independent assessment.

## D005 audit and error display

After selection, apply the frozen settings to all 28 saved D005 target maps.
Report TP/FP/FN counts, precision, recall and F1 per target and in aggregate,
including early-flood failures. D005 does not choose or revise the threshold.
Keep original raw-depth metrics and maps available.

The proposed error display is prediction minus reference, with **blue for
positive overprediction**, **red for negative underprediction**, and a fixed
**±0.3 m** range. Larger errors should saturate visibly with colorbar extension
arrows; stored values are not clipped. The audit reports how much native
wet-union error falls outside this range. Error maps should retain raw errors,
including locations suppressed by the display mask, so thresholding does not
conceal false flooding. Native-resolution zooms can distinguish physical
prediction width from the maximum-pooling display effect.

## Outputs and reproducibility

Root: `results/stage2_whole_domain_halo/20261022_mask_calibration/`

- `PLAN.json`: frozen checkpoint/config/rule hashes, validation targets,
  candidate thresholds and selection criterion.
- `CALIBRATION.json`, `validation_candidates.csv`: initial validation scores and
  selected settings, including separate depth-only and gated choices.
- `CALIBRATION_FRAME_SAFE.json`, `validation_candidates_frame_safe.csv`: final
  recommendation with the additional per-frame recall protection.
- `validation_histograms.npz`, `validation_frames.json`: raw native counts.
- `D005_mask_audit.csv`: frozen-mask test scores by target.
- `READY.json`: final settings, error color convention and audit summary.
- `REPORT.md`: concise threshold recommendation and tradeoffs.
- `preview/`: two PNG previews and a PDF using the frozen mask, blue-positive
  raw errors and fixed ±0.3 m limits.
- `JOBS.json`: calibration and audit job IDs.

Implementation: `stage2/calibrate_map_mask.py`; workflows
`34_halo_map_mask_calibrate.sh` and `35_halo_map_mask_audit.sh`.
Three calibration regression tests passed for exact threshold-boundary counts,
recall-constrained selection and preservation of the shallow-water truth
labels. Six whole-domain/display tests passed, including native-mask-before-pooling
and preservation of the largest raw error sign. A synthetic revised-render
smoke passed while preserving the original manifest. Both final resource
requests passed Slurm test-only validation. The GPU request uses the cluster's
required 375104 MiB per GPU and a four-hour allowance; the CPU audit requests
four CPUs, 32 GiB and two hours. No retraining or model checkpoint changes.
