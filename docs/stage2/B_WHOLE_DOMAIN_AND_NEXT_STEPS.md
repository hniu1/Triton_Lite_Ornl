# B whole-domain visualization and next experiments

Prepared 2026-10-22. **User approved inference and plotting; job 193940 submitted
and confirmed RUNNING on riker-gpu1. Completion is not yet verified.**
This single job performs inference followed by plotting; no training submitted.
User approval is required before each new job, including retries and experiments.

## Prepared visualization

Script: `workflows/stage2/40_B_whole_domain_maps.sh`.
Checkpoint: `results/stage2_fixed5h_local/20261022_fold0/seed_45/best_model.pt`.
Output: `results/stage2_whole_domain_halo/20261022_D005_B_fixed5h_flood/`.

Reuse the frozen D005 plan from the earlier Halo maps: 28 flood-active target
times at two-hour spacing within selected windows (6–18 h and 66–106 h),
all 6,900 domain tiles. Each target uses a fresh observed origin 5 h earlier
and its preceding half-hour hydraulic state; forcing is supplied through the
target. These are direct forecasts, without recursive predicted-state input.
Keep the plan fixed to compare the two models; it was originally selected
using already-inspected test results and is exploratory.

Deliver native depth/truth/wet-probability NetCDF, native-cell metrics CSV,
28 reference/prediction/error PNGs and one multipage PDF. Verify native affine
coordinates and use north up, east right with projected HUC12 boundaries.
Both depth panels retain the original 0.05 m wet threshold. Do not transfer
the old model's validation-selected 0.10 m cutoff to B. Display settings are
explicitly marked as uncalibrated in `workflows/stage2/B_whole_domain_display.json`.
Blue means positive raw prediction-minus-truth error, red negative, fixed
±0.3 m range with saturated larger errors. Error is computed before masking;
4×4 display cells show the signed native error of largest absolute magnitude.
Depth panels use 4×4 maxima, which can visually widen narrow streams; metrics
use native cells. B is identified in figure titles.

Proposed resources: one L40S GPU, 16 CPUs, cluster-required 375104 MB GPU
memory reservation, 4 h time limit. Previous Halo inference took about
22 minutes; B runtime is not yet measured. Allow additional time for plotting
and I/O. No training is involved. Existing output directories are preserved.

## Prioritized improvement plan (proposals, not submitted jobs)

1. Diagnose first. Compare B and old Halo on identical whole-domain targets;
   break errors down by rising/peak/recession phase, HUC, depth and newly wet,
   persistently wet and drying cells. Do training/validation diagnostics before
   choosing changes; test maps can illustrate failure modes but must not be
   used to tune thresholds. D005 selected-tile wet RMSE regressed from 0.164
   to 0.200 m, while the eight-event aggregate improved 0.238 to 0.218 m.
   Determine whether remaining errors are timing, extent, depth amplitude or
   tile-edge discontinuities before deciding on architecture changes.
2. Keep direct 5 h depth/extent prediction as the controlled baseline. First
   candidate: train the predicted depth and wet head to agree, with explicit
   attention to shallow false positives and newly flooded cells. Tune loss
   balance and sampling on validation data; retain representative cases so
   emphasizing rapid-change windows does not sacrifice ordinary conditions.
   Select one concrete change after diagnostics and request approval before
   training. Evaluate MAE, RMSE, precision/recall/F1 and transition subsets.
3. Improve localized forcing if diagnostics support it. B currently compresses
   local forcing into one fixed weighted curve (70% same HUC /30% upstream
   when both are available). Preserve separate same-HUC and upstream signals,
   encode connectivity and possible travel delays, then learn their relative
   influence. Audit point/tile routing first. More flexible attention or a
   graph is a later option, not an assumed cure.
4. After a validation-selected improvement, confirm it on a fresh fold or
   additional independent events. Current test results have already been
   inspected. Keep matched origins and persistence comparisons. Do not add
   velocity, multiple leads or recursion until depth/extent improvements are
   stable. Additional epochs alone are not an established solution.

B changed localization, lead training and output supervision together, so the
existing comparison cannot attribute the improvement to any one factor.
The selected-flood 0.150 m MAE /0.218 m RMSE and mixed extent performance do
not yet establish a dependable simulator replacement. Improvements proposed
above are hypotheses to test, not promised gains.

## Additional held-out event: D028

User requested one additional event visualization. Submitted job **193948**
using `workflows/stage2/41_B_D028_whole_domain_maps.sh`. The unchanged
event-level joint depth/velocity change gate selects only two even-hour
targets with available 5 h history: **6 h and 8 h**. All 6,900 tiles are
included; no local stream-tile filter is applied. Frozen plan:
`results/stage2_whole_domain_halo/plans/D028_B_fixed5h_flood.json`.
Output: `results/stage2_whole_domain_halo/20261022_D028_B_fixed5h_flood/`.
Same B checkpoint, direct 5 h lead, north-up orientation, 5 cm display threshold
and blue-positive ±0.3 m raw error scale as D005. Inference and plotting only;
completion not yet verified.

## Additional held-out event: D014

User requested another event visualization. Submitted job **193951** using
`workflows/stage2/42_B_D014_whole_domain_maps.sh`. Frozen plan:
`results/stage2_whole_domain_halo/plans/D014_B_fixed5h_flood.json`.
The existing 3 h joint event depth/velocity change gate selects seven even-hour
targets: **6, 8, 10, 12, 14, 16, 18 h**. These are rapid-change targets,
not a claim to cover every flooded timestamp. All domain tiles, fixed direct
5 h lead, same B checkpoint and display settings as D005/D028.
Output: `results/stage2_whole_domain_halo/20261022_D014_B_fixed5h_flood/`.
Inference and plotting only; completion not yet verified.
