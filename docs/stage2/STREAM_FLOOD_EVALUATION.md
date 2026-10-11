# Dynamic-flood evaluation on stream-containing tiles — draft

**Status: definition under discussion; no cohort or scheduler jobs submitted.**
The user clarified that inundation alone is insufficient: first select periods
with substantial hydraulic changes in an event's time series, then select
stream tiles where local depth and velocity change substantially as well.

## Purpose and scope

Requested on 11 October 2026: compare the existing corrected-data baseline,
halo, halo + graph, and full-state persistence during flooding on tiles that
contain streams. No retraining or checkpoint selection is performed. The
previous representative-domain evaluation remains a separate reference.

The static stream mask identifies **258 of 6,900 tiles** with at least one
stream cell inside the valid tile mask. Metrics score the **whole valid tile**,
not just stream pixels. Graph neighbors can include other tiles as context;
only eligible centers enter metrics.

## Proposed two-level selection

1. **Event/time gate:** detect periods of substantial hydraulic evolution
   using corrected hydraulic time series, not forcing alone. Candidate summaries
   are wet area, total water-depth volume proxy, and the fraction of stream tiles
   with substantial depth/velocity changes. Spatial masks and physical units
   must be checked before computing area/volume summaries.
2. **Tile/time gate:** inside those periods, consider only tiles containing a
   valid stream cell. Measure local depth changes, newly wet/dry extent and
   derived velocity-vector changes over a fixed time interval. Pooling only
   signed changes can cancel rising/falling or opposing flows, so summarize
   absolute changes or squared vector changes.
3. **Substantial-change thresholds:** estimate distributions from training
   events and verify cohort coverage on validation events. Document thresholds
   before selecting the test cohort; do not choose thresholds to improve model
   scores. Report selected counts and exclusions per event/time/tile.

The following choices remain unresolved:

- Change interval: adjacent half-hour states, or a wider interval such as 3 h.
- Past/origin-based changes versus origin-to-future-target changes. Past-based
  selection answers how forecasts perform when issued during observed dynamics.
  Future-based selection answers how they perform on actual changing targets
  and must be labeled retrospective/target-conditioned. Neither rule may use
  model error or model predictions to choose examples.
- Require **both** depth and velocity change, or include either with separate
  depth-only, velocity-only and joint-change reporting. A strict intersection
  follows the user's stated interest but can exclude important wetting fronts
  and needs explicit coverage reporting.
- Velocity changes need a reliable depth denominator. Stable wet cells can be
  assessed directly; newly wet/dry cells need a separate transition definition,
  rather than division by tiny depths or silently discarding the front.
- An origin-based cohort can be shared across all 48 leads. Target-conditioned
  filtering produces different cohorts per lead, which must be disclosed.

## Fixed comparison settings

- Same eight held-out events and validation-selected checkpoints as the
  corrected comparison: D005, D014, D018, D023, D024, D028, D033, D037.
- A stream tile has at least one valid static `stream_mask > 0.5` cell.
- Once selection is agreed, freeze explicit event/origin/tile batches for all
  models. Preserve two-state history and 24-hour target eligibility. Sampling
  stride and budget remain to be chosen after measuring eligible coverage;
  six-hour origin spacing could miss short-lived dynamics.
- Keep original wet probability threshold 0.5, wet depth threshold 0.05 m,
  learned discharge policy, model inputs and metric definitions.

The earlier proposal of 1% non-stream inundation at >=0.05 m is **not adopted**:
it detects wetness, not necessarily dynamics. Report origin-to-target depth change,
newly wet/dry counts and origin/target wet-cell counts at each lead alongside
the metrics, so the contribution of persistence can be interpreted.

## Outputs and interpretation

Proposed root: `results/stage2_stream_flood/20261011_fold0/`.

- `cohort/`: per-event scanned time series and frozen candidate-batch shards;
  merged `cohort.json` with source identity, thresholds, full selected centers,
  counts and a digest.
- `{baseline,halo,halo_graph}/test_stream_flood_metrics.json`: all 48 leads,
  persistence comparisons, per-event metrics, and change statistics.
- `comparison/`: Markdown table, CSV and PNG/PDF plots generated only after
  all evaluations succeed and matching cohort/persistence checks pass.

This will be **dynamic-flood, stream-tile conditional performance**, not domain-wide
accuracy. It excludes flooding on tiles without mapped streams and forecasts
outside the selected dynamic periods. It is a diagnostic on the already-used
test fold, not independent confirmation or an operational reliability claim.
No expected improvement is assumed; failures remain part of the comparison.

## Execution

After the definition is agreed, characterize change distributions on training
and validation events and freeze the selection rules. Then a CPU array can
independently scan the eight held-out events. Merge only after all
shards succeed, then run one GPU evaluation per existing model. The final
comparison and plots depend on all evaluations succeeding. Scheduler requests
and regression tests are checked before submission; job IDs are recorded here
after submission. The original training and evaluation artifacts are preserved.
