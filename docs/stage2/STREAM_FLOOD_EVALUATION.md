# Dynamic Flood Evaluation on Stream Tiles

**Task ID:** `dynamic_stream_flood_eval`.

**Current status (verified 21 October 2026): COMPLETE.** All three evaluation
tasks and the comparison/plot job completed successfully with Slurm exit code
`0:0`. Baseline 189238_0 took 32m50s; halo 189238_1 took 41m26s;
halo+graph 189238_2 took 6h29m14s. Comparison/plot job 189240 took 13s.
No jobs are currently queued or running for this user. All metrics, comparison
CSV/Markdown and four PNG/PDF plot sets are present. Shared cohort identity,
per-lead coverage and persistence were verified across all models and 48 leads.

Final artifacts:

- [Comparison table](../../results/stage2_stream_flood/20261011_fold0/comparison/COMPARISON.md)
- [Comparison CSV](../../results/stage2_stream_flood/20261011_fold0/comparison/comparison.csv)
- [Absolute metrics](../../results/stage2_stream_flood/20261011_fold0/comparison/plots/absolute_metrics.png)
- [Coverage by lead](../../results/stage2_stream_flood/20261011_fold0/comparison/plots/coverage_by_lead.png)
- [All plots and interpretation](../../results/stage2_stream_flood/20261011_fold0/comparison/plots/PLOTS.md)

**Final depth results (wet-depth RMSE, metres):**

| Lead | Persistence | Baseline | Halo | Halo + graph |
|---|---:|---:|---:|---:|
| 6 h | 0.3951 | 0.2687 | 0.2452 | 0.2257 |
| 12 h | 0.6833 | 0.5174 | 0.4309 | 0.5176 |
| 24 h | 1.0592 | 0.8902 | 0.8482 | 0.9743 |

Halo reduces depth RMSE against persistence by 37.9%, 36.9% and 19.9% at
6, 12 and 24 h respectively. Graph is strongest for depth at 6 h among
these models, but halo is strongest at 12 and 24 h. Graph runtime is about
9.4 times halo runtime. At 24 h derived velocity RMSE is 0.07759 m/s for
persistence, 0.09027 for baseline, 0.08437 for halo and 0.07826 for graph:
none beats persistence at that lead. At 0.5 h persistence also beats all
three learned models for depth. These are mixed results, not a universal
model winner or evidence of operational readiness.

This evaluates existing checkpoints on dynamic stream-tile cases; it is
not a new training run. The changed selection does not establish improvement
over the earlier representative evaluation. Lead-dependent cohort composition,
a single fold/seed and known forcing through target time limit interpretation.
Suggested next analysis: per-event and wetting/recession failure cases, plus
comparison on a common target subset across leads, before choosing another
training experiment.

All source scans completed. The initial all-leads cohort failed (189221),
cancelling its dependent jobs. The revised early-target cohort was built
locally and includes all eight events.

The revised policy is frozen in
`results/stage2_stream_flood/20261011_fold0/RULE_available_history.json`.
It references the original `RULE.json` digest and preserves all change
thresholds. Only forecast-lead eligibility changes; source scans are reused.

The user clarified that inundation alone is insufficient: first select periods
with substantial hydraulic changes in an event's time series, then select
stream tiles where local depth and velocity change substantially as well.
The subsequent user clarification specifies **target-first evaluation**:
identify flood-active target times and tiles, then use previous observed
hydraulic states to predict those targets.

## Adopted initial selection rule

Use **3-hour changes ending at the target time**. Compute thresholds from
the 75th percentile of positive training changes; validation checks coverage
only. This is a conditional diagnostic of upper-quartile hydraulic dynamics,
not an independently established extreme-flood definition.

1. **Event/time gate:** over all valid cells in the 258 stream-containing tiles,
   require both wet-union depth-change RMS >= **0.11210617663982 m** and
   common-wet velocity-vector change RMS >= **0.010411679756663374 m/s**.
   These are pixel-weighted event summary measures over stream tiles, not
   full-domain summaries or averages of signed changes.
2. **Tile/time gate:** inside those event periods, require depth-change RMS
   >= **0.1126872249587439 m** and velocity-vector change RMS
   >= **0.0075861547089562 m/s**. Both endpoints of velocity comparisons have
   depth >=0.1 m and finite HU/HV. Require at least **1% of the tile's valid
   cells** to support both the wet-union depth comparison and the common-wet
   velocity comparison (rounded up to a whole cell). This is minimum eligible
   support, not proof that 1% of cells individually exceed a change threshold.
3. Keep flood targets with available two-state input history for at least
   the shortest lead (minimum index 2; the 3 h change gate itself requires
   target index >=6). For target index T and lead L, evaluate only when
   **T − L >=1**: the model then observes state T−L and T−L−1. Freeze
   target/tile pairs once; all models and persistence use the identical
   eligible subset at each lead. Do not fabricate states before event start.
   The available subset can differ across leads.
4. Sample a deterministic, event-balanced initial evaluation budget from all
   qualifying pairs. Report candidate counts and selected counts; retain the
   complete qualifying list for expanded evaluation. The frozen budget is
   up to 10 eligible target times per event and eight eligible tiles per time,
   spread evenly by rank along the sorted eligible time and tile lists.

Validation coverage of this adopted rule (all half-hour targets):

| Event | Eligible target times | Target/tile pairs | Distinct stream tiles |
|---|---|---|---|
| D001 | 28 | 1,336 | 88 |
| D003 | 64 | 4,280 | 155 |
| D008 | 34 | 1,470 | 74 |
| D010 | 87 | 6,665 | 174 |
| D016 | 44 | 2,193 | 91 |
| D025 | 69 | 4,766 | 158 |
| D030 | 65 | 3,877 | 134 |

These counts establish coverage, not predictive skill. No test-event hydraulic
characterization or model evaluation is performed by this proposal. Velocity
changes are small in absolute units even at the upper quartile; thresholds
must be presented numerically rather than described as necessarily extreme.

Preparation outputs: [summary](../../results/stage2_stream_flood/20261011_fold0/preparation/summary/SUMMARY.md),
[change quantiles](../../results/stage2_stream_flood/20261011_fold0/preparation/summary/change_quantiles.csv),
and [observed-change time-series plot](../../results/stage2_stream_flood/20261011_fold0/preparation/summary/observed_change_timeseries.png).

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

## Target-first forecast pairing

The agreed direction is to select a list of `(event, target_time, stream_tile)`
cases based on corrected hydraulic data and the frozen thresholds above.
Each target is evaluated at supported half-hour leads from 0.5 to 24 h,
with a **different earlier forecast origin**:

`origin_time = target_time - lead`; the two observed hydraulic states are at
`origin_time` and `origin_time - 0.5 h`. Intermediate states between origin and
target are not supplied. The existing known-through-target forcing assumption
is preserved. All models and persistence use identical targets and origins.

For example, a selected target at event-relative 100 h is predicted from 94 h
and 93.5 h at a 6-hour lead, or from 76 h and 75.5 h at a 24-hour lead.
Target index >=49 supports the full lead range; earlier targets support
shorter leads. All origins and older states must remain within the event's
corrected time series.

Selecting known flood-active/dynamic targets is **retrospective conditional
evaluation**: future truth chooses what is scored, but is never a model input.
This measures performance during actual flooding, not the ability to decide
in advance where/when a flood will occur. The same target cohort across leads
also holds the scored future hydraulic states fixed as forecast difficulty
changes with lead.

The following alternatives were considered during preparation; the adopted
initial run uses the fixed 3 h interval, joint depth/velocity gates and support
rule above:

- Change interval: adjacent half-hour states, or a wider interval such as 3 h.
- Exact target-period selection: combine inundation and recent hydraulic
  changes leading up to the target; specify the fixed change interval and
  event/time and tile/time thresholds. Model errors/predictions cannot select
  examples. Do not select a different cohort based on each model's skill.
- Require **both** depth and velocity change, or include either with separate
  depth-only, velocity-only and joint-change reporting. A strict intersection
  follows the user's stated interest but can exclude important wetting fronts
  and needs explicit coverage reporting.
- Velocity changes need a reliable depth denominator. Stable wet cells can be
  assessed directly; newly wet/dry cells need a separate transition definition,
  rather than division by tiny depths or silently discarding the front.

## Fixed comparison settings

- Same eight held-out events and validation-selected checkpoints as the
  corrected comparison: D005, D014, D018, D023, D024, D028, D033, D037.
- A stream tile has at least one valid static `stream_mask > 0.5` cell.
- Freeze explicit event/target/tile batches for all models. Derive the earlier
  origin independently for each lead. Preserve two-state history and
  per-lead origin eligibility. Select up to 10 target times per event, evenly spaced by
  rank among sorted eligible times. At each time select up to eight tiles,
  evenly spaced by rank among sorted eligible block indices. This is at most
  80 batches / 640 tile-target pairs, with partial batches allowed. It samples
  the eligible cohort; it does not evaluate every qualifying tile/time pair.
  Any event with no eligible cases stops the pipeline; thresholds are not
  automatically relaxed.
- Keep original wet probability threshold 0.5, wet depth threshold 0.05 m,
  learned discharge policy, model inputs and metric definitions.

The earlier proposal of 1% non-stream inundation at >=0.05 m is **not adopted**:
it detects wetness, not necessarily dynamics. Report origin-to-target depth change,
newly wet/dry counts and origin/target wet-cell counts at each lead alongside
the metrics, so the contribution of persistence can be interpreted.

## Outputs and interpretation

Proposed root: `results/stage2_stream_flood/20261011_fold0/`.

- `preparation/`: layout, stream-tile table, training/validation per-tile state
  and change time series, and change-distribution summaries (no model predictions).
- `cohort/`: frozen target/tile candidate-batch shards;
  merged `cohort.json` with source identity, thresholds, full selected centers,
  counts and a digest, plus `coverage_by_lead.csv`.
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

Characterization has started: `27_dynamic_stream_prepare.sh` processes all
half-hour states of the 25 training and seven validation events for the 258
stream-containing tiles. Each of 32 CPU tasks requests four CPUs, 32 GiB and
four hours; at most four tasks run concurrently. It reads source spatial chunks
once across all times and stores scalar summaries, preserving original data.
Changes are measured over **0.5, 3 and 6 h**. Depth changes use cells wet at
either endpoint (>=0.05 m). Velocity-vector changes require finite HU/HV and
depth >=0.1 m at both endpoints; newly wet/dry cells are reported separately.
The 0.1 m diagnostic rule avoids tiny velocity denominators and is not a
substantial-change threshold or a change to the original evaluation metrics.

`28_dynamic_stream_summarize.sh` runs after all scans succeed and writes
training/validation quantiles, illustrative joint depth/velocity threshold
coverage, event time-series plots and `preparation/summary/SUMMARY.md`.
Percentile candidates are derived only from training data and do not select
the final evaluation cohort. Six numerical regression tests passed, plus a
real corrected-data one-spatial-chunk smoke on training event D002. Both CPU
requests passed Slurm test-only validation before submission.

The final rules were frozen before held-out processing. Submitted chain:

| Job | Purpose | Dependency |
|---|---|---|
| 189220 | Eight held-out CPU scans, up to four concurrent | None |
| 189221 | Validate scans and freeze target/tile cohort; report coverage | afterok:189220 |
| 189222 | Three GPU evaluations: baseline, halo, halo_graph | afterok:189221 |
| 189223 | Verify shared cohort/persistence, compare and plot | afterok:189222 |

All requests passed Slurm test-only checks. Eight preparation/cohort tests and
26 existing Stage 2 tests passed. A numerical target-first regression verifies
that target index 49 is unchanged for leads 1 and 48, with corresponding
persistence depth errors of 1 and 48 on a linear synthetic time series.
The GPU tasks each allow 24 h because scattered stream centers can increase
graph context size. This is evaluation of existing checkpoints; no retraining.
Dependent jobs are cancelled if prerequisites fail.

The evaluator records the cohort SHA-256, frozen rule, coverage, per-event
metrics, and lead-specific observed origin/target wet counts, newly wet/dry
counts and wet-union depth-change RMS. The 3 h selection statistic is distinct
from these lead-specific changes. Full valid center-tile cells are scored;
neighbor tiles provide graph context but are not scored. Comparison verifies
that persistence agrees across all three models. Outputs remain under the
separate conditional-results root; original results and shared-code snapshot
are preserved.


## Held-out coverage audit after cohort failure

| Event | Eligible target times for 24 h lead | Eligible tile/target pairs |
|---|---:|---:|
| D005 | 86 | 7,254 |
| D014 | 0 | 0 |
| D018 | 68 | 4,664 |
| D023 | 19 | 960 |
| D024 | 57 | 4,878 |
| D028 | 0 | 0 |
| D033 | 53 | 3,847 |
| D037 | 41 | 1,855 |

D014 passes the event/time change gate only at target indices 6–38; D028
only at indices 6–18. Both precede minimum target index 49, which is needed
to provide two historical states for a 24 h forecast. This is a coverage
constraint, not a model evaluation failure. The original fixed rule has not
been preserved for audit. This describes the initial failed attempt; the
revised available-origin-history policy and replacement jobs are recorded below.

Initial recommendation (superseded): evaluate the six eligible events using unchanged
thresholds and shared targets across all 48 leads, reporting D014/D028 as
ineligible. This was the recommendation before the user clarified that early targets
should contribute to shorter supported leads. A separate shorter-lead diagnostic can cover earlier
flood dynamics, with an explicit different cohort and no direct comparison
to the fixed 24 h cohort.


## Revised early-target evaluation

The model never requires 24 h of hydraulic input history. It uses **two states
separated by 0.5 h** at forecast origin and origin−0.5 h. A **24 h forecast lead**
requires an origin 24 h before its target; an early target can instead be
predicted at shorter leads. The original all-leads intersection unnecessarily
excluded entire events whose dynamics occur early. The revised evaluator
filters the frozen target cohort independently at each lead according to
available origin history, with identical cases across all models at that lead.

All eight events now have qualifying targets. We selected 10 times and eight
tiles per time per event: 80 batches and 640 tile/target pairs in total.

| Lead (h) | Events | Target batches | Tile/target pairs |
|---|---:|---:|---:|
| 0.5 | 8 | 80 | 640 |
| 3 | 8 | 72 | 576 |
| 6 | 8 | 65 | 520 |
| 12 | 7 | 49 | 392 |
| 24 | 6 | 37 | 296 |

At unsupported event/lead combinations, per-event metrics are explicitly
`null`, with zero scored tile/target pairs. Aggregate metrics include only
supported cases, and carry per-lead event IDs and counts. Lead curves now
combine changing forecast difficulty with changing cohort composition;
compare model skill at the same lead and inspect the accompanying coverage
plot. All 48 lead coverage records are saved in `cohort/coverage_by_lead.csv`.

Nine preparation/cohort tests and 26 existing Stage 2 tests pass. The new
regression checks the boundary at target index 6 with two-state history:
lead 5 is valid (origin 1, older state 0), lead 6 is skipped. A later target
remains valid at lead 48. Plain and graph evaluations have identical
persistence results and coverage. Coverage-plot rendering was also checked.

The completed cohort was source-identity validated before resubmission.
Replacement array 189238 runs the three existing models; job 189240 verifies
matching cohort and per-lead coverage, then writes CSV/Markdown comparisons
and absolute-metric, baseline-gain, physical-score and coverage PNG/PDF plots.
The original failed-job IDs remain in `JOBS_initial_failed.json` for audit.
