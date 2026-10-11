# Critical target-data audit — 6 October 2026

**Current model-development priority is data repair, not a larger architecture.**
Raw-target audit job `186233` completed successfully on Riker in 100 seconds.
It confirms that legacy labels use alphabetically ordered hydraulic snapshots
and positive-only discharge filtering. Previous stage-1/V5 performance numbers
on those assets are historical pipeline results, **not validated chronological
0.5–24-hour hydraulic forecasting skill**. They cannot determine the best model,
the value of graph communication, or the true velocity/persistence relationship.

## Confirmed chronology defect

The converter used string sorting on filenames with variable-width counters:
`H_01_00`, ..., `H_09_00`, `H_100_00`, ..., `H_309_00`, `H_30_00`, `H_310_00`.
It assigned monotonically increasing half-hour times to that nonchronological
sequence. Actual current dataset reads exactly match this conversion in all 24
audited patches (four training events × two times × three variables).

| Example | Assigned label time | Raw snapshot actually stored | Source-counter time |
|---|---:|---|---:|
| D029, origin index 239 | 120 h | H_30_00.dat | 15 h |
| D029, target index 240 | 120.5 h | H_310_00.dat | 155 h |
| D006, target index 393 | 197 h | H_44_00.dat | 22 h |
| D026, target index 349 | 175 h | H_40_00.dat | 20 h |

These source times assume the established half-hour output interval. The
counter-order finding itself does not depend on calendar-date interpretation.
All 40 raw archives have 480 files per component; alphabetical order differs
from numeric order at 471 positions per component. Filename ordering was
checked across all events; raw patch values were compared for four events,
not every pixel in every event.

The supposedly abrupt depth changes in the diagnostic examples disappear in
numerically reconstructed local time series:

![Legacy order versus numeric order](../../results/stage2_target_audit/job_186233/depth_time_order_audit.png)

This explains the audited false jumps. It is strong evidence for the periodic
horizon artifacts but not a complete audit of every historical error pattern.
Origin/previous states, targets, future-forcing alignment, change-aware sampling,
and reported physical lead times are affected. Changing just the evaluation
table or relabeling the timestamps does not repair training.

## Confirmed signed-component defect

`bin2array()` applied `array > threshold` with threshold zero to H, U and V.
That removes legitimate negative HU/HV as though they were missing data; the
loader fills the resulting missing values with zero. The audited legacy patches
contain **50,641 negative-component cell occurrences** that were clipped to
zero (not 50,641 unique domain cells). Negative flow information cannot be
recovered by reordering NetCDF snapshots.

Legacy variable names `output_velocity_x/y` actually represent HU/HV, supported
by existing TRITON source-semantic documentation. Old NetCDF unit annotations
also differed from those semantics; corrected assets must declare m²/s and
preserve both signs. Correct depth/forcing timing is required independently.

## Implemented safeguards

- Converter now sorts numeric output counters, rejects missing/duplicate steps,
  verifies sequential write order, and preserves signed U/V including zeros.
  The depth threshold behavior is unchanged; known -9999 nodata remains missing.
- New conversion metadata identifies `numeric-order-signed-components-v1`.
- The manifest builder records converter provenance and does not certify mixed
  or legacy inputs as corrected.
- New V5 training, standalone evaluation and graph learning checks reject
  legacy manifest metadata. Forensic audit remains explicitly read-only and
  can inspect old assets. Other legacy entry points have not been comprehensively
  hardened; do not use them for new forecasting experiments.
- Regression tests cover ordering past counters 9/99, duplicates/missing steps,
  signed components, and rejection of legacy targets. Existing datasets,
  checkpoints and numeric reports have **not** been overwritten or regenerated.

## Revised execution sequence

1. Validate a repaired pilot event in a **new** output location. Use numeric
   counters and signed raw values; compare multiple tiles/times against original
   binaries and verify output-time/forcing alignment and masks.
2. Before broad conversion, measure output size/runtime and available storage.
   Current converter extracts an entire event archive into staging; assess that
   footprint or implement streaming conversion before processing all 40 events.
3. Rebuild corrected hydraulic assets and dynamic manifest; rebuild timestamp-
   and transition-dependent sampling indices and any affected normalization.
   Audit static/peak products separately; do not assume every static asset needs
   regeneration, or that every dynamic-derived cache remains valid.
4. Freeze corrected-data provenance and unit/time tests. Train from scratch on
   the existing complete-event folds; old weights are not a trustworthy baseline
   for the corrected forecasting task.
5. Re-establish persistence and direct U-Net performance at all 48 leads.
6. Resume multi-tile loss/velocity formulation checks and the matched-capacity
   graph comparison on corrected data. Select depth/extent and full-hydraulic
   candidates with separate objectives and gates.

The original data contain useful simulation information; this finding does not
show that additional events or a Transformer are necessary. It shows that our
earlier architectural conclusions were confounded by preprocessing defects.

## Leadership message

“Our audit identified and confirmed temporal-ordering and signed-flow errors in
the training targets. We have corrected the converter and blocked further use
of the affected assets. We will validate rebuilt data and rerun the baseline
before making forecast-accuracy or architecture claims.”

Evidence: [machine-readable raw audit](../../results/stage2_target_audit/job_186233/TARGET_AUDIT.json).

## Authorized corrected single-event pilot

The next authorized run is **D029 only**, using
`workflows/stage2/19_corrected_event_pilot.sh` on Riker's CPU batch partition.
It writes to `processed_data/stage2_corrected_pilot/job_<job_id>/`, never the
legacy data location. A streaming ZIP reader holds one snapshot at a time,
avoiding roughly 250 GB of extraction staging per event. Compression level 1
is retained so the measured output footprint can inform a full rebuild.
The four-hour scheduler limit is a safety ceiling, not a runtime prediction.

The output remains `.nc.partial` until conversion and validation succeed.
Completion metadata and explicit numeric source counters are required for a
corrected manifest record. Independent patch validation uses blocks 0, 2264,
5000 and 6899 at counters 1, 9, 10, 30, 99, 100, 240, 241 and 480: 108
component/tile/time comparisons. Checks include exact float32 values, missing
masks, negative components, physical units, grid coordinates and relative
half-hour times. This is sampled validation, not an exhaustive all-pixel audit.
`PILOT_VALIDATION.json` records checks, source identity, converter hash, measured
runtime and output size. Failed runs do not produce a success report.

The original pilot authorization covered D029 only. The subsequent full-rebuild
authorization is recorded below; training remains pending validation. Neither the
active manifest nor legacy checkpoints are changed. Forcing timestamp provenance
was checked separately below: 480 tensor rows alone do not prove alignment.
The preserved CF epoch is labeled as compatibility metadata, not a verified
calendar date; the validated time axis is event-relative 0.5–240 hours.

Submitted job **186241** on 6 October 2026; confirmed running on `riker35`.
Local regression suite: **33 tests passed**. Pilot validation results are
pending; submission and passing synthetic tests are not evidence that the
full event has completed successfully.

## Independent forcing and simulator-time audit

The follow-up read-only audit on 6 October 2026 passed for **all 40 events**.
`stage2/audit_forcing_alignment.py` reconstructs each stored forcing tensor from
the hydrograph inside the original simulation archive. Every value in each
480 × 300 tensor matches exactly (maximum absolute error **0**). Source columns
are Loc1 through Loc300 in source order. The existing conversion uses float32
hydrograph values, linear interpolation, and float32 output at 0.5, 1.0, …,
240 hours; no existing tensor was changed.

The audit also reads each simulation's original configuration and nine output
checkpoint configurations (360 checkpoint checks total). Original runs start
at zero, last 864,000 seconds, and print every 1,800 seconds. Checkpoint IDs
1, 9, 10, 30, 99, 100, 240, 241 and 480 agree with their recorded simulator
times. This independently confirms the counter-to-relative-time mapping used
by the corrected converter; it is not just an assumption from file counts.

Evidence: [FORCING_ALIGNMENT.json](../../results/stage2_target_audit/forcing_alignment_20261006/FORCING_ALIGNMENT.json).

Scope limitations: calendar-date/CF epoch interpretation is not established.
Hydrograph comments declare discharge in cfs; simulator input-unit handling
needs a separate source-code audit before physical conservation losses use
these inputs. Passing this check validates existing forcing values, source
order and relative timing, not forecast-map skill or hydraulic-target repair.
The running pilot's report was programmed before this audit and may still say
`forcing_provenance_pending`; use this separate audit for that evidence.

## Authorized 40-event corrected rebuild (6 October 2026)

Pilot job **186241 timed out after four hours**, with depth and X discharge
fully converted and the last Y progress entry at 100/480 snapshots. Its partial
file remains preserved; no hydraulic validation report was produced.

The user explicitly authorized submitting all 40 rebuild jobs without waiting
for a successful pilot. This overlaps conversion and validation; it does not
establish that the new targets are already validated.

`workflows/stage2/20_corrected_event_rebuild.sh` runs a Riker CPU array,
tasks 1–40 mapped to D001–D040, with at most **four simultaneous events**.
Each task requests four CPUs, 32 GiB RAM, and a **12-hour limit**. Slurm
accepted the request in test-only mode; all **36 tests passed**. A preflight
verified 480 contiguous H/U/V source snapshots in every archive and a passed
independent forcing-audit record for every event.

Submitted as array job **186578** on 6 October 2026.

### Recovery status (7 October 2026)

Array **186578** finished with **D001–D008 completed and validated** (all 108
checks per event; approximately 5.1–5.3 hours and 5.6–8.1 GiB per event).
D009–D040 failed at Python startup in 1–3 seconds, before conversion. Their
logs report missing standard-library initialization/encoding support. Inspection
confirmed that `encodings/__init__.py`, `os.py`, and package source files are
missing from the shared `triton_andes` environment; the cause of those missing
files is not established.

The rebuild workflow now activates the intact
`/ccs/home/haoranniu/miniconda3/envs/triton` environment. Required imports and
all **seven conversion regression tests passed** there. The original environment
was not modified. Under the existing rebuild authorization, **D009–D040 only**
were resubmitted as array **187031**, still with four concurrent tasks and a
12-hour limit. Their outputs use `processed_data/stage2_corrected_rebuild/job_187031/`;
the eight validated outputs remain under `job_186578/`. Recovery logs use
`slurm_output/stage2-corrected-rebuild-187031_<task_id>.out` and `.err`.
Preparation remains incomplete until all remaining events pass validation.

Recovery startup exposed an additional OpenMP runtime conflict in tasks 9 and
11 from importing training libraries through the shared audit helpers. Plotting,
PyTorch, and pandas imports are now deferred to their respective command-line
entry points; conversion needs only NumPy and netCDF4. All seven conversion
tests passed again, and an import check confirmed the rebuild loads none of
those optional libraries. Tasks 9 and 11 were requeued within array 187031,
preserving its four-task concurrency ceiling. D010, D012, D013, and D014 were
confirmed running with growing `.nc.partial` outputs at this check.

### Rebuild completion verified (9 October 2026)

All **40 events have final NetCDF files and passed hydraulic validation reports**:
D001–D008 under `job_186578/`, D009–D040 under `job_187031/`.
All 32 recovery tasks report `COMPLETED` with exit code `0:0`; none remain
running or queued. Each event report contains 108 sampled raw comparisons with
zero maximum absolute error and references the passed forcing audit, for
**4,320 comparisons in total**. Combined corrected outputs occupy **267.78 GiB**.
These remain sampled comparisons rather than an exhaustive all-pixel audit.

The existing `timestamp_stage1/m3_dynamic_manifest` still lacks corrected
conversion metadata and points to the legacy data. A corrected manifest and
affected sampling/normalization inputs must be generated before retraining.
No corrected-data training results are established by rebuild completion.

Outputs go to `processed_data/stage2_corrected_rebuild/job_<array_id>/Dxxx/`.
Each event uses the streaming converter and the same 108 raw-patch checks as
the pilot, including signed components, masks, coordinates, units, and timing.
Only after hydraulic validation passes is `.nc.partial` renamed to `.nc` and
`HYDRAULIC_VALIDATION.json` written. A usable event requires both the final
NetCDF and its successful validation report. Reports reference and hash the
passed independent forcing audit. Sampling remains a limitation: this is not
an exhaustive all-pixel comparison.

Logs are `slurm_output/stage2-corrected-rebuild-<array_id>_<task_id>.out` and
the corresponding `.err`. Legacy data, active manifests, and checkpoints are
preserved. Sampling/normalization regeneration, retraining, and reevaluation
are still separate downstream work.
