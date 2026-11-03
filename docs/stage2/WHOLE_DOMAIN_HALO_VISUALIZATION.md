# Whole-domain Halo visualization

Status (verified 22 October 2026): job **193453** completed. All 28 maps,
native arrays, per-frame metrics and the multipage PDF are present; manifest
status is `complete`. The inference stage took 1,293.6 s (about 21.6 min).

A separate [mask calibration](HALO_MAP_MASK_CALIBRATION.md) now evaluates
shallow-water filtering and wet-probability masks before changing the plots.

The requested output is a sequence of whole-domain maps every two event-relative
hours during dynamic flood periods. The trained Halo checkpoint is reused;
there is no retraining. We move observed forecast origins through the event
and keep the forecast lead fixed. Predictions are independent direct forecasts,
not a rollout of previous predictions.

## Event, lead and target times

The initial event is held-out **D005**, a large flood event in the corrected
fold-0 test split. Use all **6,900 tiles**, including non-stream tiles. The model
raster is **7,976 rows × 5,474 columns**, with approximately 9.506 m cells.

Use a **5 h forecast lead**, which has the lowest Halo true-wet-cell depth RMSE
among the 48 leads in the existing conditional evaluation: **0.2380 m**, versus
**0.3489 m** for persistence (31.8% lower). This is an exploratory lead choice
based on already-inspected test results, not an independent demonstration that
5 h is optimal. Different leads have different target coverage in that evaluation.

Retain every even target hour passing the existing event-level flood-activity
gate: both 3 h wet-union depth-change RMS and common-wet velocity-vector-change
RMS meet the training-derived 75th-percentile thresholds. Preserve those rules;
remove the stream-tile/local-change restriction for these whole-domain maps.
Targets must have both observed input states available at a 5 h lead.

**28 target maps:** 6, 8, 10, 12, 14, 16, 18 h; then 66, 68, …, 106 h.
These cover all qualifying two-hour targets in D005; gaps are omitted because
they do not pass the event-level dynamic-change gate. This is not a sequence
of every wet timestamp or an animation of all 240 event hours.

For target T, the model observes depth and signed HU/HV at **T−5 h** and
**T−5.5 h**, along with the original static and block features. Forcing is
known through T, as in the existing model evaluation. Reference target depths
are read for comparison only; intermediate reference hydraulic states are
not model inputs. Event-relative hours are used because the NetCDF calendar
epoch is a compatibility label rather than a verified event date.

The frozen plan and evidence hashes are in
`results/stage2_whole_domain_halo/plans/D005_fixed5h_flood.json`.

## Orientation and rendering

The original NetCDF x coordinates increase west to east and y coordinates
**decrease from north to south**. They were numerically checked against the
model tile-raster affine transform, in **EPSG:26916 (NAD83 / UTM zone 16N)**.
Mosaics retain native row order and use **origin=upper** with projected extents.
No transpose or vertical flip is applied. Tile center padding is removed
before insertion into each original raster window. The reconstructed reference
is checked against the native NetCDF frame over every valid model cell.

The 34 original HUC12 boundaries are reprojected into the same CRS and drawn
in brown. North arrows, coordinate axes and a 10 km scale bar support review.
An orientation reference was visually checked against the supplied map:
[North-up reference preview](../../results/stage2_whole_domain_halo/orientation_audit/D005_reference_north_up.png).
It places the upstream branches at the top, their convergence through the
center, and the downstream channel at the bottom, matching the supplied map.

Each target figure contains **reference depth, Halo depth and their displayed
difference**. Truth and prediction share fixed depth bins from 0.05 to 8 m,
with over-range colors; the difference uses fixed ±2 m colors. Depth below
0.05 m is hidden equally in both panels. Halo depth is not wet-head gated.
To keep narrow channels visible at whole-domain size, displayed depth is the
maximum over each 4 × 4 native-cell footprint. Displayed error subtracts
these maxima; numerical metrics and exported arrays retain native resolution.

## Outputs and validation

Output root:
`results/stage2_whole_domain_halo/20261021_D005_fixed5h_flood/`

- `depth_maps.nc`: native truth, predicted depth and wet probability, with
  coordinates, origin/target times, CRS and lead metadata.
- `metrics.csv`: native all-valid-cell and true-wet-cell depth RMSE per target.
- `plots/target_*h_lead_05.0h.png`: 28 comparison figures.
- `plots/whole_domain_comparison.pdf`: all comparisons in one multipage PDF.
- `README.md`: browsable image links, settings and interpretation notes.
- `MANIFEST.json`: checkpoint/config/source hashes, orientation audit and status.

Scripts: `stage2/whole_domain_halo.py`, `stage2/plot_whole_domain_halo.py`, and
`workflows/stage2/33_whole_domain_halo.sh`. Full-frame hydraulic caching avoids
repeated NetCDF chunk decompression but preserves the original tile/halo
adapters and training normalization. Cached signed components are retained.
The job requests one L40S GPU, 16 CPUs, 375104 MiB RAM and up to 24 h, and
runs plotting after successful prediction. It prints frame and tile progress.

Four regression checks passed for axis/transform agreement, north/south error
rejection, tile unpadding/placement, frame caching and channel-preserving
reduction. A synthetic NetCDF-to-PNG/PDF render also passed. The GPU resource
request passed Slurm test-only validation before submission. Source-grid
orientation was checked with a real D005 reference map.
