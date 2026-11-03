# Event forcing and quiet-period review

Prepared 2026-11-03 following PI/team concerns about quiet early training
periods. Descriptive plots cover every event and location without changing
training, thresholds, data or checkpoints. No Slurm jobs submitted.

Each of 40 events has 300 source forcing hydrographs (Loc1–Loc300), with
480 half-hour samples at 0.5–240 h. These are simulator inflow locations,
not a claim of 300 physical field sensors. The corrected manifest and every
array's shape/finite values were checked. Source-file comments label cfs;
simulator unit conversion remains outside the established source audit.

Outputs: `results/stage2_event_forcing_review/20261103/`.

- `all_events_300_locations.pdf`: 40 event pages. Each has all 300 raw
  curves, an independently scaled first-24-hour zoom and a location/time
  heatmap normalized by each location's event maximum.
- `all_events_overview.pdf`: two pages showing all 40 events, full duration
  and first 24 h; maxima/p95/medians normalized by each event's maximum.
- `events/`: 40 PNGs and 40 location-level descriptive-statistics CSVs.
- `event_summary.csv`, `FINDINGS.md`, `README.md`, `MANIFEST.json`.

Reproduce with `python -m stage2.plot_event_forcing_review` in an environment
with NumPy and Matplotlib. Output directory must not exist. The 2026-11-03
run used isolated plotting dependencies in `/tmp/triton-event-plot-deps`
because the earlier training Python environment is currently unavailable.
All 40 event plots and both PDFs (40 and 2 pages) were verified; example
plots were visually inspected. Provenance includes input hashes and the
independently checked corrected time-axis manifest hash.

Early forcing is generally nonzero and varies substantially across events.
Low source forcing is not equivalent to an absence of hydraulic flooding.
Plotting alone does not demonstrate a training bias.

The sampler already requests 20% quiet, 25% rising, 25% peak and 30%
recession phases, subject to pool availability and batch feasibility. It
also balances dry/boundary/wet/deep tiles. These requests are not measured
realized draw proportions. Fixed-5 h B requires target index >=11 (6 h);
the multi-lead run requires >=25 (13 h). Earlier forcing remains context.

Next investigate actual sampled target/phase/depth/change coverage and
compare local forcing changes to hydraulic depth/extent changes. Define any
new active-window rule on training events, preserving a quiet/dry component
for false-alarm and recession performance. All-split plots here are descriptive;
do not tune training rules on inspected held-out events.
