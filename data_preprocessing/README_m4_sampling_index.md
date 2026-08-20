# Milestone 4: Label-Aware Stage-1 Sampling Index

## Purpose

M4 measures actual TRITON depth labels before training and creates a compact
candidate pool for stratified sampling. It addresses the fact that high
forcing and high flow accumulation are useful proxies but do not guarantee
that a particular `(event, timestamp, block)` target is wet.

The full Cartesian product has more than 132 million samples and would require
scanning roughly 3.3 TB of uncompressed depth values. M4 therefore indexes a
reproducible candidate pool instead of duplicating or exhaustively scanning
the dynamic labels.

Each indexed anchor records:

- event and timestamp;
- anchor block and local batch start;
- hydrograph phase: `quiet`, `rising`, `peak`, or `recession`;
- label category: `dry`, `boundary`, `wet`, or `deep`;
- wet-cell fraction, maximum depth, and mean wet-cell depth.

Training places the anchor near the center of a same-event, same-time local
block batch. The default category quotas are 15% dry, 25% boundary, 40% wet,
and 20% deep. Phase quotas are 15% quiet, 30% rising, 30% peak, and 25%
recession. If a category/phase is unavailable, probabilities are renormalized
over available candidates.

## Recommended order

First measure the old forcing/flow-proxy sampler:

```bash
sbatch 01b_stage1_sampling_diagnostics.sh
```

The result is written to:

```text
results/stage1_timestamp/sampling_diagnostics/proxy_sampler.json
```

Then build one M4 shard per event:

```bash
sbatch 02_stage1_build_sampling_index.sh
```

After all 40 array tasks finish successfully, merge the shards:

```bash
sbatch 02b_stage1_merge_sampling_index.sh
```

Measure the actual full-batch distribution from the new anchor sampler:

```bash
sbatch 02c_stage1_stratified_sampling_diagnostics.sh
```

Compare `proxy_sampler.json` with `label_aware_sampler.json` before training.

The merged output is:

```text
processed_data/timestamp_stage1/m4_sampling_index/
  sampling_candidates.parquet
  sampling_metadata.json
  sampling_summary.csv
  shards/D###/...
```

After reviewing that comparison, `02_stage1_train.sh` reads this directory through
`--sampling-index-dir` and writes a new run to
`results/stage1_timestamp_stratified/`. The original baseline run remains
unchanged in `results/stage1_timestamp/`.

## Checkpoint selection and diagnostics

The updated trainer logs the actual dry-patch fraction and wet-cell fraction
for every epoch. It saves:

- `best_val_loss_model.pt` for minimum composite validation loss;
- `best_physical_model.pt` for minimum physical score;
- `best_model.pt` according to `--checkpoint-metric`.

The default physical score is minimized and combines wet-depth RMSE,
velocity-component RMSE, and wet/dry F1. Its weights are configurable. This
keeps both the original loss-selected checkpoint and the physically selected
checkpoint available for comparison.

## Important limitation

M4 guarantees the category of the anchor patch, not every neighboring patch
in its local batch. Training logs report the actual full-batch distribution,
which is the quantity to use when deciding whether the quotas need adjustment.

## Dense whole-batch extension

The follow-up workflow uses `04_stage1_build_dense_sampling_index.sh` to build
50,000 candidates per event under `m4_sampling_index_dense/`. It defines deep
water using both wet fraction and wet-cell 90th-percentile depth. The
`balanced_batch` sampler then selects all 16 unique blocks from one event and
timestamp and enforces a minimum mean wet-cell fraction.

See `STAGE1_MAX_PERFORMANCE.md` before running the dense-index and hybrid-loss
experiment.
