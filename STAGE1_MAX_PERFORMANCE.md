# Stage-1 Maximum-Performance Experiment

This experiment addresses the remaining Stage-1 weaknesses after the first
label-aware run: 55% of complete training batches were still dry, only about
5% of training cells were wet, and wet-depth RMSE on D030 remained 1.49 m.

The architecture is intentionally unchanged so the experiment isolates
stronger sampling and supervision.

## 1. Dense dynamic-label index

Run one event per array task:

```bash
sbatch 04_stage1_build_dense_sampling_index.sh
```

The dense index requests 50,000 candidates per event, ten times the previous
index density. It retains a 25% uniform spatial component and assigns 75% of
spatial candidate probability from flow accumulation.

After every array task succeeds:

```bash
sbatch 04b_stage1_merge_dense_sampling_index.sh
```

The merged index is written to:

```text
processed_data/timestamp_stage1/m4_sampling_index_dense/
```

Dynamic categories use actual TRITON depth:

- `dry`: no cells at or above 0.05 m;
- `boundary`: positive wet fraction below 10%, even if one isolated cell is deep;
- `deep`: at least 10% wet and wet-cell 90th-percentile depth at least 1 m;
- `wet`: remaining patches with at least 10% wet cells.

This prevents an otherwise dry patch with one extreme cell from being treated
as a useful deep-water sample.

## 2. Whole-batch balancing

The previous sampler labeled one anchor and loaded 15 uncontrolled neighbors.
The new sampler chooses all 16 unique blocks from the same event and timestamp
using indexed labels. Initial category counts are:

| Category | Blocks per batch |
|---|---:|
| Dry | 2 |
| Boundary | 4 |
| Wet | 6 |
| Deep | 4 |

If that selection does not reach a 15% mean wet-cell fraction, lower-wet
blocks are replaced with available higher-wet blocks from the same event and
timestamp. The event/time group is accepted only if the index proves that the
target is attainable.

Before training, verify the actual distribution:

```bash
sbatch 04c_stage1_dense_sampling_diagnostics.sh
```

The diagnostic must show that the full batches—not only anchors—reach the
desired wet-cell fraction. Its output is:

```text
results/stage1_timestamp_max/sampling_diagnostics/balanced_batch_sampler.json
```

## 3. Hybrid depth loss

For true wet cells, depth supervision combines log-depth and physical-depth
Huber losses:

```text
L_depth = 1.0 * Huber(log1p(pred), log1p(true), delta=0.20)
        + 0.5 * Huber(pred, true, delta=1.0 m)
```

Both terms use true-depth weights:

| True depth | Weight |
|---|---:|
| 0.05–0.25 m | 1 |
| 0.25–1.0 m | 2 |
| 1.0–2.0 m | 3 |
| At least 2.0 m | 4 |

The log term handles the large dynamic range. The weighted physical term keeps
the objective tied to errors in meters and gives deeper-water errors stronger
gradients than the previous delta-0.25 Huber loss.

## 4. Inundation and velocity losses

The full configured objective is:

```text
L = L_depth
  + 0.02 * L_dry_depth
  + 0.20 * L_wet_BCE
  + 0.30 * L_wet_Dice
  + 0.50 * L_wet_velocity
  + 0.02 * L_dry_velocity
```

Wet BCE positive weight is reduced from 3 to 2 because the preceding run
already increased recall while decreasing precision. Dice directly rewards
spatial overlap. Signed x/y velocity values remain unchanged and can be
positive or negative.

The training log and `metrics.json` record every component separately.

## 5. Training volume

After reviewing the dense-sampler diagnostic:

```bash
sbatch 05_stage1_train_max.sh
```

The run uses:

- 5,000 training batches per epoch;
- 16 blocks per batch;
- 20 epochs maximum;
- up to 1.6 million training patches;
- 1,000 validation/test batches at a six-timestamp stride;
- physical-score checkpoint selection;
- a separate output directory: `results/stage1_timestamp_max/`.

The current allocation permits 24 hours on the `extended` partition. The job
script has been validated with `sbatch --test-only` but has not been submitted.

## Decision gate

Do not submit training until all dense-index shards merge successfully and the
balanced-batch diagnostic confirms the wet-cell target without extreme reuse.
The first comparison should use the same D030 split, followed by evaluation on
several additional held-out events.
