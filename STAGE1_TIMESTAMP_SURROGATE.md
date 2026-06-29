# Stage 1: Timestamp-Conditioned TRITON Surrogate

![Stage-1 timestamp-conditioned surrogate architecture](docs/images/stage1_timestamp_surrogate_architecture.png)

## Prediction contract

For a requested output timestamp `t`, the model predicts one block-local state:

```text
forcing history through t + timestamp + block attributes + static 10m rasters
    -> depth(t), component_x(t), component_y(t), wet_probability(t)
```

The temporal encoder is an eight-layer causal dilated TCN with a 511-step
receptive field. Although the event tensor is loaded as a `480 x 300` array for
efficient batching, values after `t` cannot influence the embedding selected
at `t`. Per-timestamp LayerNorm is used instead of temporal BatchNorm to
preserve this property during training as well as inference.

The signed component fields are deliberately named `component_x/y`. The TRITON
paper states that native U/V output is unit discharge, while the existing
netCDF converter labels it velocity. Set and document `--component-semantics`
only after auditing the binary convention.

## Files

- `data_preprocessing/m3_build_dynamic_manifest.py`: validates netCDF trajectories.
- `stage1_data.py`: event splits, normalization, streaming block/time reads.
- `stage1_model.py`: causal TCN plus conditioned spatial U-Net.
- `stage1_train.py`: losses, metrics, checkpointing, held-out testing.
- `stage1_predict.py`: selected event/timestamp/block inference.
- `01_stage1_build_manifest.sh`: manifest scheduler job.
- `02_stage1_train.sh`: training scheduler job.

## Why labels are streamed

The dynamic netCDF fields contain hundreds of gigabytes per event when
uncompressed. Expanding them into `(event,time,block,channel,80,80)` samples
would create a multi-terabyte duplicate. The dataset therefore reads requested
patches directly from netCDF.

The files use large HDF5 chunks. Each minibatch shares one event and timestamp
and contains spatially consecutive blocks, allowing the netCDF chunk cache to
reuse decompressed chunks.

## Build the manifest

```bash
sbatch 01_stage1_build_manifest.sh
```

The command writes:

```text
processed_data_depth_velocity/blockwise_global/milestone_03_dynamic_manifest/
  dynamic_manifest.parquet
  dynamic_metadata.json
  rejected_events.json
```

`--skip-incomplete` records unreadable or incomplete events rather than hiding
them. Repair rejected events before the final scientific training campaign.

## Train

```bash
sbatch 02_stage1_train.sh
```

The default split is by complete event. Training randomly samples event/time
groups and local block batches. Validation and testing use deterministic,
evenly distributed samples across events, timestamps, and blocks.

Important outputs:

```text
results/stage1_timestamp/
  best_model.pt
  normalization_stats.npz
  run_config.json
  metrics.json
```

Metrics include:

- depth RMSE/MAE over all valid cells;
- depth RMSE/MAE over wet cells;
- signed-component RMSE/MAE over wet cells;
- wet precision, recall, F1, and CSI.

## Predict selected states

```bash
python stage1_predict.py \
  --run-dir results/stage1_timestamp \
  --output-dir results/stage1_predictions_D030 \
  --event-id D030 \
  --time-indices 120 240 360 \
  --block-indices 0 1 2 3 \
  --device cuda
```

Each output `.npz` contains depth, wet probability, both signed components, and
the block mask. A prediction manifest records the event, timestamp, and block.

## Loss

The training objective combines:

1. wet-cell depth Huber loss and a lower-weight dry-cell depth penalty;
2. wet/dry BCE;
3. signed-component Huber loss on truly wet cells;
4. a dry-cell penalty that drives flow components toward zero.

Training sampling is also imbalance-aware: timestamps are sampled from a
mixture of uniform and hydrograph-intensity distributions, and spatial batches
are sampled from a mixture of uniform and flow-accumulation-weighted block
distributions. Validation and testing remain deterministic and unweighted.

This is a timestamp emulator, not an autoregressive simulator. Predictions at
different timestamps do not consume earlier predicted hydraulic states.

## Before a production run

1. Repair or regenerate rejected netCDF events.
2. Resolve whether U/V are velocity or unit discharge.
3. Select held-out normal and extreme events before tuning.
4. Benchmark netCDF throughput and tune batch size/chunk-cache size.
5. Calibrate the wet-probability threshold on validation events only.
