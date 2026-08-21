# Milestone 3: 10m Dynamic Labels from netCDF

Scripts:

- `m0_generate_netcdf_from_zip.py`
- `m3_prepare_10m_label_assets.py`
- `m3_build_dynamic_manifest.py`

## Goal

Milestone 3 connects dynamic Triton netCDF outputs to the block-wise Stage 1
training dataset.

For the timestamp-conditioned model, M3 does not materialize one `.npy` label
file per timestamp.  The M0 netCDF files already contain the full dynamic fields:

```text
output_depth[t, y, x]
output_velocity_x[t, y, x]
output_velocity_y[t, y, x]
```

The active Stage 1 M3 path writes a lightweight dynamic manifest plus the static
10 m grid-to-block lookup needed to crop block-local labels during training.

## Outputs

### Active timestamp Stage 1 outputs

Under `processed_data/timestamp_stage1/`, the active Stage 1 layout is:

```text
m3_dynamic_manifest/
  dynamic_manifest.parquet
  dynamic_metadata.json
  rejected_events.json

m3_labels_10m/
  block_index_10m.npy
  block_index_lookup.parquet
  labels_10m_metadata.json
```

`m3_dynamic_manifest/` answers: which event netCDF contains the dynamic labels?

`dynamic_manifest.parquet` records columns such as:

- `event_id`
- `watershed_id`
- `path_to_netcdf`
- `n_times`
- `rows`
- `cols`
- `time_start`
- `time_end`
- `time_step`
- `path_to_X_event`
- `forcing_T`
- `forcing_F`

`m3_labels_10m/` answers: which 10 m grid cells belong to which block?

`block_index_10m.npy` stores `int32` indices aligned with the netCDF grid:

- `-1` means no block
- `0..N-1` map to rows in `block_index_lookup.parquet`

Together, these folders let the training dataset read:

```text
(event_id, timestamp, block_id) -> depth patch + velocity-x patch + velocity-y patch
```

### Legacy peak-label outputs

`m3_prepare_10m_label_assets.py` is still useful for the older peak-response
model and for creating the shared block-index assets.  Under `--output-dir`, it
writes:

- `events_peak_10m/D###_peak_10m.npy`
- `events_peak_10m/D###_peak_velx_10m.npy` (when `--include-velocity`)
- `events_peak_10m/D###_peak_vely_10m.npy` (when `--include-velocity`)
- `events_peak_10m/D###_peak_velmag_10m.npy` (when `--include-velocity-magnitude`)
- `block_index_10m.npy`
- `block_index_lookup.parquet`
- `labels_10m_manifest.parquet`
- `labels_10m_metadata.json`

`labels_10m_manifest.parquet` columns:

- `event_id`
- `watershed_id`
- `path_to_peak_10m`
- `path_to_peak_velx_10m` (optional)
- `path_to_peak_vely_10m` (optional)
- `path_to_peak_velmag_10m` (optional)
- `rows`
- `cols`

## Upstream netCDF assumptions

- one netCDF per event
- event token in filename, such as `D001`
- depth variable defaults to `output_depth` unless overridden
- x/y coordinates represent the same grid used for all events

When `m0_generate_netcdf_from_zip.py` is run with `--output-types H U V`, each
event netCDF includes:

- `output_depth`
- `output_velocity_x`
- `output_velocity_y`

This keeps the same netCDF format as before and adds velocity fields for V3 multi-task labels.

## m0: raw zip -> netCDF (depth + velocity)

Example single-event smoke test:

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python \
  data_preprocessing/m0_generate_netcdf_from_zip.py \
  --zip-dir /lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr \
  --zip-pattern D001.zip \
  --output-dir processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3_test \
  --output-types H U V \
  --overwrite \
  --log-level INFO
```

Example full run with scheduler script:

```bash
sbatch data_preprocessing/m0_andes_m0_depth_velocity.sh
```

Notes:

- `--output-types H` remains supported for depth-only legacy behavior.
- Temporary ZIP extraction now uses system temp space so random event temp folders
  are not created under your output directory.

## Example full run

### Active Stage 1 dynamic manifest

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python \
  data_preprocessing/m3_build_dynamic_manifest.py \
  --netcdf-dir processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3 \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --watershed-id conasauga \
  --component-semantics velocity \
  --skip-incomplete \
  --output-dir processed_data/timestamp_stage1/m3_dynamic_manifest
```

The equivalent scheduler script is:

```bash
sbatch workflows/stage1/01_stage1_build_manifest.sh
```

The current validated output contains 40 events (`D001` through `D040`), 480
timestamps per event, forcing tensors of shape `480 x 300`, and no rejected
events.

### Legacy peak-label assets

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python \
  data_preprocessing/m3_prepare_10m_label_assets.py \
  --netcdf-dir processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3 \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --depth-var output_depth \
  --include-velocity \
  --include-velocity-magnitude \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --output-dir processed_data_depth_velocity/blockwise_global/milestone_03_labels_10m_v3 \
  --log-level INFO
```

## Smoke test

Use reduced subsets before long runs:

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python \
  data_preprocessing/m3_prepare_10m_label_assets.py \
  --netcdf-dir processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3 \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --include-velocity \
  --max-events 2 \
  --max-blocks 50 \
  --output-dir processed_data_depth_velocity/blockwise_global/milestone_03_labels_10m_v3_test \
  --log-level INFO
```

## Notes

- The legacy low-resolution scalar `labels.parquet` generator has been removed.
- For Stage 1 timestamp training, use `m3_build_dynamic_manifest.py` plus the
  block-index assets under `processed_data/timestamp_stage1/m3_labels_10m`.
- For the older peak-depth model, use `m3_prepare_10m_label_assets.py`.
