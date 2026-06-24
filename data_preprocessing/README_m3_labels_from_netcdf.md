# Milestone 3: 10m Label Assets from netCDF

Scripts:

- `m0_generate_netcdf_from_zip.py`
- `m3_prepare_10m_label_assets.py`

## Goal

Milestone 3 now prepares 10m-aligned label assets for spatial block-wise modeling.
Instead of collapsing each block to one scalar label, this path preserves the full
10m event peak depth field and a companion block-index raster.

## Outputs

Under `--output-dir`, the script writes:

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

`block_index_10m.npy` stores `int32` indices aligned with each event peak grid:

- `-1` means no block
- `0..N-1` map to rows in `block_index_lookup.parquet`

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
- This Milestone 3 path is the active source for 10m label assets.
