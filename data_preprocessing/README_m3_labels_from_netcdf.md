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
- `block_index_10m.npy`
- `block_index_lookup.parquet`
- `labels_10m_manifest.parquet`
- `labels_10m_metadata.json`

`labels_10m_manifest.parquet` columns:

- `event_id`
- `watershed_id`
- `path_to_peak_10m`
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

## Example full run

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/m3_prepare_10m_label_assets.py \
  --netcdf-dir processed_data_v1/netcdf \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --depth-var output_depth \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --output-dir processed_data/blockwise_global/milestone_03_labels_10m \
  --log-level INFO
```

## Smoke test

Use reduced subsets before long runs:

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/m3_prepare_10m_label_assets.py \
  --netcdf-dir processed_data_v1/netcdf \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --max-events 2 \
  --max-blocks 50 \
  --output-dir processed_data/blockwise_global/milestone_03_labels_10m_test \
  --log-level INFO
```

## Notes

- The legacy low-resolution scalar `labels.parquet` generator has been removed.
- This Milestone 3 path is the active source for 10m label assets.
