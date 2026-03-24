# Milestone 3 (Upstream Input): labels from netCDF

Script:

- `m3_construct_labels_from_netcdf.py`

## Why this script

This version uses the same upstream input lineage as
`data_preprocessing/02_depth_processing/01_generate_netcdf.py`:

- netCDF files in `processed_data/netcdf` (generated from raw simulation ZIPs)
- block polygons

It does **not** require `processed_data/Block_tiffs_*` outputs.

## Input

- `--netcdf-dir` (e.g., `processed_data/netcdf`)
- `--blocks-file` (e.g., `shapefiles/blocks_conasauga.shp`)
- `--watershed-id` (e.g., `conasauga`)

## Output

- `labels.parquet` with columns:
  - `event_id`
  - `watershed_id`
  - `block_id`
  - `y` (peak depth)

## Behavior

- For each netCDF event and each block polygon:
  - extracts `output_depth` values in block footprint
  - computes `y = max(depth)`
- Enforces no missing event-block pairs in the produced set.
- Optional `--drop-nonhydro-blocks` keeps only blocks with max `y > threshold` across events.

## Example (full)

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m3_construct_labels_from_netcdf.py \
  --netcdf-dir processed_data/netcdf \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --output-parquet processed_data/blockwise_global/milestone_03_labels_netcdf/labels.parquet \
  --log-level INFO
```

## Example (quick test)

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m3_construct_labels_from_netcdf.py \
  --netcdf-dir processed_data/netcdf \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --max-events 2 \
  --max-blocks 50 \
  --output-parquet processed_data/blockwise_global/milestone_03_labels_netcdf_test/labels.parquet
```
