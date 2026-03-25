# Milestone 3: Block-wise labels from netCDF

Scripts:

- `m0_generate_netcdf_from_zip.py`
- `m3_construct_labels_from_netcdf.py`

## Goal

Milestone 3 builds the block-level target table used by the block-wise surrogate.
Each row represents one `(event_id, watershed_id, block_id)` pair, and the target
`y` is the peak flood depth observed inside that block over the event cube.

This is the `y` side of the final training sample:

- Milestone 1: event tensor (`X_event`)
- Milestone 2: block metadata/features (`X_block`)
- Milestone 3: block target (`y`)

The intended path is:

1. Generate event netCDF files directly from raw Triton Lite ZIP outputs.
2. Intersect each event depth cube with the watershed block inventory.
3. Write `labels.parquet` with one row per event-block pair.

This path avoids dependence on older raster/block-TIFF label products.

## Upstream netCDF generation

Use `m0_generate_netcdf_from_zip.py` to create event netCDF files.

Required inputs:

- Raw simulation ZIP directory: `--zip-dir`
- Output directory for netCDF files: `--output-dir`

Typical output files:

- `D001_ACC_future.nc`
- `D002_ACC_future.nc`
- ...

Example:

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/03_blockwise_preprocessing/m0_generate_netcdf_from_zip.py \
  --zip-dir /lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr \
  --zip-pattern 'D*.zip' \
  --output-dir processed_data/blockwise_global/milestone_03_netcdf \
  --log-level INFO
```

The downstream `m3` script expects:

- one netCDF per event
- filenames containing an event token like `D001`
- a depth variable named `output_depth` unless overridden
- projected x/y coordinates compatible with the block polygons

## Script inputs

Use `m3_construct_labels_from_netcdf.py` to build `labels.parquet`.

Required inputs:

- NetCDF directory: `--netcdf-dir`
- Block polygon file: `--blocks-file`
- Watershed ID for the block inventory: `--watershed-id`
- Output parquet path: `--output-parquet`

Optional data controls:

- `--netcdf-pattern`
- `--depth-var`
- `--max-events`
- `--max-blocks`

Optional CRS and ID controls:

- `--nc-crs`
- `--blocks-crs`
- `--block-id-column`
- `--block-id-mode`

Optional validation controls:

- `--events-csv`
- `--blocks-parquet`

Optional filtering:

- `--drop-nonhydro-blocks`
- `--hydro-threshold`

## Output

The script writes `labels.parquet` with columns:

- `event_id`
- `watershed_id`
- `block_id`
- `y`

`y` is computed as the maximum finite value of `output_depth` across:

- all timesteps in the event cube
- all raster cells that fall inside the block polygon

The script sorts the final table by:

- `watershed_id`
- `event_id`
- `block_id`

## Label construction behavior

For each event netCDF:

1. Parse `event_id` from the filename using the `D###` token.
2. Read the depth cube from `--depth-var` (default `output_depth`).
3. Reproject block geometries to the netCDF CRS.
4. Build one raster mask per block against the netCDF grid.
5. Compute block `y` as the peak depth over all timesteps and covered cells.

The script enforces complete event-block coverage for the produced set. For each
watershed, it checks that:

`row_count == number_of_events * number_of_blocks`

The script raises an error if:

- no netCDF files match `--netcdf-pattern`
- the depth variable is missing
- the blocks file has no CRS and `--blocks-crs` is not supplied
- no usable block masks are generated
- any label value is `NaN`
- any event-block combinations are missing from the produced table

## Block ID behavior

`watershed_id`:

- The CLI value from `--watershed-id` is normalized to lowercase underscore form.
- Example: `Conasauga Basin` becomes `conasauga_basin`.

`block_id`:

- If `--block-id-column` is provided and exists in the block file, that column is used.
- Otherwise IDs are generated according to `--block-id-mode`.

Supported `--block-id-mode` values:

- `watershed_b_padded`: `conasauga_b000001`, `conasauga_b000002`, ...
- `watershed_block`: `conasauga_block_0`, `conasauga_block_1`, ...
- `index`: `0`, `1`, ...

This must match the `block_id` convention used by Milestone 2 if you validate
against `blocks.parquet` or plan to join the tables later.

## CRS rules

- The script infers the raster transform from the first netCDF file in `--netcdf-dir`.
- Blocks are reprojected to `--nc-crs` before raster masking.
- If the blocks file is missing CRS metadata, provide `--blocks-crs`.
- The default netCDF CRS is `EPSG:26916`.

In practice, `--nc-crs` should match the CRS used when the netCDFs were generated.

## Validation behavior

`--events-csv`:

- Requires columns `event_id` and `watershed_id`.
- Validates that the watershed-event pairs in `labels.parquet` match the events table.

`--blocks-parquet`:

- Requires columns `watershed_id` and `block_id`.
- Validates that the watershed-block pairs in `labels.parquet` match the block table.

These validation steps check key coverage, not numeric equivalence to another label source.

## Example

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/03_blockwise_preprocessing/m3_construct_labels_from_netcdf.py \
  --netcdf-dir processed_data/blockwise_global/milestone_03_netcdf \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --events-csv processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered_30min/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks_test/blocks.parquet \
  --output-parquet processed_data/blockwise_global/milestone_03_labels_netcdf/labels.parquet \
  --log-level INFO
```

## Quick test

Use `--max-events` and `--max-blocks` to smoke-test the pipeline on a small subset:

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/03_blockwise_preprocessing/m3_construct_labels_from_netcdf.py \
  --netcdf-dir processed_data/blockwise_global/milestone_03_netcdf \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --events-csv processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered_30min/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks_test/blocks.parquet \
  --max-events 2 \
  --max-blocks 50 \
  --output-parquet processed_data/blockwise_global/milestone_03_labels_netcdf_test/labels.parquet
```

## Role in the final block-wise model

For the full watershed inventory path, the intended final training join is:

- `events.csv` from Milestone 1
- `blocks.parquet` from Milestone 2
- `labels.parquet` from Milestone 3

joined through:

- `event_id`
- `watershed_id`
- `block_id`

That join is what allows a shared model to consume event features, block features,
and a per-block target without requiring a fixed watershed raster layout.
