# Milestone 2: Block-level feature extraction

Script:

- `m2_block_feature_extraction.py`

## Goal

Milestone 2 builds the static block feature table used by the block-wise surrogate.
Each row is one block in the full watershed block inventory, and each column is a
fixed block attribute.

This is the `X_block` side of the final training sample:

- Milestone 1: event tensor (`X_event`)
- Milestone 2: block metadata/features (`X_block`)
- Milestone 3: block target (`y`)

The purpose of this step is to support one model across watersheds.
Instead of predicting an entire watershed grid at once, the model predicts one
block at a time from:

- event information
- block information

## Script inputs

Required inputs:

- Block polygons: `--blocks-files`
- DEM raster: `--dem-raster`
- Watershed outlet points: `--outlets-file`
- Output parquet path: `--output-parquet`

Optional ID/CRS controls:

- `--watershed-id-column`
- `--block-id-column`
- `--blocks-crs`
- `--outlets-crs`
- `--outlet-x-column`
- `--outlet-y-column`

## Features computed per block

The script currently computes:

- `centroid_x`
- `centroid_y`
- `area`
- `mean_elevation`
- `elevation_range`
- `mean_slope`
- `distance_to_outlet`

Feature meaning:

- `centroid_x`, `centroid_y`: projected centroid coordinates
- `area`: polygon area in the DEM CRS units
- `mean_elevation`: mean DEM value inside the block
- `elevation_range`: max minus min DEM value inside the block
- `mean_slope`: mean local slope derived from the DEM
- `distance_to_outlet`: centroid distance to the watershed outlet geometry

These features are intentionally simple, static, and comparable across watersheds.

## Output

The script writes `blocks.parquet` with one row per block.

Required output columns:

- `watershed_id`
- `block_id`
- `centroid_x`
- `centroid_y`
- `area`
- `mean_elevation`
- `elevation_range`
- `mean_slope`
- `distance_to_outlet`

This table is the static lookup table for block-wise training and inference.

## ID behavior

`watershed_id`:

- If the block file already contains the watershed column named by `--watershed-id-column`, that column is used.
- Otherwise the script infers `watershed_id` from the block filename.

`block_id`:

- If the block file already contains the block column named by `--block-id-column`, that column is used.
- Otherwise the script generates canonical IDs in the form `watershed_b000001`, `watershed_b000002`, ...
- Generated block IDs are assigned across the merged full watershed inventory, not restarted independently per source file.
- The script raises an error if block IDs are duplicated or if a watershed mixes explicit and generated block IDs.

This matters because Milestone 3 labels must use the same `block_id` convention.

## CRS rules

- DEM CRS must be projected.
- Blocks and outlets are reprojected to DEM CRS before feature calculation.
- If blocks or outlets are missing CRS metadata, pass `--blocks-crs` and/or `--outlets-crs`.

The DEM CRS is the reference CRS for geometry-based features such as area and distance.

## Outlet input format

If `--outlets-file` is a CSV, it must contain:

- `watershed_id`
- `X`
- `Y`

or equivalent column names provided through:

- `--outlet-x-column`
- `--outlet-y-column`

If `--outlets-file` is a vector file, it must contain the watershed ID column.

## Example

```bash
python3 data_preprocessing/m2_block_feature_extraction.py   --blocks-files shapefiles/blocks_conasauga.shp   --blocks-crs EPSG:26916   --dem-raster shapefiles/DEM/D001_dem.tif   --outlets-file processed_data/blockwise_global/milestone_02_blocks_test/outlets_test.csv   --watershed-id-column watershed_id   --outlets-crs EPSG:26916   --output-parquet processed_data/blockwise_global/milestone_02_blocks_test/blocks.parquet
```

## Role in the final block-wise model

Milestone 2 is what makes the block-wise architecture possible.
The final model can combine:

- event embedding from Milestone 1
- block features from Milestone 2

and predict the response for one block.

For the full watershed inventory path, the intended final training join is:

- `events.csv` from Milestone 1
- `blocks.parquet` from Milestone 2
- `block_index_lookup.parquet` and `labels_10m_manifest.parquet` from Milestone 3

joined through:

- `event_id`
- `watershed_id`
- `block_id`

That is the mechanism that removes the need for a fixed number of blocks and
makes a shared cross-watershed model feasible.
