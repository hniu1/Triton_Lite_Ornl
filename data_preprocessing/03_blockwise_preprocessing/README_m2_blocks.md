# Milestone 2: Block-level feature extraction

Script:

- `m2_block_feature_extraction.py`

## Inputs

- Block polygons (one or multiple files): `--blocks-files`
- DEM raster: `--dem-raster`
- Watershed outlet points: `--outlets-file`

## Features per block

- `centroid_x`, `centroid_y` (projected coordinates)
- `area`
- `mean_elevation`
- `elevation_range`
- `mean_slope`
- `distance_to_outlet`

Output always uses the same feature definitions across all watersheds.

## CRS rules

- DEM CRS must be projected.
- Blocks and outlets are reprojected to DEM CRS before feature calculation.
- If blocks/outlets source has missing CRS, pass `--blocks-crs` and/or `--outlets-crs`.

## Output

- `blocks.parquet` (path provided by `--output-parquet`)
- Columns:
  - `watershed_id`
  - `block_id`
  - feature columns listed above

## Example

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m2_block_feature_extraction.py \
  --blocks-files shapefiles/blocks_conasauga.shp \
  --blocks-crs EPSG:26916 \
  --dem-raster shapefiles/DEM/D001_dem.tif \
  --outlets-file my_outlets.csv \
  --watershed-id-column watershed_id \
  --outlets-crs EPSG:26916 \
  --output-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet
```

If outlet input is CSV, it must contain:

- `watershed_id`
- `X`
- `Y`

or custom names via `--outlet-x-column` / `--outlet-y-column`.
