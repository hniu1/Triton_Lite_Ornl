# Milestone 3: Construct labels table

Script:

- `m3_construct_labels.py`

## Goal

Build `labels.parquet` with columns:

- `event_id`
- `watershed_id`
- `block_id`
- `y` (peak flood depth)

`y` is computed as the max raster value inside each event-block TIFF file.

## Input

- One or more directories with hydraulic block TIFF outputs (`--input-dirs`), e.g.:
  - `processed_data/Block_tiffs_chicken_creek`
  - `processed_data/Block_tiffs_pinhook_creek`
  - `processed_data/Block_tiffs_sugar_creek`

Expected filename pattern (default parser):

- `{watershed}_ACC_{event_id}_{watershed}_block_{index}.tif`

Example:

- `chicken_creek_ACC_D001_chicken_creek_block_10.tif`

## Block ID modes

- `watershed_block` (default): `watershed_block_<index>`
- `index`: `<index>`
- `watershed_b_padded`: `watershed_b000010`

Choose mode that matches your `blocks.parquet` IDs if validating against blocks.

## Validations

By default, script enforces:

- No duplicate `(event_id, watershed_id, block_id)` rows
- No NaN `y`
- No missing event-block Cartesian pairs per watershed (based on hydro blocks present)

Optional cross-table validation:

- `--events-csv` checks watershed/event membership
- `--blocks-parquet` checks watershed/block membership

Use `--allow-missing-pairs` only if sparse pair coverage is expected.

## Example

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m3_construct_labels.py \
  --input-dirs \
    processed_data/Block_tiffs_chicken_creek \
    processed_data/Block_tiffs_pinhook_creek \
    processed_data/Block_tiffs_sugar_creek \
  --block-id-mode watershed_block \
  --output-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --log-level INFO
```
