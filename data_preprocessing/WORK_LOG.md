# Triton-Lite Block-Wise Surrogate: Work Log

## Last Update Date

- 2026-03-24

## Scope completed so far

- Implemented preprocessing-focused milestones (no model code):
  - Milestone 1: event-level dataset preparation for `(Event_i, Block_j)` framework.
  - Milestone 2: block-level feature extraction to `blocks.parquet`.

---

## Milestone 1 (Event-level preprocessing)

### Milestone 1 description

- Goal: build standardized event-level tensors for the block-wise dataset, where each event has a consistent time axis and feature dimension.
- Preferred workflow is a two-step pipeline:
  1. `m1a_build_event_sources.py`: index raw event sources into a canonical table.
  2. `m1b_event_to_tensor.py`: parse event hydrographs, optionally filter sensors, align time, and write tensor outputs.
- Legacy single-script option `m1_event_preprocessing.py` is retained for backward compatibility, but `m1a` + `m1b` is the recommended path.

### Milestone 1 input files

- Required inputs (Step A):
  - Raw event directory containing source files (tested with `.zip` events):
    - `/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr`
- Required inputs (Step B):
  - `processed_data/blockwise_global/milestone_01_events_test/events_source.csv`
- Optional inputs (Step B):
  - `processed_data/hyg/loc_watershed.csv` for watershed sensor filtering

### Milestone 1 output files

- Step A outputs:
  - `processed_data/blockwise_global/milestone_01_events_test/events_source.csv`
  - optional materialized per-event CSV files under:
    - `processed_data/blockwise_global/milestone_01_events_test/materialized_events/`
- Step B trim-mode outputs:
  - `processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered/events.csv`
  - per-event tensors under:
    - `processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered/events/conasauga/<event_id>/X_event.npy`
- Step B preferred 30-minute outputs:
  - `processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered_30min/events.csv`
  - per-event tensors under:
    - `processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered_30min/events/conasauga/<event_id>/X_event.npy`

### Validation runs performed

- Ran end-to-end from raw zip source directory:
  - `/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr`
- Produced outputs in:
  - `processed_data/blockwise_global/milestone_01_events_test/events_source.csv`
  - `processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered/events.csv`
  - `processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered_30min/events.csv`
- Result summary:
  - 40 events processed (`D001`–`D040`)
  - trim-mode tensors: `T=81`, `F=300`
  - preferred 30-minute tensors: `T=480`, `F=300`
  - 30-minute `D001` output matches legacy preprocessing within float32 interpolation precision

---

## Milestone 2 (Block-level feature extraction)

### Implementation

- Added `m2_block_feature_extraction.py`.
- Inputs:
  - block polygon shapefile(s)
  - DEM raster
  - outlet points (vector or CSV)
- Computes per-block features:
  - centroid_x, centroid_y
  - area
  - mean_elevation
  - elevation_range
  - mean_slope
  - distance_to_outlet
- Outputs unified `blocks.parquet`:
  - `watershed_id`, `block_id`, plus feature columns.

### CRS and consistency handling

- Enforces projected DEM CRS.
- Reprojects blocks and outlets to DEM CRS before geometry/raster operations.
- Supports CRS override for missing block/outlet CRS metadata.
- Ensures same feature definitions across watersheds.

### Validation runs performed

- Tested with:
  - `shapefiles/blocks_conasauga.shp`
  - `shapefiles/DEM/D001_dem.tif`
  - test outlet CSV under `processed_data/blockwise_global/milestone_02_blocks_test/outlets_test.csv`
- Output:
  - `processed_data/blockwise_global/milestone_02_blocks_test/blocks.parquet`
- Result summary:
  - 6900 block rows
  - expected schema confirmed.

### Environment note

- Installed `pyarrow` in `/ccs/home/haoranniu/miniconda3/envs/triton` to enable parquet write support.

---

## Key decisions made

- Keep preprocessing independent from old multi-step pipelines where possible.
- Reuse old outputs only as optional source artifacts, not as mandatory pipeline dependencies.
- Prefer simple, modular scripts with explicit intermediate tables:
  - `events_source.csv` (Step A)
  - final `events.csv` (Step B)
  - `blocks.parquet` (Milestone 2)

---

## Open items / next steps

- Milestone 1:
  - sensor filtering is supported via `--sensor-map-default-watershed-id` when the sensor map uses subwatershed names.
  - preferred output is the 30-minute Step B tensor set for downstream block-wise modeling.
- Milestone 2:
  - define canonical outlet points for each watershed.
  - run multi-watershed production extraction with finalized inputs.
- Milestone 3+ (not started):
  - labels table and sample index table for `(event_id, block_id) -> y`.

---

## Runbook (copy-paste commands)

### Environment

```bash
conda activate triton
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
```

### Milestone 1A: raw zip -> events_source.csv (+ optional materialized CSV)

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/m1a_build_event_sources.py \
  --input-dir /lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr \
  --file-pattern 'D*.zip' \
  --default-watershed-id conasauga \
  --materialize-csv-dir processed_data/blockwise_global/milestone_01_events_test/materialized_events \
  --output-events-source-csv processed_data/blockwise_global/milestone_01_events_test/events_source.csv \
  --log-level INFO
```

### Milestone 1B: preferred 30-minute output

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/m1b_event_to_tensor.py \
  --events-source-csv processed_data/blockwise_global/milestone_01_events_test/events_source.csv \
  --base-dir . \
  --sensor-map-csv processed_data/hyg/loc_watershed.csv \
  --sensor-map-watershed-column Name \
  --sensor-map-default-watershed-id conasauga \
  --time-align-mode resample \
  --resample-interval-hours 0.5 \
  --resample-drop-first-row \
  --output-dir processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered_30min \
  --log-level INFO
```

### Milestone 1B: trim-mode alternative

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/m1b_event_to_tensor.py \
  --events-source-csv processed_data/blockwise_global/milestone_01_events_test/events_source.csv \
  --base-dir . \
  --sensor-map-csv processed_data/hyg/loc_watershed.csv \
  --sensor-map-watershed-column Name \
  --sensor-map-default-watershed-id conasauga \
  --time-align-mode trim \
  --output-dir processed_data/blockwise_global/milestone_01_events_test/final_events_sensor_filtered \
  --log-level INFO
```

### Milestone 2: block features -> blocks.parquet

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python \
  data_preprocessing/m2_block_feature_extraction.py \
  --blocks-files shapefiles/blocks_conasauga.shp \
  --blocks-crs EPSG:26916 \
  --dem-raster shapefiles/DEM/D001_dem.tif \
  --outlets-file processed_data/blockwise_global/milestone_02_blocks_test/outlets_test.csv \
  --watershed-id-column watershed_id \
  --outlets-crs EPSG:26916 \
  --output-parquet processed_data/blockwise_global/milestone_02_blocks_test/blocks.parquet \
  --log-level INFO
```
