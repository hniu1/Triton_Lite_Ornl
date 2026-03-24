# Milestone 1: Event-level preprocessing

Scripts:

- `m1a_build_event_sources.py` (raw input indexing/materialization)
- `m1b_event_to_tensor.py` (final event tensors and events table)
- `m1_event_preprocessing.py` (legacy single-script path, still available)

## Two-step simple pipeline

### Step A (`m1a_build_event_sources.py`)

- Reads raw event sources (zip/txt/hyg/csv).
- Builds a canonical event source table (`events_source.csv`):
  - `event_id`
  - `watershed_id`
  - `hyg_path`
  - `zip_member_path` (optional)
- Optional: materialize raw events into cleaned CSV files.

### Step B (`m1b_event_to_tensor.py`)

- Reads events from `events_source.csv`.
- Selects sensors by watershed (if `--sensor-map-csv` is provided).
- Supports assigning all sensor-map rows to one parent watershed via `--sensor-map-default-watershed-id`.
- Enforces a consistent `T` across all events (`trim` or `resample`).
- Supports legacy-compatible 30-minute interpolation via `--time-align-mode resample --resample-interval-hours 0.5 --resample-drop-first-row`.
- Saves each event as `X_event.npy` with shape `T x F`.
- Writes `events.csv` with columns:
  - `event_id`
  - `watershed_id`
  - `path_to_X_event`
  - `T`
  - `F`

No normalization is applied in either step.

## Step A inputs

- `--input-dir` + `--file-pattern` + `--default-watershed-id`, or
- `--event-manifest-csv` with columns:
  - `event_id`
  - `watershed_id`
  - `hyg_path`
  - optional `zip_member_path`

Optional for Step A:

- `--materialize-csv-dir` to write cleaned per-event CSVs.
- `--zip-member-template` for predictable member location inside zip files.

## Step B inputs

- `--events-source-csv` (from Step A).
- Optional sensor map (`Loc` + watershed column).
- Optional `--zip-member-template` if `hyg_path` points to zip archives.

### Optional sensor map (Step B)

- Provide `--sensor-map-csv` with columns:
  - `Loc`
  - watershed column (default name: `watershed_id`, configurable via `--sensor-map-watershed-column`)
- If the sensor map uses subwatershed names but Milestone 1 events use a single parent watershed ID, pass `--sensor-map-default-watershed-id` to assign all sensor rows to that parent watershed.

If no sensor map is provided, all detected `Loc*` columns are used in each event.

## Example commands

### Step A: raw zip/txt to event source table

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m1a_build_event_sources.py \
  --input-dir /path/to/raw_hyg \
  --file-pattern "D*.zip" \
  --default-watershed-id conasauga \
  --materialize-csv-dir processed_data/blockwise_global/m1a_materialized \
  --output-events-source-csv processed_data/blockwise_global/m1a_events_source.csv
```

### Step B: event source table to final tensors (preferred output)

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m1b_event_to_tensor.py \
  --events-source-csv processed_data/blockwise_global/m1a_events_source.csv \
  --base-dir . \
  --sensor-map-csv processed_data/hyg/loc_watershed.csv \
  --sensor-map-watershed-column Name \
  --sensor-map-default-watershed-id conasauga \
  --time-align-mode resample \
  --resample-interval-hours 0.5 \
  --resample-drop-first-row \
  --output-dir processed_data/blockwise_global/milestone_01_events_30min
```

### Step B: trim-mode alternative

```bash
python3 data_preprocessing/03_blockwise_preprocessing/m1b_event_to_tensor.py \
  --events-source-csv processed_data/blockwise_global/m1a_events_source.csv \
  --base-dir . \
  --time-align-mode trim \
  --output-dir processed_data/blockwise_global/milestone_01_events_trim
```

## Output layout

`--output-dir`:

- `events.csv`
- `events/<watershed_id>/<event_id>/X_event.npy`

## Logging

Step A logs source/materialization records.

Step B logs per-event shapes before and after alignment and prints paths to saved `.npy` files and `events.csv`. For legacy-compatible 30-minute preprocessing, the preferred output is the resampled Step B directory rather than the Step A materialized CSVs.
