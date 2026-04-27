# Triton Lite Block-wise Surrogate

This repository now centers on a block-wise flood-depth surrogate for Triton Lite.
The active workflow is:

- build event tensors from hydrologic inputs
- build static block metadata/features
- build 10m label assets from depth netCDF outputs
- train one shared model that predicts an 80x80 depth field for one block at a time

The older watershed-level training, tuning, and prediction scripts have been removed.

## Active Files

```text
├── data_preprocessing/
│   ├── m0_generate_netcdf_from_zip.py
│   ├── m1a_build_event_sources.py
│   ├── m1b_event_to_tensor.py
│   ├── m2_block_feature_extraction.py
│   ├── m3_prepare_10m_label_assets.py
│   ├── README_m1_events.md
│   ├── README_m2_blocks.md
│   └── README_m3_labels_from_netcdf.md
├── blockwise_data.py                       # shared split and normalization helpers
├── blockwise_matrix_data.py                # builds 80x80 block-local training samples
├── blockwise_model.py                      # temporal encoder + block encoder + 80x80 decoder
├── train_blockwise_matrix.py               # trains the 10m block-wise depth-field model
├── tune_blockwise_matrix.py                # tunes matrix-model hyperparameters
├── predict_blockwise_matrix.py             # writes matrix predictions from a trained model
├── model.py                               # retained legacy watershed model definition
├── data_loader.py                         # retained legacy watershed data loader
└── README.md
```

## Current Training Target

Each supervised sample is one `(event_id, watershed_id, block_id)` row with a spatial target:

- `X_event`: event time series from Milestone 1, stored as `X_event.npy` with shape `T x F`
- `X_block`: static block features from Milestone 2
- `Y_10m`: an 80x80 peak-depth patch plus a matching block mask from Milestone 3

The model architecture matches the current design direction:

- temporal encoder over the hydrologic event time series
- block encoder over static block metadata
- late fusion
- spatial decoder to an 80x80 depth field for one block

## Requirements

The active code path requires:

- Python with `numpy`, `pandas`, `scikit-learn`, and `torch`
- `pyarrow` or `fastparquet` for parquet I/O
- geospatial dependencies for preprocessing scripts, including `geopandas`, `rasterio`, and `shapely`

Example install:

```bash
pip install numpy pandas scikit-learn torch pyarrow geopandas rasterio shapely
```

## Data Processing

The supported preprocessing flow is under [data_preprocessing](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/data_preprocessing).

### Milestone 1: Event Tensors

Use:

- `m1a_build_event_sources.py`
- `m1b_event_to_tensor.py`

Outputs:

- `events.csv`
- `events/<watershed_id>/<event_id>/X_event.npy`

Reference docs:

- [README_m1_events.md](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/data_preprocessing/README_m1_events.md)

### Milestone 2: Block Features

Use:

- `m2_block_feature_extraction.py`

Output:

- `blocks.parquet`

Reference docs:

- [README_m2_blocks.md](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/data_preprocessing/README_m2_blocks.md)

### Milestone 3: 10m Label Assets

Use:

- `m0_generate_netcdf_from_zip.py`
- `m3_prepare_10m_label_assets.py`

Output:

- `events_peak_10m/*.npy`
- `block_index_10m.npy`
- `block_index_lookup.parquet`
- `labels_10m_manifest.parquet`
- `labels_10m_metadata.json`

Reference docs:

- [README_m3_labels_from_netcdf.md](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/data_preprocessing/README_m3_labels_from_netcdf.md)

## Training

Train the current model with [train_blockwise_matrix.py](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/train_blockwise_matrix.py):

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python train_blockwise_matrix.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-10m-dir processed_data/blockwise_global/milestone_03_labels_10m \
  --output-dir results_blockwise_matrix \
  --test-events D040
```

Important behavior:

- splitting is done by event, not by individual event-block rows
- normalization is fit on the training split only
- outputs include a model checkpoint, split tables, normalization stats, and masked metrics

### Training Outputs

The training output directory contains:

- `best_model.pt`
- `metrics.json`
- `run_config.json`
- `normalization_stats.npz`
- `splits/train_samples.csv`
- `splits/val_samples.csv`
- `splits/test_samples.csv`

Current implementation scope:

- matrix training
- matrix hyperparameter tuning
- matrix inference and evaluation

## Tuning

Tune the matrix model with [tune_blockwise_matrix.py](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/tune_blockwise_matrix.py):

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python tune_blockwise_matrix.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-10m-dir processed_data/blockwise_global/milestone_03_labels_10m \
  --output-dir results_blockwise_matrix_tuning \
  --test-events D040
```

Tuning outputs:

- `best_config.json`
- `trials.json`
- `summary.json`

The saved `best_config.json` can be passed directly to `train_blockwise_matrix.py` with `--config-json`.

## Inference

Run matrix inference with [predict_blockwise_matrix.py](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/predict_blockwise_matrix.py):

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python predict_blockwise_matrix.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-10m-dir processed_data/blockwise_global/milestone_03_labels_10m \
  --checkpoint results_blockwise_matrix_train/best_model.pt \
  --normalization-stats results_blockwise_matrix_train/normalization_stats.npz \
  --output-dir results_blockwise_matrix_predictions \
  --event-ids D040 \
  --evaluate
```

Inference outputs:

- `predictions.npy` with shape `(N, 80, 80)` written as a memmapped `.npy`
- `prediction_manifest.parquet` with `sample_index`, `event_id`, `watershed_id`, `block_id`, and `block_index`
- `summary.json`
- optional `metrics.json` when `--evaluate` is supplied

## Status

Supported now:

- block-wise preprocessing through Milestones 1, 2, and 3
- block-wise 10m matrix training with `train_blockwise_matrix.py`
- block-wise 10m matrix tuning with `tune_blockwise_matrix.py`
- block-wise 10m matrix inference with `predict_blockwise_matrix.py`

The repository now keeps only the active block-wise preprocessing path.
