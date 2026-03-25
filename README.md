# Triton Lite Block-wise Surrogate

This repository now centers on a block-wise flood-depth surrogate for Triton Lite.
The active workflow is:

- build event tensors from hydrologic inputs
- build static block metadata/features
- build block-wise labels from depth netCDF outputs
- train one shared model that predicts flood depth for one block at a time

The older watershed-level training, tuning, and prediction scripts have been removed.

## Active Files

```text
├── data_preprocessing/
│   ├── m0_generate_netcdf_from_zip.py
│   ├── m1a_build_event_sources.py
│   ├── m1b_event_to_tensor.py
│   ├── m2_block_feature_extraction.py
│   ├── m3_construct_labels_from_netcdf.py
│   ├── README_m1_events.md
│   ├── README_m2_blocks.md
│   └── README_m3_labels_from_netcdf.md
├── blockwise_data.py                       # joins events/blocks/labels and builds datasets
├── blockwise_model.py                      # temporal encoder + block encoder + predictor
├── tune_blockwise.py                       # tunes block-wise hyperparameters
├── train_blockwise.py                      # trains the block-wise depth model
├── predict_blockwise.py                    # predicts block-wise depth from a trained model
├── model.py                               # retained legacy watershed model definition
├── data_loader.py                         # retained legacy watershed data loader
└── README.md
```

## Current Training Target

Each supervised sample is one `(event_id, watershed_id, block_id)` row:

- `X_event`: event time series from Milestone 1, stored as `X_event.npy` with shape `T x F`
- `X_block`: static block features from Milestone 2
- `y`: peak flood depth for that block from Milestone 3

The model architecture matches the current design direction:

- temporal encoder over the hydrologic event time series
- block encoder over static block metadata
- late fusion
- scalar flood-depth prediction for one block

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

### Milestone 3: Block-wise Labels

Use:

- `m0_generate_netcdf_from_zip.py`
- `m3_construct_labels_from_netcdf.py`

Output:

- `labels.parquet`

Reference docs:

- [README_m3_labels_from_netcdf.md](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/data_preprocessing/README_m3_labels_from_netcdf.md)

## Training

Train the current model with [train_blockwise.py](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/train_blockwise.py):

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python train_blockwise.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --output-dir results_blockwise \
  --test-events D040
```

Important behavior:

- splitting is done by event, not by individual event-block rows
- normalization is fit on the training split only
- outputs include a model checkpoint, split tables, normalization stats, and metrics

### Training Outputs

The training output directory contains:

- `best_model.pt`
- `metrics.json`
- `run_config.json`
- `normalization_stats.npz`
- `splits/train_samples.csv`
- `splits/val_samples.csv`
- `splits/test_samples.csv`

You can also train from a tuned config:

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python train_blockwise.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --output-dir results_blockwise_train \
  --config-json results_blockwise_tuning/best_config.json
```

## Tuning

Tune the model with [tune_blockwise.py](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/tune_blockwise.py):

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python tune_blockwise.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --output-dir results_blockwise_tuning \
  --test-events D040
```

Tuning outputs:

- `best_config.json`
- `trials.json`
- `summary.json`

The saved `best_config.json` can be passed directly to `train_blockwise.py`.

## Prediction

Run inference with [predict_blockwise.py](/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/predict_blockwise.py):

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python predict_blockwise.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --checkpoint results_blockwise_train/best_model.pt \
  --normalization-stats results_blockwise_train/normalization_stats.npz \
  --output-parquet results_blockwise_predictions/preds.parquet
```

Optional evaluation against known labels:

```bash
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python predict_blockwise.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --checkpoint results_blockwise_train/best_model.pt \
  --normalization-stats results_blockwise_train/normalization_stats.npz \
  --labels-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --output-parquet results_blockwise_predictions/preds.parquet
```

Prediction outputs:

- `preds.parquet` with `event_id`, `watershed_id`, `block_id`, and `y_pred`
- optional `preds.metrics.json` when labels are supplied

## Status

Supported now:

- block-wise preprocessing through Milestones 1, 2, and 3
- block-wise training with `train_blockwise.py`
- block-wise hyperparameter tuning with `tune_blockwise.py`
- block-wise prediction with `predict_blockwise.py`

The repository now keeps only the active block-wise preprocessing path.
