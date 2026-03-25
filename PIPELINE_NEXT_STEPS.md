# Block-wise Training and Testing Pipeline Note

This note describes the next execution steps for the current Triton Lite
block-wise surrogate workflow. It is intended as the operating plan before we
launch real tuning, training, and testing runs.

## Goal

Train one shared model that predicts block-wise peak flood depth from:

- `X_event` from Milestone 1
- `X_block` from Milestone 2
- `y` from Milestone 3

Each supervised sample is one `(event_id, watershed_id, block_id)` row.

## Required Inputs

Before starting, we need these three artifacts:

- `events.csv` from `data_preprocessing/m1b_event_to_tensor.py`
- `blocks.parquet` from `data_preprocessing/m2_block_feature_extraction.py`
- `labels.parquet` from `data_preprocessing/m3_construct_labels_from_netcdf.py`

These must align on:

- `event_id`
- `watershed_id`
- `block_id`

## Stage 1: Data Contract Check

First step is to validate the joined training inputs before any tuning or
training run.

Checks:

- all `path_to_X_event` files exist
- all event tensors share one common `(T, F)` shape
- `events.csv` and `labels.parquet` agree on event coverage
- `blocks.parquet` and `labels.parquet` agree on block coverage
- selected block feature columns are numeric and non-null
- `labels.parquet` has no missing `y`

Primary purpose:

- catch Milestone 1/2/3 key mismatches before spending compute

## Stage 2: Freeze the Split Strategy

We should define the held-out test set before tuning.

Rules:

- split by event, not by row
- never let rows from the same event appear in both train and test
- keep the test events fixed across all runs

Recommended setup:

- choose a fixed `--test-events` list for final evaluation
- use the remaining events for train/validation only

If we want to measure cross-watershed generalization explicitly, we should also
run at least one experiment with unseen watershed coverage in the test set.

## Stage 3: Hyperparameter Tuning

Run `tune_blockwise.py` using the fixed test split.

Purpose:

- search model size and optimization settings on train/validation only
- produce a reusable `best_config.json`

Recommended first pass:

- modest number of trials
- early stopping enabled
- CPU or GPU depending availability

Outputs to review:

- `best_config.json`
- `trials.json`
- `summary.json`

Questions to answer after tuning:

- are the best trials consistently better, or just noisy
- is the model too small or too large
- is validation error stable across candidate settings

## Stage 4: Final Training

Run `train_blockwise.py` with the tuned config.

Inputs:

- Milestone 1/2/3 data
- fixed test event list
- `best_config.json` from tuning

Expected outputs:

- `best_model.pt`
- `normalization_stats.npz`
- `metrics.json`
- `run_config.json`
- split CSVs

What to review:

- train vs validation loss
- RMSE and MAE on validation
- signs of overfitting
- whether one watershed dominates the data distribution

## Stage 5: Held-out Testing and Prediction

Run `predict_blockwise.py` on the held-out test events.

Use:

- `best_model.pt`
- `normalization_stats.npz`
- `events.csv`
- `blocks.parquet`

Two modes:

- with `labels.parquet` for evaluation
- without `labels.parquet` for pure inference

Outputs:

- prediction parquet with `y_pred`
- optional metrics JSON if labels are provided

What to review:

- overall RMSE / MAE / R2
- per-event behavior
- per-watershed behavior
- bias toward underprediction or overprediction
- behavior on low-depth vs high-depth blocks

## Recommended Directory Layout

Use separate directories for each step:

- `results_blockwise_tuning/`
- `results_blockwise_train/`
- `results_blockwise_predictions/`

This keeps tuning artifacts, final training artifacts, and prediction outputs
from getting mixed together.

## Recommended First Real Run

Suggested order:

1. validate Milestone 1/2/3 coverage and shapes
2. define the fixed `--test-events`
3. run a small tuning sweep
4. train one final model from the tuned config
5. run prediction on the held-out test events
6. summarize errors and decide whether to revise features, split design, or model size

## Likely Early Failure Modes

The most likely blockers are:

- event ID mismatch between Milestone 1 and Milestone 3
- block ID mismatch between Milestone 2 and Milestone 3
- inconsistent event tensor shape across events
- missing event tensor files
- too-large joined dataset for memory if we scale up without staging

## Decision Points After First Run

After the first complete tuning-training-testing pass, we should decide:

- whether the current block feature set is sufficient
- whether event encoder capacity should be increased or reduced
- whether the split should stress unseen watersheds more directly
- whether we need weighted losses or target transforms for skewed depth values
- whether prediction outputs should be expanded into a downstream export product

