# Block-wise Training and Testing Pipeline Note

This note describes the next execution steps for the current Triton Lite
block-wise 10m surrogate workflow.

## Goal

Train one shared model that predicts a block-local 10m depth field from:

- `X_event` from Milestone 1
- `X_block` from Milestone 2
- `Y_10m` and block mask from Milestone 3

Each supervised sample is one `(event_id, watershed_id, block_id)` row.

## Required Inputs

Before starting, we need these three artifacts:

- `events.csv` from `data_preprocessing/m1b_event_to_tensor.py`
- `blocks.parquet` from `data_preprocessing/m2_block_feature_extraction.py`
- 10m label assets from `data_preprocessing/m3_prepare_10m_label_assets.py`

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
- `events.csv` and `labels_10m_manifest.parquet` agree on event coverage
- `blocks.parquet` and `block_index_lookup.parquet` agree on block coverage
- selected block feature columns are numeric and non-null
- block masks and target patches are aligned and padded consistently to `80 x 80`

Primary purpose:

- catch Milestone 1/2/3 key mismatches before spending compute

## Stage 2: Freeze the Split Strategy

We should define the held-out test set before training.

Rules:

- split by event, not by row
- never let rows from the same event appear in both train and test
- keep the test events fixed across all runs

Recommended setup:

- choose a fixed `--test-events` list for final evaluation
- use the remaining events for train/validation only

If we want to measure cross-watershed generalization explicitly, we should also
run at least one experiment with unseen watershed coverage in the test set.

## Stage 3: Matrix Training

Run `train_blockwise_matrix.py` with the fixed test split.

Inputs:

- Milestone 1/2/3 data
- fixed test event list
- target patch size of `80 x 80`

Expected outputs:

- `best_model.pt`
- `normalization_stats.npz`
- `metrics.json`
- `run_config.json`
- split CSVs

What to review:

- train vs validation masked loss
- RMSE and MAE on valid cells
- signs of overfitting
- whether one watershed dominates the data distribution

## Stage 4: Held-out Testing

Evaluate the saved matrix model on the held-out test events using the training script outputs.

What to review:

- overall RMSE / MAE / R2 on valid cells
- per-event behavior
- per-watershed behavior
- spatial underprediction or overprediction patterns inside blocks

## Recommended Directory Layout

Use separate directories for each step:

- `results_blockwise_matrix_train/`

This keeps matrix training artifacts isolated.

## Recommended First Real Run

Suggested order:

1. validate Milestone 1/2/3 coverage and shapes
2. define the fixed `--test-events`
3. train one matrix model on a small subset
4. train one full matrix model
5. summarize errors and decide whether to revise features, split design, or decoder size

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
- whether event encoder or decoder capacity should be increased or reduced
- whether the split should stress unseen watersheds more directly
- whether we need weighted losses or target transforms for skewed depth values
- whether prediction outputs should be expanded into a downstream export product

