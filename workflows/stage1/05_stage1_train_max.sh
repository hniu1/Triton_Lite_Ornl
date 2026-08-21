#!/bin/bash
#SBATCH -A cli138
#SBATCH -J stage1_max
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage1-max-%j.out
#SBATCH -e ./slurm_output/stage1-max-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

TEST_EVENT=${TEST_EVENT:-D030}
OUTPUT_DIR=${OUTPUT_DIR:-results/stage1_timestamp_max}
SAMPLING_INDEX_DIR=${SAMPLING_INDEX_DIR:-processed_data/timestamp_stage1/m4_sampling_index_dense}
BATCH_SIZE=${BATCH_SIZE:-16}
TRAIN_BATCHES=${TRAIN_BATCHES:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-1000}

python -u stage1_train.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --sampling-index-dir "${SAMPLING_INDEX_DIR}" \
  --sampling-mode balanced_batch \
  --sampling-target-wet-cell-fraction 0.15 \
  --sample-dry-fraction 0.125 \
  --sample-boundary-fraction 0.25 \
  --sample-wet-fraction 0.375 \
  --sample-deep-fraction 0.25 \
  --sample-quiet-fraction 0.05 \
  --sample-rising-fraction 0.30 \
  --sample-peak-fraction 0.40 \
  --sample-recession-fraction 0.25 \
  --output-dir "${OUTPUT_DIR}" \
  --test-events "${TEST_EVENT}" \
  --batch-size "${BATCH_SIZE}" \
  --train-batches-per-epoch "${TRAIN_BATCHES}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-time-stride 6 \
  --netcdf-chunk-cache-mb 32 \
  --max-open-netcdf-handles 8 \
  --epochs 20 \
  --early-stop-patience 6 \
  --num-workers 2 \
  --depth-loss-mode hybrid_log_weighted \
  --depth-log-huber-delta 0.20 \
  --depth-physical-huber-delta 1.0 \
  --depth-log-loss-weight 1.0 \
  --depth-physical-loss-weight 0.5 \
  --depth-weight-shallow 1.0 \
  --depth-weight-moderate 2.0 \
  --depth-weight-deep 3.0 \
  --depth-weight-extreme 4.0 \
  --dry-depth-loss-weight 0.02 \
  --wet-loss-weight 0.20 \
  --wet-dice-loss-weight 0.30 \
  --wet-pos-weight 2.0 \
  --component-loss-weight 0.50 \
  --dry-component-loss-weight 0.02 \
  --checkpoint-metric physical_score \
  --device cuda \
  --disable-cudnn
