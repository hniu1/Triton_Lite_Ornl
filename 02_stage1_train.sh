#!/bin/bash
#SBATCH -A cli138
#SBATCH -J stage1_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage1-train-%j.out
#SBATCH -e ./slurm_output/stage1-train-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

TEST_EVENT=${TEST_EVENT:-D030}
OUTPUT_DIR=${OUTPUT_DIR:-results/stage1_timestamp_stratified}
SAMPLING_INDEX_DIR=${SAMPLING_INDEX_DIR:-processed_data/timestamp_stage1/m4_sampling_index}
BATCH_SIZE=${BATCH_SIZE:-16}
TRAIN_BATCHES=${TRAIN_BATCHES:-1000}
EVAL_BATCHES=${EVAL_BATCHES:-300}

python -u stage1_train.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --sampling-index-dir "${SAMPLING_INDEX_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --test-events "${TEST_EVENT}" \
  --batch-size "${BATCH_SIZE}" \
  --train-batches-per-epoch "${TRAIN_BATCHES}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-time-stride 12 \
  --epochs 50 \
  --checkpoint-metric physical_score \
  --num-workers 0 \
  --device cuda \
  --disable-cudnn
