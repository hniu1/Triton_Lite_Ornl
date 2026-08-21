#!/bin/bash
#SBATCH -A cli138
#SBATCH -J stage1_v4_ft
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o ./slurm_output/stage1-v4-finetune-%j.out
#SBATCH -e ./slurm_output/stage1-v4-finetune-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_train.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --sampling-index-dir processed_data/timestamp_stage1/m4_sampling_index_dense \
  --sampling-mode balanced_batch --sampling-strict-category-quotas \
  --sampling-target-wet-cell-fraction 0.15 \
  --sample-dry-fraction 0.125 --sample-boundary-fraction 0.25 \
  --sample-wet-fraction 0.3125 --sample-deep-fraction 0.3125 \
  --sample-quiet-fraction 0.20 --sample-rising-fraction 0.25 \
  --sample-peak-fraction 0.25 --sample-recession-fraction 0.30 \
  --initial-checkpoint results/stage1_timestamp_max/best_model.pt \
  --output-dir results/stage1_timestamp_v4_finetune --test-events D030 \
  --batch-size 16 --train-batches-per-epoch 2000 --eval-batches 1000 \
  --eval-time-stride 6 --epochs 5 --early-stop-patience 3 \
  --learning-rate 3e-5 --num-workers 2 \
  --depth-loss-mode hybrid_log_weighted --depth-log-huber-delta 0.20 \
  --depth-physical-huber-delta 1.0 --depth-log-loss-weight 1.0 \
  --depth-physical-loss-weight 1.0 --depth-weight-shallow 1.0 \
  --depth-weight-moderate 2.0 --depth-weight-deep 3.0 --depth-weight-extreme 4.0 \
  --couple-depth-with-wet-probability --dry-depth-loss-weight 0.10 \
  --wet-loss-weight 0.20 --wet-dice-loss-weight 0.15 --wet-pos-weight 1.25 \
  --component-loss-mode speed_aware --component-loss-weight 0.50 \
  --speed-loss-weight 0.50 --direction-loss-weight 0.10 \
  --direction-min-speed 0.05 --velocity-weight-scale 2.0 \
  --velocity-weight-reference-speed 0.25 --velocity-weight-cap 3.0 \
  --dry-component-loss-weight 0.05 --checkpoint-metric physical_score \
  --netcdf-chunk-cache-mb 32 --max-open-netcdf-handles 8 \
  --device cuda --disable-cudnn

