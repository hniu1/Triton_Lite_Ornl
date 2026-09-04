#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_trans_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-transition-smoke-%j.out
#SBATCH -e ./slurm_output/stage1-transition-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_train.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --sampling-index-dir processed_data/timestamp_stage1/m4_sampling_index_dense \
  --initial-checkpoint results/stage1_timestamp_max/best_model.pt \
  --output-dir results/stage1_transition_v1_smoke --test-events D030 \
  --lag 1 --batch-size 8 --train-batches-per-epoch 10 --eval-batches 10 \
  --eval-time-stride 12 --epochs 1 --early-stop-patience 1 \
  --learning-rate 3e-5 --num-workers 0 --device cuda --disable-cudnn

