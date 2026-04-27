#!/bin/bash
#SBATCH -A cli138
#SBATCH -J tune_bw_10m
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/tune-bw-10m-%j.out
#SBATCH -e ./slurm_output/tune-bw-10m-%j.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

module load cuda/11.0.2
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate triton

mkdir -p slurm_output

PYTHON_BIN=/ccs/home/haoranniu/miniconda3/envs/triton/bin/python

echo "[$(date)] Starting matrix tuning"

${PYTHON_BIN} -u tune_blockwise_matrix.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-10m-dir processed_data/blockwise_global/milestone_03_labels_10m \
  --output-dir results_blockwise_matrix_tuning \
  --test-events D040 \
  --epochs 8 \
  --early-stop-patience 3 \
  --num-trials 3 \
  --device cuda

echo "[$(date)] Matrix tuning completed"
