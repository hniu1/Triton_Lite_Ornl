#!/bin/bash
#SBATCH -A cli138
#SBATCH -J blockwise_tune
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o ./slurm_output/blockwise-tune-%j.out
#SBATCH -e ./slurm_output/blockwise-tune-%j.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

module load cuda/11.0.2
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate triton

mkdir -p slurm_output

PYTHON_BIN=/ccs/home/haoranniu/miniconda3/envs/triton/bin/python

echo "[$(date)] Starting blockwise tuning"
${PYTHON_BIN} -u tune_blockwise.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --output-dir results_blockwise_tuning \
  --test-events D040

echo "[$(date)] Blockwise tuning completed successfully"