#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_v4_diag
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ./slurm_output/stage1-v4-diagnostics-%j.out
#SBATCH -e ./slurm_output/stage1-v4-diagnostics-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output results/stage1_timestamp_v4_finetune/sampling_diagnostics

python -u stage1_sampling_diagnostics.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --sampling-index-dir processed_data/timestamp_stage1/m4_sampling_index_dense \
  --sampling-mode balanced_batch \
  --sampling-strict-category-quotas \
  --sampling-target-wet-cell-fraction 0.15 \
  --sample-dry-fraction 0.125 \
  --sample-boundary-fraction 0.25 \
  --sample-wet-fraction 0.3125 \
  --sample-deep-fraction 0.3125 \
  --sample-quiet-fraction 0.20 \
  --sample-rising-fraction 0.25 \
  --sample-peak-fraction 0.25 \
  --sample-recession-fraction 0.30 \
  --output-path results/stage1_timestamp_v4_finetune/sampling_diagnostics/strict_sampler.json \
  --test-events D030 --batch-size 16 --batches 500 \
  --netcdf-chunk-cache-mb 32 --max-open-netcdf-handles 8

