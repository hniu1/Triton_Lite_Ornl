#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_sample_diag
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage1-sampling-diagnostics-%j.out
#SBATCH -e ./slurm_output/stage1-sampling-diagnostics-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output results/stage1_timestamp/sampling_diagnostics

BATCHES=${BATCHES:-300}

python -u stage1_sampling_diagnostics.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --output-path results/stage1_timestamp/sampling_diagnostics/proxy_sampler.json \
  --test-events D030 \
  --batch-size 16 \
  --batches "${BATCHES}"
