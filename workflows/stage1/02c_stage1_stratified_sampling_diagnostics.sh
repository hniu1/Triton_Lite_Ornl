#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_strat_diag
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage1-stratified-diagnostics-%j.out
#SBATCH -e ./slurm_output/stage1-stratified-diagnostics-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output results/stage1_timestamp/sampling_diagnostics

BATCHES=${BATCHES:-300}
SAMPLING_INDEX_DIR=${SAMPLING_INDEX_DIR:-processed_data/timestamp_stage1/m4_sampling_index}

python -u stage1_sampling_diagnostics.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --blocks-parquet processed_data/timestamp_stage1/m2_blocks/blocks.parquet \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --sampling-index-dir "${SAMPLING_INDEX_DIR}" \
  --output-path results/stage1_timestamp/sampling_diagnostics/label_aware_sampler.json \
  --test-events D030 \
  --batch-size 16 \
  --batches "${BATCHES}"
