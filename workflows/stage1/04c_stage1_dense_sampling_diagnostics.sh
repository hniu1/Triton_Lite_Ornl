#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_dense_diag
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-dense-diagnostics-%j.out
#SBATCH -e ./slurm_output/stage1-dense-diagnostics-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output results/stage1_timestamp_max/sampling_diagnostics

BATCHES=${BATCHES:-500}
SAMPLING_INDEX_DIR=${SAMPLING_INDEX_DIR:-processed_data/timestamp_stage1/m4_sampling_index_dense}

python -u stage1_sampling_diagnostics.py \
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
  --output-path results/stage1_timestamp_max/sampling_diagnostics/balanced_batch_sampler.json \
  --test-events D030 \
  --batch-size 16 \
  --netcdf-chunk-cache-mb 32 \
  --max-open-netcdf-handles 8 \
  --batches "${BATCHES}"
