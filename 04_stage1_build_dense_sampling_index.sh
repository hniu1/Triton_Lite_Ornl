#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_dense_idx
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --array=1-40%8
#SBATCH -o ./slurm_output/stage1-dense-index-%A_%a.out
#SBATCH -e ./slurm_output/stage1-dense-index-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

EVENT_ID=$(printf 'D%03d' "${SLURM_ARRAY_TASK_ID}")
CANDIDATES_PER_EVENT=${CANDIDATES_PER_EVENT:-50000}
OUTPUT_ROOT=${OUTPUT_ROOT:-processed_data/timestamp_stage1/m4_sampling_index_dense}

python -u data_preprocessing/m4_build_stage1_sampling_index.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --output-dir "${OUTPUT_ROOT}/shards/${EVENT_ID}" \
  --event-ids "${EVENT_ID}" \
  --candidates-per-event "${CANDIDATES_PER_EVENT}" \
  --batch-size 16 \
  --wet-threshold 0.05 \
  --boundary-max-fraction 0.10 \
  --deep-threshold 1.0 \
  --deep-min-wet-fraction 0.10 \
  --deep-depth-statistic p90 \
  --flow-weight-fraction 0.75 \
  --netcdf-chunk-cache-mb 512 \
  --overwrite
