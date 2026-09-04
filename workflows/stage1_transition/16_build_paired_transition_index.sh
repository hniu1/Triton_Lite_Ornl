#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_pair_idx
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --array=1-40%8
#SBATCH -o ./slurm_output/stage1-paired-index-%A_%a.out
#SBATCH -e ./slurm_output/stage1-paired-index-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

event_id=$(printf 'D%03d' "${SLURM_ARRAY_TASK_ID}")
source_root=${SOURCE_ROOT:-processed_data/timestamp_stage1/m4_sampling_index_dense}
output_root=${OUTPUT_ROOT:-processed_data/timestamp_stage1/m6_paired_transition_index}

python -u data_preprocessing/m6_build_paired_transition_index.py \
  --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --input-candidates "${source_root}/shards/${event_id}/sampling_candidates.parquet" \
  --output-dir "${output_root}/shards/${event_id}" \
  --event-ids "${event_id}" \
  --wet-threshold 0.05 \
  --stable-storage-threshold 0.01 --stable-extent-threshold 0.01 \
  --rapid-storage-threshold 0.05 --rapid-extent-threshold 0.05 \
  --netcdf-chunk-cache-mb 512 --overwrite
