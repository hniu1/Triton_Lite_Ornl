#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_dense_merge
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-dense-merge-%j.out
#SBATCH -e ./slurm_output/stage1-dense-merge-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

OUTPUT_ROOT=${OUTPUT_ROOT:-processed_data/timestamp_stage1/m4_sampling_index_dense}

python -u data_preprocessing/m4_merge_stage1_sampling_index.py \
  --shards-dir "${OUTPUT_ROOT}/shards" \
  --output-dir "${OUTPUT_ROOT}" \
  --expected-events 40 \
  --overwrite
