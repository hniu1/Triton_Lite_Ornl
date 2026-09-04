#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_pair_merge
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-paired-merge-%j.out
#SBATCH -e ./slurm_output/stage1-paired-merge-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

output_root=${OUTPUT_ROOT:-processed_data/timestamp_stage1/m6_paired_transition_index}

python -u data_preprocessing/m6_merge_paired_transition_index.py \
  --shards-dir "${output_root}/shards" \
  --output-dir "${output_root}" \
  --expected-events 40 --overwrite
