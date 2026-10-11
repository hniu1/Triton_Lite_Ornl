#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_corrected_merge
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-corrected-merge-%j.out
#SBATCH -e ./slurm_output/stage2-corrected-merge-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
data_root=processed_data/stage2_corrected_training/20261009_fold0
python -u -m data_preprocessing.m4_merge_stage1_sampling_index \
  --shards-dir "${data_root}/sampling/shards" --output-dir "${data_root}/sampling" \
  --expected-events 25
python -u -m stage2.verify_corrected_training --data-root "${data_root}"
