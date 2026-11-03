#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_dynamic_cohort
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-dynamic-cohort-%j.out
#SBATCH -e ./slurm_output/stage2-dynamic-cohort-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-dynamic-stream-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-dynamic-stream-cache-${SLURM_JOB_ID}"
python -u -m stage2.dynamic_stream_cohort merge --rule-path results/stage2_stream_flood/20261011_fold0/RULE_available_history.json
