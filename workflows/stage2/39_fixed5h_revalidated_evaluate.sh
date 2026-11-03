#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_B_reaudit_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-B-reaudit-eval-%j.out
#SBATCH -e ./slurm_output/stage2-B-reaudit-eval-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
python -u -m stage2.reaudit_fixed5h_sources verify
bash workflows/stage2/37_fixed5h_local_evaluate.sh
