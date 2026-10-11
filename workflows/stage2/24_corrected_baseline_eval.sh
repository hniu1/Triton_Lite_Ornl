#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_corrected_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-corrected-eval-%j.out
#SBATCH -e ./slurm_output/stage2-corrected-eval-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
run_dir="${1:-results/stage2_corrected_baseline/20261009_fold0/seed_45}"
python -u -m stage2.evaluate_direct \
  --run-dir "${run_dir}" \
  --split test --evaluation-batches 80 --batch-size 8 --start-time-stride 12 \
  --per-event --component-policy learned --device cuda --disable-cudnn
