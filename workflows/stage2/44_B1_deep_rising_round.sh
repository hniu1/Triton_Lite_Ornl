#!/bin/bash
# One user-requested experimental round; no nested Slurm submissions.
#SBATCH -A cli138
#SBATCH -J s2_B1_deep_round
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-B1-deep-round-%j.out
#SBATCH -e ./slurm_output/stage2-B1-deep-round-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
root=results/stage2_fixed5h_deep_rising/20261022_fold0
baseline=results/stage2_fixed5h_local/20261022_fold0/seed_45
test ! -e "${root}/seed_45"
test ! -e "${root}/evaluation/B_validation.json"
mkdir -p "${root}/evaluation"
# Establish baseline validation diagnostics before any experimental training.
python -u -m stage2.evaluate_direct --run-dir "$baseline" --split val --horizons 10 \
  --batch-size 8 --evaluation-batches 256 --start-time-stride 12 --per-event \
  --wet-classification-mode depth --depth-diagnostics --device cuda --disable-cudnn \
  --output-path "${root}/evaluation/B_validation.json"
# Smoke and fresh training run sequentially within this single allocation.
bash workflows/stage2/43_B1_deep_rising_train.sh smoke
bash workflows/stage2/43_B1_deep_rising_train.sh train "${root}/smoke_job_${SLURM_JOB_ID}"
python -u -m stage2.evaluate_direct --run-dir "${root}/seed_45" --split val --horizons 10 \
  --batch-size 8 --evaluation-batches 256 --start-time-stride 12 --per-event \
  --wet-classification-mode depth --depth-diagnostics --device cuda --disable-cudnn \
  --output-path "${root}/evaluation/B1_validation.json"
python -u -m stage2.report_deep_rising --root "$root"
