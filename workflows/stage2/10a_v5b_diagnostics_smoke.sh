#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_diag_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-v5b-diag-smoke-%j.out
#SBATCH -e ./slurm_output/stage2-v5b-diag-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

run_dir=results/stage2_v5b_robustness_fold0/halo/seed_45
python -u -m stage2.calibrate_extent \
  --run-dir "${run_dir}" \
  --output-path "${run_dir}/extent_strategy_calibration_smoke.json" \
  --horizons 1 48 --evaluation-batches 2 --batch-size 2 \
  --start-time-stride 24 --device cuda --disable-cudnn
python -u -m stage2.evaluate_direct \
  --run-dir "${run_dir}" \
  --output-path "${run_dir}/test_diagnostic_smoke.json" \
  --split test --horizons 1 48 --evaluation-batches 2 --batch-size 2 \
  --start-time-stride 24 \
  --extent-calibration "${run_dir}/extent_strategy_calibration_smoke.json" \
  --component-policy velocity_persistence --per-event \
  --device cuda --disable-cudnn
