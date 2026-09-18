#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_cv_calib
#SBATCH -p gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o ./slurm_output/stage2-v5b-cv-calib-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v5b-cv-calib-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

fold="${SLURM_ARRAY_TASK_ID}"
if [[ "${fold}" == 0 ]]; then
  run_dir=results/stage2_v5b_robustness_fold0/halo/seed_45
else
  run_dir="results/stage2_v5b_crossval/fold_${fold}/seed_45"
fi
python -u -m stage2.calibrate_extent \
  --run-dir "${run_dir}" --output-path "${run_dir}/extent_strategy_calibration.json" \
  --horizons 1 6 12 24 48 --evaluation-batches 80 \
  --batch-size 16 --start-time-stride 12 --device cuda --disable-cudnn

