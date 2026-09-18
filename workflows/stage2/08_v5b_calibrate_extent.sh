#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_calib
#SBATCH -p gpu
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o ./slurm_output/stage2-v5b-calib-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v5b-calib-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

variants=(baseline baseline baseline halo halo halo halo_gated_velocity halo_gated_velocity halo_gated_velocity)
seeds=(45 145 245 45 145 245 45 145 245)
run_dir="results/stage2_v5b_robustness_fold0/${variants[${SLURM_ARRAY_TASK_ID}]}/seed_${seeds[${SLURM_ARRAY_TASK_ID}]}"
python -u -m stage2.calibrate_extent \
  --run-dir "${run_dir}" --horizons 1 6 12 24 48 \
  --evaluation-batches 80 --batch-size 16 --start-time-stride 12 \
  --device cuda --disable-cudnn

