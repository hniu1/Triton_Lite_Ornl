#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_wet_calibrate
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ./slurm_output/stage1-wet-calibrate-%j.out
#SBATCH -e ./slurm_output/stage1-wet-calibrate-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_calibrate_wet_threshold.py \
  --run-dir results/stage1_timestamp_max \
  --checkpoint best_model.pt \
  --device cuda \
  --eval-batches 1000 \
  --threshold-min 0.05 \
  --threshold-max 0.95 \
  --threshold-step 0.025 \
  --metric csi
