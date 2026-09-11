#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-v5-direct-smoke-%j.out
#SBATCH -e ./slurm_output/stage2-v5-direct-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u -m stage2.train_direct \
  --initial-run-dir results/stage1_transition_multistep_v4b \
  --output-dir results/stage2_v5_direct_smoke \
  --max-horizon-steps 48 --step-hours 0.5 \
  --epochs 1 --batch-size 2 --train-batches-per-epoch 4 \
  --selection-batches 2 --selection-horizons 1 6 12 24 48 \
  --backbone-learning-rate 1e-5 --new-conditioning-learning-rate 1e-4 \
  --num-workers 0 --device cuda --disable-cudnn

