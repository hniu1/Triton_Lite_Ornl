#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multi_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-smoke-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_multistep_train.py \
  --initial-run-dir results/stage1_transition_v1_continued \
  --output-dir results/stage1_transition_multistep_v1_smoke \
  --rollout-steps 3 --epochs 1 --batch-size 2 \
  --train-batches-per-epoch 2 --eval-batches 2 --eval-time-stride 12 \
  --learning-rate 5e-6 --predicted-state-probability-start 0.5 \
  --predicted-state-probability-end 0.5 --num-workers 0 \
  --device cuda --disable-cudnn

