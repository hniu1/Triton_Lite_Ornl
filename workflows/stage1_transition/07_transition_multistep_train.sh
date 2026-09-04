#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multistep_v1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v1-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v1-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_multistep_train.py \
  --initial-run-dir results/stage1_transition_v1_continued \
  --output-dir results/stage1_transition_multistep_v1 \
  --rollout-steps 6 --epochs 3 --batch-size 8 \
  --train-batches-per-epoch 500 --eval-batches 200 --eval-time-stride 6 \
  --learning-rate 5e-6 --predicted-state-probability-start 0.25 \
  --predicted-state-probability-end 0.75 --num-workers 2 \
  --device cuda --disable-cudnn

