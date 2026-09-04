#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multi_v2_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v2-smoke-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v2-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_multistep_v2_train.py \
  --initial-run-dir results/stage1_transition_multistep_v1 \
  --sampling-index-dir processed_data/timestamp_stage1/m5_transition_sampling_index \
  --output-dir results/stage1_transition_multistep_v2_smoke \
  --rollout-steps 3 --epochs 1 --batch-size 2 \
  --train-batches-per-epoch 2 --eval-batches 2 --eval-time-stride 12 \
  --learning-rate 3e-6 --predicted-state-probability-start 0.5 \
  --predicted-state-probability-end 0.5 --num-workers 0 \
  --device cuda --disable-cudnn
