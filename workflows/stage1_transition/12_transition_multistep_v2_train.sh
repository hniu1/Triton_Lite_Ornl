#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multistep_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v2-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v2-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_multistep_v2_train.py \
  --initial-run-dir results/stage1_transition_multistep_v1 \
  --sampling-index-dir processed_data/timestamp_stage1/m5_transition_sampling_index \
  --output-dir results/stage1_transition_multistep_v2 \
  --rollout-steps 6 --epochs 3 --batch-size 8 \
  --train-batches-per-epoch 500 --eval-batches 200 --eval-time-stride 6 \
  --learning-rate 3e-6 --predicted-state-probability-start 0.35 \
  --predicted-state-probability-end 0.80 --num-workers 2 \
  --device cuda --disable-cudnn
