#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multi_v3_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v3-smoke-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v3-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_multistep_v2_train.py \
  --training-mode paired_local_12step_multistep_v3 \
  --initial-run-dir results/stage1_transition_multistep_v1 \
  --sampling-index-dir processed_data/timestamp_stage1/m6_paired_transition_index \
  --output-dir results/stage1_transition_multistep_v3_smoke_eval \
  --rollout-steps 12 --epochs 1 --batch-size 8 \
  --train-batches-per-epoch 4 --eval-batches 16 --eval-time-stride 6 \
  --learning-rate 2e-6 --predicted-state-probability-start 0.50 \
  --predicted-state-probability-end 0.50 \
  --sample-stable-fraction 0.40 --sample-filling-fraction 0.10 \
  --sample-draining-fraction 0.10 --sample-rapid-filling-fraction 0.20 \
  --sample-rapid-draining-fraction 0.20 \
  --stable-depth-delta-loss-weight 1.5 \
  --rapid-depth-delta-loss-weight 1.0 \
  --component-delta-loss-weight 0.5 \
  --derived-velocity-loss-type mse --derived-velocity-loss-weight 1.0 \
  --storage-change-loss-weight 0.5 \
  --num-workers 0 --device cuda --disable-cudnn
