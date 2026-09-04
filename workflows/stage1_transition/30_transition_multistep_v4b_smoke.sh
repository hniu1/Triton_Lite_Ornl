#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multi_v4b_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v4b-smoke-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v4b-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_multistep_v2_train.py \
  --training-mode two_state_fast_adapter_gate_v4b \
  --initial-run-dir results/stage1_transition_multistep_v1 \
  --sampling-index-dir processed_data/timestamp_stage1/m6_paired_transition_index \
  --output-dir results/stage1_transition_multistep_v4b_smoke \
  --history-states 2 --history-fusion adapter \
  --use-activity-gate --activity-gate-initial-bias -1.5 \
  --rollout-steps 12 --epochs 1 --batch-size 8 \
  --train-batches-per-epoch 8 --eval-batches 16 --eval-time-stride 6 \
  --learning-rate 2e-6 --adaptation-learning-rate 1e-3 \
  --predicted-state-probability-start 0.15 \
  --predicted-state-probability-end 0.15 \
  --sample-stable-fraction 0.30 --sample-filling-fraction 0.10 \
  --sample-draining-fraction 0.10 --sample-rapid-filling-fraction 0.25 \
  --sample-rapid-draining-fraction 0.25 \
  --stable-depth-delta-loss-weight 0.75 \
  --rapid-depth-delta-loss-weight 1.0 \
  --component-delta-loss-weight 0.5 \
  --derived-velocity-loss-type mse --derived-velocity-loss-weight 1.0 \
  --storage-change-loss-weight 0.5 --activity-gate-loss-weight 0.02 \
  --selection-derived-velocity-weight 1.0 --save-every-epoch \
  --num-workers 0 --device cuda --disable-cudnn
