#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_v4b_dense24h
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o ./slurm_output/stage1-transition-v4b-dense24h-%j.out
#SBATCH -e ./slurm_output/stage1-transition-v4b-dense24h-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

# One model step is 0.5 hour. Evaluate every lead from 0.5 through 24 hours
# over a single cohort that is eligible for the full 48-step rollout.
python -u stage1_transition_rollout.py \
  --run-dir results/stage1_transition_multistep_v4b \
  --checkpoint best_model.pt \
  --output-path results/stage1_transition_multistep_v4b/dense_24h_rollout_metrics.json \
  --horizons {1..48} --rollout-batches 100 --batch-size 16 \
  --start-time-stride 12 --step-hours 0.5 --device cuda --disable-cudnn
