#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_v4a_regimes
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-transition-v4a-regimes-%j.out
#SBATCH -e ./slurm_output/stage1-transition-v4a-regimes-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_regime_eval.py \
  --run-dir results/stage1_transition_multistep_v4a \
  --checkpoint best_model.pt \
  --output-path results/stage1_transition_multistep_v4a/regime_metrics.json \
  --batches 200 --batch-size 16 --time-stride 3 \
  --device cuda --disable-cudnn
