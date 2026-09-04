#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multi_v3_domain
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v3-domain-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v3-domain-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u stage1_transition_whole_area.py \
  --run-dir results/stage1_transition_multistep_v3 \
  --output-dir results/stage1_transition_multistep_v3/D030_whole_area_key_times \
  --checkpoint best_model.pt --event-id D030 \
  --time-indices 60 140 240 360 440 \
  --wet-probability-threshold 0.5 --batch-size 64 --num-workers 2 --device cuda
