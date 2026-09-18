#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_gate_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-v5b-gate-smoke-%j.out
#SBATCH -e ./slurm_output/stage2-v5b-gate-smoke-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

mapfile -t test_events < <(python -c 'import json; print(*json.load(open("processed_data/stage2/event_folds.json"))["folds"][0], sep="\n")')
python -u -m stage2.train_direct \
  --initial-run-dir results/stage1_transition_multistep_v4b \
  --output-dir results/stage2_v5b_gated_smoke \
  --test-events "${test_events[@]}" --data-split-seed 4300 \
  --no-v4b-initialization --hydraulic-halo-cells 8 \
  --use-component-correction-gate --component-gate-initial-bias -3.0 \
  --epochs 1 --batch-size 2 --train-batches-per-epoch 4 \
  --selection-batches 2 --selection-horizons 1 6 12 24 48 \
  --num-workers 0 --device cuda --disable-cudnn
