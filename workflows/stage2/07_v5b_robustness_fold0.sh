#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_robust
#SBATCH -p gpu
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-v5b-robust-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v5b-robust-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

variants=(baseline baseline baseline halo halo halo halo_gated_velocity halo_gated_velocity halo_gated_velocity)
seeds=(45 145 245 45 145 245 45 145 245)
variant="${variants[${SLURM_ARRAY_TASK_ID}]}"
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
variant_args=()
if [[ "${variant}" == halo* ]]; then variant_args+=(--hydraulic-halo-cells 8); fi
if [[ "${variant}" == halo_gated_velocity ]]; then
  variant_args+=(--use-component-correction-gate --component-gate-initial-bias -3.0)
fi
mapfile -t test_events < <(python -c 'import json; print(*json.load(open("processed_data/stage2/event_folds.json"))["folds"][0], sep="\n")')

python -u -m stage2.train_direct \
  --initial-run-dir results/stage1_transition_multistep_v4b \
  --output-dir "results/stage2_v5b_robustness_fold0/${variant}/seed_${seed}" \
  --test-events "${test_events[@]}" --data-split-seed 4300 --seed "${seed}" \
  --no-v4b-initialization "${variant_args[@]}" \
  --max-horizon-steps 48 --step-hours 0.5 \
  --epochs 10 --batch-size 8 --train-batches-per-epoch 750 \
  --selection-batches 80 --selection-horizons 1 6 12 24 48 \
  --selection-start-time-stride 12 \
  --backbone-learning-rate 1e-4 --new-conditioning-learning-rate 1e-4 \
  --selection-derived-velocity-weight 1.0 --save-every-epoch \
  --num-workers 2 --device cuda --disable-cudnn

