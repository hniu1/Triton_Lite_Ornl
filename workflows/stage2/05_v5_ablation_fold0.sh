#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5_ablate
#SBATCH -p gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-v5-ablate-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v5-ablate-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

mapfile -t test_events < <(python -c 'import json; print(*json.load(open("processed_data/stage2/event_folds.json"))["folds"][0], sep="\n")')
variant_args=()
case "${SLURM_ARRAY_TASK_ID}" in
  0) variant=baseline ;;
  1) variant=halo; variant_args+=(--hydraulic-halo-cells 8) ;;
  2) variant=halo_coupled; variant_args+=(--hydraulic-halo-cells 8 --velocity-direction-loss-weight 0.05 --boundary-flux-loss-weight 0.5) ;;
  3) variant=halo_coupled_local; variant_args+=(--hydraulic-halo-cells 8 --velocity-direction-loss-weight 0.05 --boundary-flux-loss-weight 0.5 --forcing-weights-path processed_data/stage2/m7_tile_forcing/tile_forcing_weights.npy --local-forcing-embedding-dim 32) ;;
  *) echo "Unsupported task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

python -u -m stage2.train_direct \
  --initial-run-dir results/stage1_transition_multistep_v4b \
  --output-dir "results/stage2_v5_ablation_fold0/${variant}" \
  --test-events "${test_events[@]}" --data-split-seed 4300 \
  --no-v4b-initialization "${variant_args[@]}" \
  --max-horizon-steps 48 --step-hours 0.5 \
  --epochs 10 --batch-size 8 --train-batches-per-epoch 750 \
  --selection-batches 80 --selection-horizons 1 6 12 24 48 \
  --selection-start-time-stride 12 \
  --backbone-learning-rate 1e-4 --new-conditioning-learning-rate 1e-4 \
  --selection-derived-velocity-weight 1.0 --save-every-epoch \
  --num-workers 2 --device cuda --disable-cudnn

