#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v4b_policy
#SBATCH -p gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o ./slurm_output/stage2-v4b-policy-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v4b-policy-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

# Both array tasks use the same 48-step-eligible origins as the existing V4b
# dense evaluation. Only the component update policy changes.
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    mode=velocity_persistence
    ;;
  1)
    mode=component_persistence
    ;;
  *)
    echo "Unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

python -u archive/stage1/code/stage1_transition_rollout.py \
  --run-dir results/stage1_transition_multistep_v4b \
  --checkpoint best_model.pt \
  --output-path "results/stage1_transition_multistep_v4b/dense_24h_${mode}_metrics.json" \
  --component-update-mode "${mode}" \
  --horizons {1..48} --rollout-batches 100 --batch-size 16 \
  --start-time-stride 12 --step-hours 0.5 --device cuda --disable-cudnn
