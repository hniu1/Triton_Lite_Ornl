#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_D030_maps
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage1-D030-whole-area-%j.out
#SBATCH -e ./slurm_output/stage1-D030-whole-area-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

OUTPUT_DIR=${OUTPUT_DIR:-results/stage1_timestamp_max/D030_whole_area_interval20_gated}
TIME_INDICES=${TIME_INDICES:-}
mkdir -p slurm_output "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${TIME_INDICES}" ]]; then
  read -r -a SELECTED_TIMES <<< "${TIME_INDICES}"
  EXTRA_ARGS+=(--time-indices "${SELECTED_TIMES[@]}")
fi

python -u stage1_whole_area_inference.py \
  --run-dir results/stage1_timestamp_max \
  --output-dir "${OUTPUT_DIR}" \
  --event-id D030 \
  --checkpoint best_model.pt \
  --wet-calibration results/stage1_timestamp_max/wet_threshold_calibration.json \
  --time-interval 20 \
  --batch-size 64 \
  --num-workers 2 \
  --device cuda \
  "${EXTRA_ARGS[@]}"
