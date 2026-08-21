#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_fair_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-fair-eval-%j.out
#SBATCH -e ./slurm_output/stage1-fair-eval-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

CURRENT=results/stage1_timestamp_max
PREVIOUS=results/stage1_timestamp_stratified
PREDICTIONS=${CURRENT}/predictions_D030_representative
PLOTS=${CURRENT}/plots
mkdir -p "${PREDICTIONS}" "${PLOTS}"

python -u stage1_evaluate.py \
  --run-dir "${PREVIOUS}" \
  --checkpoint best_model.pt \
  --output-path "${CURRENT}/previous_stratified_fair_evaluation.json" \
  --device cuda --eval-batches 1000 --eval-time-stride 6 --num-workers 2

python -u stage1_predict.py \
  --run-dir "${CURRENT}" --checkpoint best_model.pt \
  --output-dir "${PREDICTIONS}" --event-id D030 --time-indices 240 \
  --block-indices 1109 1874 2028 2560 --device cuda --num-workers 0

python -u plot/plot_stage1_prediction_samples.py \
  --prediction-dir "${PREDICTIONS}" --output-dir "${PLOTS}/prediction_samples"

python -u plot/plot_stage1_evaluation_comparison.py \
  --previous "${CURRENT}/previous_stratified_fair_evaluation.json" \
  --current "${CURRENT}/evaluation_metrics.json" --output-dir "${PLOTS}"
