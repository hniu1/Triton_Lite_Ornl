#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_max_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ./slurm_output/stage1-max-eval-%j.out
#SBATCH -e ./slurm_output/stage1-max-eval-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

RUN_DIR=results/stage1_timestamp_max
PREDICTION_DIR=${RUN_DIR}/predictions_D030_representative
PLOT_DIR=${RUN_DIR}/plots
mkdir -p slurm_output "${PREDICTION_DIR}" "${PLOT_DIR}"

python -u stage1_evaluate.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint best_model.pt \
  --output-path "${RUN_DIR}/evaluation_metrics.json" \
  --device cuda \
  --eval-batches 1000 \
  --num-workers 2

python -u stage1_predict.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint best_model.pt \
  --output-dir "${PREDICTION_DIR}" \
  --event-id D030 \
  --time-indices 240 \
  --block-indices 1109 1874 2028 2560 \
  --device cuda \
  --num-workers 0

python -u plot/plot_stage1_prediction_samples.py \
  --prediction-dir "${PREDICTION_DIR}" \
  --output-dir "${PLOT_DIR}/prediction_samples"

python -u plot/plot_stage1_current_run.py \
  --log slurm_output/stage1-max-3400623.out \
  --output-dir "${PLOT_DIR}" \
  --baseline-diagnostic results/stage1_timestamp/sampling_diagnostics/proxy_sampler.json \
  --current-diagnostic "${RUN_DIR}/sampling_diagnostics/balanced_batch_sampler.json" \
  --previous-metrics results/stage1_timestamp_stratified/metrics.json \
  --evaluation "${RUN_DIR}/evaluation_metrics.json"
