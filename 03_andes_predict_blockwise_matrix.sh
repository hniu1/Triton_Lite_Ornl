#!/bin/bash
#SBATCH -A cli138
#SBATCH -J pred_bw_10m
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/predict-bw-10m-%j.out
#SBATCH -e ./slurm_output/predict-bw-10m-%j.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

module load cuda/11.0.2
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

mkdir -p slurm_output

PYTHON_BIN=/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python

CHECKPOINT_PATH=${CHECKPOINT_PATH:-results_blockwise_matrix_train/best_model.pt}
NORMALIZATION_PATH=${NORMALIZATION_PATH:-results_blockwise_matrix_train/normalization_stats.npz}
OUTPUT_DIR=${OUTPUT_DIR:-results_blockwise_matrix_predictions}
EVENT_IDS=${EVENT_IDS:-D040}
DEVICE=${DEVICE:-auto}

echo "[$(date)] Starting matrix inference"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Normalization: ${NORMALIZATION_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Events: ${EVENT_IDS}"
echo "Device: ${DEVICE}"

${PYTHON_BIN} -u predict_blockwise_matrix.py \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --labels-10m-dir processed_data/blockwise_global/milestone_03_labels_10m \
  --checkpoint "${CHECKPOINT_PATH}" \
  --normalization-stats "${NORMALIZATION_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --event-ids ${EVENT_IDS} \
  --evaluate \
  --device "${DEVICE}"

echo "[$(date)] Matrix inference completed"
