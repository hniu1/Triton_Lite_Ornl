#!/bin/bash
#SBATCH -A cli138
#SBATCH -J train_bw_10m
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o ./slurm_output/train-bw-10m-%j.out
#SBATCH -e ./slurm_output/train-bw-10m-%j.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

module load cuda/11.0.2
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

mkdir -p slurm_output

PYTHON_BIN=/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python
CONFIG_JSON=${CONFIG_JSON:-results_blockwise_matrix_tuning/best_config.json}
OUTPUT_DIR=${OUTPUT_DIR:-results_blockwise_matrix_train}
RUN_WITH_TUNED_CONFIG=${RUN_WITH_TUNED_CONFIG:-auto}
DEVICE=${DEVICE:-auto}
DEPTH_WEIGHT_ALPHA=${DEPTH_WEIGHT_ALPHA:-0.0}
DEPTH_WEIGHT_CAP=${DEPTH_WEIGHT_CAP:-3.0}
AUX_WET_LOSS_WEIGHT=${AUX_WET_LOSS_WEIGHT:-0.2}
WET_THRESHOLD=${WET_THRESHOLD:-0.05}
STATIC_RASTERS_DIR=${STATIC_RASTERS_DIR:-processed_data/blockwise_global/milestone_02_5_static_rasters_v3}
RASTER_ENC_CHANNELS=${RASTER_ENC_CHANNELS:-16}

echo "[$(date)] Starting matrix training"
echo "Config: ${CONFIG_JSON}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Depth weight alpha: ${DEPTH_WEIGHT_ALPHA}"
echo "Depth weight cap: ${DEPTH_WEIGHT_CAP}"
echo "Aux wet loss weight: ${AUX_WET_LOSS_WEIGHT}"
echo "Wet threshold: ${WET_THRESHOLD}"
echo "Static rasters dir: ${STATIC_RASTERS_DIR}"
echo "Raster enc channels: ${RASTER_ENC_CHANNELS}"

declare -a CONFIG_ARGS
if [[ "${RUN_WITH_TUNED_CONFIG}" == "always" ]]; then
  CONFIG_ARGS=(--config-json "${CONFIG_JSON}")
elif [[ "${RUN_WITH_TUNED_CONFIG}" == "never" ]]; then
  CONFIG_ARGS=()
elif [[ -f "${CONFIG_JSON}" ]]; then
  CONFIG_ARGS=(--config-json "${CONFIG_JSON}")
else
  CONFIG_ARGS=()
fi

if [[ ${#CONFIG_ARGS[@]} -gt 0 ]]; then
  echo "Using tuned config"
else
  echo "No tuned config detected or requested; using trainer defaults"
fi

${PYTHON_BIN} -u train_blockwise_matrix.py \
	--events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
	--blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
	--labels-10m-dir processed_data/blockwise_global/milestone_03_labels_10m \
	--output-dir "${OUTPUT_DIR}" \
	"${CONFIG_ARGS[@]}" \
	--depth-weight-alpha "${DEPTH_WEIGHT_ALPHA}" \
	--depth-weight-cap "${DEPTH_WEIGHT_CAP}" \
	--aux-wet-loss-weight "${AUX_WET_LOSS_WEIGHT}" \
	--wet-threshold "${WET_THRESHOLD}" \
	--static-rasters-dir "${STATIC_RASTERS_DIR}" \
	--raster-enc-channels "${RASTER_ENC_CHANNELS}" \
	--test-events D040 \
	--device "${DEVICE}"

echo "[$(date)] Matrix training completed"

