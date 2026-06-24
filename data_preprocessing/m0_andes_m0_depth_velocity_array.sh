#!/bin/bash
#SBATCH -A cli138
#SBATCH -J m0_dv_arr
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=1-40%4
#SBATCH -o ./slurm_output/m0-depth-velocity-arr-%A_%a.out
#SBATCH -e ./slurm_output/m0-depth-velocity-arr-%A_%a.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

mkdir -p slurm_output processed_data_depth_velocity/logs processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3

PYTHON_BIN=/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python
ZIP_DIR=${ZIP_DIR:-/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr}
OUTPUT_DIR=${OUTPUT_DIR:-processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3}
TEMP_ROOT_DIR=${TEMP_ROOT_DIR:-processed_data_depth_velocity/tmp_unzip}

EVENT_ID=$(printf "D%03d" "${SLURM_ARRAY_TASK_ID}")
ZIP_PATTERN="${EVENT_ID}.zip"
LOG_FILE="processed_data_depth_velocity/logs/m0_${EVENT_ID}.log"
TASK_TEMP_DIR="${TEMP_ROOT_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${TASK_TEMP_DIR}"

echo "[$(date)] Starting m0 for ${EVENT_ID}"
echo "ZIP_DIR=${ZIP_DIR}"
echo "ZIP_PATTERN=${ZIP_PATTERN}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "TASK_TEMP_DIR=${TASK_TEMP_DIR}"

${PYTHON_BIN} -u data_preprocessing/m0_generate_netcdf_from_zip.py \
  --zip-dir "${ZIP_DIR}" \
  --zip-pattern "${ZIP_PATTERN}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-types H U V \
  --compression-level 1 \
  --sync-every 0 \
  --temp-root-dir "${TASK_TEMP_DIR}" \
  --overwrite \
  --log-level INFO \
  > "${LOG_FILE}" 2>&1

rm -rf "${TASK_TEMP_DIR}" || true

echo "[$(date)] Completed m0 for ${EVENT_ID}"
