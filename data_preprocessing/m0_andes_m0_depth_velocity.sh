#!/bin/bash
#SBATCH -A cli138
#SBATCH -J m0_dv_nc
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/m0-depth-velocity-%j.out
#SBATCH -e ./slurm_output/m0-depth-velocity-%j.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

mkdir -p slurm_output processed_data_depth_velocity/logs

PYTHON_BIN=/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python
ZIP_DIR=${ZIP_DIR:-/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr}
ZIP_PATTERN=${ZIP_PATTERN:-D*.zip}
OUTPUT_DIR=${OUTPUT_DIR:-processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3}
LOG_FILE=${LOG_FILE:-processed_data_depth_velocity/logs/m0_full_depth_velocity.log}

echo "[$(date)] Starting m0 full depth+velocity netCDF generation"
echo "ZIP_DIR=${ZIP_DIR}"
echo "ZIP_PATTERN=${ZIP_PATTERN}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

${PYTHON_BIN} -u data_preprocessing/m0_generate_netcdf_from_zip.py \
  --zip-dir "${ZIP_DIR}" \
  --zip-pattern "${ZIP_PATTERN}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-types H U V \
  --compression-level 1 \
  --sync-every 0 \
  --log-level INFO \
  > "${LOG_FILE}" 2>&1

echo "[$(date)] m0 preprocessing completed"
