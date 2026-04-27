#!/bin/bash
#SBATCH -A cli138
#SBATCH -J blockwise_m3
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -o ./slurm_output/blockwise-m3-%j.out
#SBATCH -e ./slurm_output/blockwise-m3-%j.err

set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

module load cuda/11.0.2
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate triton

mkdir -p slurm_output

PYTHON_BIN=/ccs/home/haoranniu/miniconda3/envs/triton/bin/python

echo "[$(date)] Starting milestone 3 label generation"
${PYTHON_BIN} data_preprocessing/m3_construct_labels_from_netcdf.py \
  --netcdf-dir processed_data_v1/netcdf \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --blocks-file shapefiles/blocks_conasauga.shp \
  --watershed-id conasauga \
  --nc-crs EPSG:26916 \
  --blocks-crs EPSG:26916 \
  --block-id-mode watershed_b_padded \
  --events-csv processed_data/blockwise_global/milestone_01_events/events.csv \
  --blocks-parquet processed_data/blockwise_global/milestone_02_blocks/blocks.parquet \
  --output-parquet processed_data/blockwise_global/milestone_03_labels/labels.parquet \
  --log-level INFO

echo "[$(date)] Milestone 3 completed successfully"