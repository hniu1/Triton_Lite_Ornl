#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_manifest
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-manifest-%j.out
#SBATCH -e ./slurm_output/stage1-manifest-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u data_preprocessing/m3_build_dynamic_manifest.py \
  --netcdf-dir processed_data_depth_velocity/blockwise_global/milestone_00_netcdf_v3 \
  --netcdf-pattern 'D*_ACC_future.nc' \
  --events-csv processed_data/timestamp_stage1/m1_events/events.csv \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --watershed-id conasauga \
  --component-semantics velocity \
  --skip-incomplete \
  --output-dir processed_data/timestamp_stage1/m3_dynamic_manifest
