#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_corrected_sampling
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --array=1-25%8
#SBATCH -o ./slurm_output/stage2-corrected-sampling-%A_%a.out
#SBATCH -e ./slurm_output/stage2-corrected-sampling-%A_%a.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
data_root=processed_data/stage2_corrected_training/20261009_fold0
mapfile -t train_events < "${data_root}/train_events.txt"
event_id="${train_events[$((SLURM_ARRAY_TASK_ID - 1))]}"
python -u -m data_preprocessing.m4_build_stage1_sampling_index \
  --manifest-dir "${data_root}/manifest" \
  --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
  --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
  --output-dir "${data_root}/sampling/shards/${event_id}" --event-ids "${event_id}" \
  --candidates-per-event 50000 --batch-size 8 --wet-threshold 0.05 \
  --boundary-max-fraction 0.10 --deep-threshold 1.0 --deep-min-wet-fraction 0.10 \
  --deep-depth-statistic p90 --flow-weight-fraction 0.75 --netcdf-chunk-cache-mb 512
