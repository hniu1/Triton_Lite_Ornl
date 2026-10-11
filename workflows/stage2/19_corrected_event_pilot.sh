#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_corrected_pilot
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-corrected-pilot-%j.out
#SBATCH -e ./slurm_output/stage2-corrected-pilot-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
python -m unittest discover -s tests
python -u -m data_preprocessing.corrected_event_pilot \
  --event D029 --output-dir "processed_data/stage2_corrected_pilot/job_${SLURM_JOB_ID}"
