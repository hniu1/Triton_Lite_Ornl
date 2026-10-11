#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_target_audit
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o ./slurm_output/stage2-target-audit-%j.out
#SBATCH -e ./slurm_output/stage2-target-audit-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
python -m unittest discover -s tests
python -u -m stage2.audit_raw_targets \
  --output-dir "results/stage2_target_audit/job_${SLURM_JOB_ID}"
