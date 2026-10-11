#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_corrected_rebuild
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --array=1-40%4
#SBATCH -o ./slurm_output/stage2-corrected-rebuild-%A_%a.out
#SBATCH -e ./slurm_output/stage2-corrected-rebuild-%A_%a.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
# The previous environment lost standard-library/package source files during
# array 186578. Use the intact environment verified by conversion regression.
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=4
printf -v event_id 'D%03d' "${SLURM_ARRAY_TASK_ID}"
output_dir="processed_data/stage2_corrected_rebuild/job_${SLURM_ARRAY_JOB_ID}/${event_id}"
printf 'Rebuilding %s; array %s task %s; host %s; output %s\n' \
  "${event_id}" "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)" "${output_dir}"
python -u -m data_preprocessing.corrected_event_pilot \
  --event "${event_id}" --output-dir "${output_dir}" \
  --run-scope full_event_rebuild \
  --forcing-audit results/stage2_target_audit/forcing_alignment_20261006/FORCING_ALIGNMENT.json
