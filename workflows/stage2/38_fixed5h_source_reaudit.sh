#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_B_source_reaudit
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-8%4
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-B-source-reaudit-%A_%a.out
#SBATCH -e ./slurm_output/stage2-B-source-reaudit-%A_%a.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
root=results/stage2_fixed5h_local/20261022_fold0/source_reaudit
event_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${root}/events.txt")
test -n "${event_id}"
python -u -m stage2.prepare_dynamic_stream --root "${root}" event --event-id "${event_id}"
