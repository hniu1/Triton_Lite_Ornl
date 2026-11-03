#!/bin/bash
# One approved training job, with an in-allocation smoke check.
#SBATCH -A cli138
#SBATCH -J s2_multilead_round
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-multilead-round-%j.out
#SBATCH -e ./slurm_output/stage2-multilead-round-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
root=results/stage2_multilead_local/20261023_fold0
test ! -e "${root}/seed_45"
bash workflows/stage2/45_multilead_local_train.sh smoke
bash workflows/stage2/45_multilead_local_train.sh train "${root}/smoke_job_${SLURM_JOB_ID}"
