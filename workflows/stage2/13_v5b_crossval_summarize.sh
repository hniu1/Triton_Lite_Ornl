#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_cv_summary
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o ./slurm_output/stage2-v5b-cv-summary-%j.out
#SBATCH -e ./slurm_output/stage2-v5b-cv-summary-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output
python -u -m stage2.summarize_v5b_crossval
