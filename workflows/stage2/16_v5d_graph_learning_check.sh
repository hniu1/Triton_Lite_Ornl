#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_graph_learncheck
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 02:00:00
#SBATCH -o ./slurm_output/stage2-graph-learncheck-%j.out
#SBATCH -e ./slurm_output/stage2-graph-learncheck-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_JOB_ID}"
python -m unittest discover -s tests
python -u -m stage2.graph_learning_check \
  --output-dir "results/stage2_v5d_learning_check/job_${SLURM_JOB_ID}" \
  --steps 1024 --learning-rate 0.0003 --max-loss-ratio 0.7 --seed 45
