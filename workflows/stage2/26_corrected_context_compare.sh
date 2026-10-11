#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_context_compare
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o ./slurm_output/stage2-corrected-compare-%j.out
#SBATCH -e ./slurm_output/stage2-corrected-compare-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
/ccs/home/haoranniu/miniconda3/envs/triton/bin/python -m stage2.compare_corrected_context \
  --baseline results/stage2_corrected_baseline/20261009_fold0/seed_45 \
  --halo results/stage2_corrected_context/20261009_fold0/halo/seed_45 \
  --halo-graph results/stage2_corrected_context/20261009_fold0/halo_graph/seed_45 \
  --output-dir results/stage2_corrected_context/20261009_fold0/comparison
