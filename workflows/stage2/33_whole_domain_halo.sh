#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_halo_whole_maps
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage2-halo-whole-maps-%j.out
#SBATCH -e ./slurm_output/stage2-halo-whole-maps-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
root=results/stage2_whole_domain_halo/20261021_D005_fixed5h_flood
python -u -m stage2.whole_domain_halo --event-id D005 --batch-size 16 --device cuda \
  --plan results/stage2_whole_domain_halo/plans/D005_fixed5h_flood.json --output-dir "${root}"
python -u -m stage2.plot_whole_domain_halo --input-dir "${root}" --display-factor 4
