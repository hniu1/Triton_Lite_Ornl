#!/bin/bash
# Submit only after user approval.
#SBATCH -A cli138
#SBATCH -J s2_B_D028_maps
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-B-whole-maps-%j.out
#SBATCH -e ./slurm_output/stage2-B-whole-maps-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
root=results/stage2_whole_domain_halo/20261022_D028_B_fixed5h_flood
# Preserve any previous attempt; do not silently overwrite expensive outputs.
if [[ -e "${root}/MANIFEST.json" || -e "${root}/depth_maps.nc" ]]; then
  echo "Output already exists: ${root}; review it before rerunning." >&2
  exit 1
fi
python -u -m stage2.whole_domain_halo \
  --run-dir results/stage2_fixed5h_local/20261022_fold0/seed_45 \
  --event-id D028 --batch-size 16 --device cuda \
  --plan results/stage2_whole_domain_halo/plans/D028_B_fixed5h_flood.json \
  --output-dir "${root}"
python -u -m stage2.plot_whole_domain_halo --input-dir "${root}" --display-factor 4 \
  --settings workflows/stage2/B_whole_domain_display.json --output-dir "${root}/plots"
