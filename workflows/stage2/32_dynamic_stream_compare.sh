#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_dynamic_compare
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-dynamic-compare-%j.out
#SBATCH -e ./slurm_output/stage2-dynamic-compare-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-dynamic-stream-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-dynamic-stream-cache-${SLURM_JOB_ID}"
root=results/stage2_stream_flood/20261011_fold0
python -u -m stage2.compare_corrected_context \
  --baseline results/stage2_corrected_baseline/20261009_fold0/seed_45 \
  --halo results/stage2_corrected_context/20261009_fold0/halo/seed_45 \
  --halo-graph results/stage2_corrected_context/20261009_fold0/halo_graph/seed_45 \
  --evaluation-root "${root}" --output-dir "${root}/comparison"
python -u -m stage2.plot_corrected_context --csv "${root}/comparison/comparison.csv" \
  --coverage-csv "${root}/cohort/coverage_by_lead.csv" \
  --caption 'Dynamic flood targets on stream tiles · fold 0 · seed 45 · known forcing through target
Retrospective selection · training-only 3 h change thresholds
Same cases across models at each lead; coverage varies with available origins'
