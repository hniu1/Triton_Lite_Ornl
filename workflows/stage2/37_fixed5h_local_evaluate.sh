#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_fixed5h_local_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-fixed5h-local-eval-%j.out
#SBATCH -e ./slurm_output/stage2-fixed5h-local-eval-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
root=results/stage2_fixed5h_local/20261022_fold0
for arm in B previous_halo; do
  if [[ "$arm" == B ]]; then run_dir=${root}/seed_45; else run_dir=results/stage2_corrected_context/20261009_fold0/halo/seed_45; fi
  for cohort in representative conditional; do
    extra=()
    if [[ "$cohort" == conditional ]]; then extra=(--cohort "${root}/conditional_cohort_revalidated.json"); fi
    if [[ -f "${root}/evaluation/${arm}_${cohort}.json" ]]; then continue; fi
    python -u -m stage2.evaluate_direct --run-dir "$run_dir" --split test --horizons 10 \
      --batch-size 8 --evaluation-batches 80 --start-time-stride 12 --per-event \
      --output-path "${root}/evaluation/${arm}_${cohort}.json" \
      --device cuda --disable-cudnn "${extra[@]}"
  done
done
python -u -m stage2.report_fixed5h_local
