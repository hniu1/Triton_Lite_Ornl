#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_dynamic_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage2-dynamic-eval-%A_%a.out
#SBATCH -e ./slurm_output/stage2-dynamic-eval-%A_%a.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
arms=(baseline halo halo_graph)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
if [[ "$arm" == baseline ]]; then
  run_dir=results/stage2_corrected_baseline/20261009_fold0/seed_45
else
  run_dir=results/stage2_corrected_context/20261009_fold0/${arm}/seed_45
fi
root=results/stage2_stream_flood/20261011_fold0
python -u -m stage2.evaluate_direct --run-dir "${run_dir}" \
  --cohort "${root}/cohort/cohort.json" \
  --output-path "${root}/${arm}/test_stream_flood_metrics.json" \
  --split test --batch-size 8 --per-event --component-policy learned --device cuda --disable-cudnn
