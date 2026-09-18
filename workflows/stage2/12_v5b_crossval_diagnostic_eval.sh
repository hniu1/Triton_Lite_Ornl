#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_cv_eval
#SBATCH -p gpu
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-v5b-cv-eval-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v5b-cv-eval-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

fold=$((SLURM_ARRAY_TASK_ID / 2))
policy_index=$((SLURM_ARRAY_TASK_ID % 2))
policies=(learned velocity_persistence)
policy="${policies[${policy_index}]}"
if [[ "${fold}" == 0 ]]; then
  run_dir=results/stage2_v5b_robustness_fold0/halo/seed_45
else
  run_dir="results/stage2_v5b_crossval/fold_${fold}/seed_45"
fi
python -u -m stage2.evaluate_direct \
  --run-dir "${run_dir}" --checkpoint best_model.pt \
  --output-path "${run_dir}/test_diagnostic_${policy}.json" \
  --split test --horizons {1..48} --evaluation-batches 80 \
  --batch-size 16 --start-time-stride 12 \
  --extent-calibration "${run_dir}/extent_strategy_calibration.json" \
  --component-policy "${policy}" --per-event \
  --device cuda --disable-cudnn

