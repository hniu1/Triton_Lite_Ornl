#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5b_dense
#SBATCH -p gpu
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-v5b-dense-%A_%a.out
#SBATCH -e ./slurm_output/stage2-v5b-dense-%A_%a.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

variants=(baseline baseline baseline halo halo halo halo_gated_velocity halo_gated_velocity halo_gated_velocity)
seeds=(45 145 245 45 145 245 45 145 245)
run_dir="results/stage2_v5b_robustness_fold0/${variants[${SLURM_ARRAY_TASK_ID}]}/seed_${seeds[${SLURM_ARRAY_TASK_ID}]}"
threshold=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_probability_threshold"])' "${run_dir}/wet_threshold_calibration.json")
python -u -m stage2.evaluate_direct \
  --run-dir "${run_dir}" --checkpoint best_model.pt \
  --output-path "${run_dir}/test_dense_24h_calibrated_metrics.json" \
  --split test --horizons {1..48} \
  --evaluation-batches 80 --batch-size 16 --start-time-stride 12 \
  --wet-probability-threshold "${threshold}" \
  --device cuda --disable-cudnn

